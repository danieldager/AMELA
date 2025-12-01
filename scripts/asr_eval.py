#!/usr/bin/env python3
"""
ASR Evaluation Pipeline for LibriSpeech test.clean + test.other

Evaluates Whisper Large V3 and Canary-Qwen-2.5B on:
- Original LibriSpeech audio
- STS-resynthesized audio (via mHuBERT + HiFiGAN)

Computes WER and CER with standard normalization. Adds transcriptions and metrics
to the input manifest CSV, and saves global metrics to a separate CSV.

Usage:
    python asr_eval.py --manifest metadata/ls-clean_25-11-25.csv

Output:
    - Updates manifest with transcriptions and per-row WER/CER
    - Saves global metrics to output/{split}_global_metrics.csv
"""

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import pandas as pd  # type: ignore
import torch  # type: ignore
from jiwer import cer, wer  # type: ignore
from nemo.collections.speechlm2.models import SALM  # type: ignore
from transformers import pipeline  # type: ignore
from whisper_normalizer.english import EnglishTextNormalizer  # type: ignore

from utils import print_device_info

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("nemo_logger").setLevel(logging.WARNING)


def transcribe_whisper(whisper_model, audio_path: str) -> str:
    """
    Args:
        whisper_model: Whisper pipeline
        audio_path: Path to audio file
    """
    result = whisper_model(audio_path)
    return result["text"]


def transcribe_canary(canary_model, audio_path: str) -> str:
    """
    Args:
        canary_model: SALM model instance
        audio_path: Path to audio file
    """
    prompt = [
        {
            "role": "user",
            "content": f"Transcribe the following: {canary_model.audio_locator_tag}",
            "audio": [audio_path],
        }
    ]

    with torch.no_grad():
        answer_ids = canary_model.generate(prompts=[prompt], max_new_tokens=1000)

    text = canary_model.tokenizer.ids_to_text(answer_ids[0].cpu())
    return text


def compute_metrics(prediction: str, reference: str, normalizer) -> tuple[float, float]:
    """
    Compute WER and CER with normalization.

    Args:
        prediction: Model prediction
        reference: Ground truth
        normalizer: Text normalizer instance

    Returns:
        (wer, cer) as floats between 0 and 1
    """
    # Normalize both strings
    norm_pred = normalizer(prediction)
    norm_ref = normalizer(reference)
    return wer(norm_ref, norm_pred), cer(norm_ref, norm_pred)


def process_manifest(manifest_path: str):
    """Evaluate ASR models on original + resynthesized audio, compute metrics."""
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_types = ["raw", "syn"]
    models = ["whisper", "canary"]
    
    # Load manifest and setup paths
    df = pd.read_csv(manifest_path)
    split = Path(manifest_path).stem.split("_")[0]
    resynth_dir = Path("output") / split

    print("=" * 60)
    print(f"ASR Evaluation: {split}")
    print(f"Manifest: {manifest_path}")
    print(f"Utterances: {len(df)}")
    print(f"Resynth dir: {resynth_dir}")
    print("=" * 60)

    # Initialize result columns
    for audio_type in audio_types:
        for model in models:
            df[f"{audio_type}_{model}_transcription"] = None
            df[f"{audio_type}_{model}_wer"] = None
            df[f"{audio_type}_{model}_cer"] = None

    # Load models
    normalizer = EnglishTextNormalizer()
    
    print("\nLoading Whisper Large V3...")
    whisper_model = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        device=0 if device == "cuda" else -1,
        return_timestamps=True,  # Required for audio >30s
    )
    print("Whisper loaded...")

    print("Loading Canary-Qwen-2.5B...")
    canary_model = SALM.from_pretrained("nvidia/canary-qwen-2.5b")
    if device == "cuda":
        canary_model = canary_model.cuda()
    canary_model.eval()
    print("Canary loaded...")

    # Tracking for global metrics (weighted by reference length)
    global_stats = {
        "raw_whisper": {"total_edits": 0, "total_chars": 0, "total_wer": 0, "total_cer": 0, "count": 0},
        "raw_canary": {"total_edits": 0, "total_chars": 0, "total_wer": 0, "total_cer": 0, "count": 0},
        "syn_whisper": {"total_edits": 0, "total_chars": 0, "total_wer": 0, "total_cer": 0, "count": 0},
        "syn_canary": {"total_edits": 0, "total_chars": 0, "total_wer": 0, "total_cer": 0, "count": 0},
    }

    missing_resynth = 0
    start_time = time.time()
    progress_interval = max(1, len(df) // 20)  # Log every 5%

    for idx, row in df.iterrows():
        file_id = row["file_id"]
        reference = row["transcription"]

        if idx % progress_interval == 0:
            pct = (idx / len(df)) * 100
            print(f"[{idx+1}/{len(df)} - {pct:.0f}%]")

        # Prepare audio paths
        audio_configs = [("raw", row["audio_filepath"])]
        resynth_path = resynth_dir / f"{file_id}.flac"
        if resynth_path.exists():
            audio_configs.append(("syn", str(resynth_path)))
        else:
            print(f"  Missing resynth: {resynth_path}")
            missing_resynth += 1

        # Process each audio type
        for audio_type, audio_path in audio_configs:
            transcription_tasks = [
                ("whisper", lambda p=audio_path: transcribe_whisper(whisper_model, p)),
                ("canary", lambda p=audio_path: transcribe_canary(canary_model, p)),
            ]

            for model_name, transcribe_fn in transcription_tasks:
                try:
                    prediction = transcribe_fn()
                    wer_score, cer_score = compute_metrics(prediction, reference, normalizer)

                    # Store results
                    col_prefix = f"{audio_type}_{model_name}"
                    df.at[idx, f"{col_prefix}_transcription"] = prediction
                    df.at[idx, f"{col_prefix}_wer"] = wer_score
                    df.at[idx, f"{col_prefix}_cer"] = cer_score

                    # Accumulate weighted metrics
                    key = f"{audio_type}_{model_name}"
                    ref_words = len(reference.split())
                    ref_chars = len(reference)
                    global_stats[key]["total_wer"] += wer_score * ref_words
                    global_stats[key]["total_cer"] += cer_score * ref_chars
                    global_stats[key]["total_edits"] += ref_words
                    global_stats[key]["total_chars"] += ref_chars
                    global_stats[key]["count"] += 1

                except Exception as e:
                    print(f"  ERROR {audio_type} {model_name}: {e}")
                    continue

    # Save updated manifest
    df.to_csv(manifest_path, index=False)
    print(f"\nUpdated manifest: {manifest_path}")

    # Compute weighted global metrics
    global_metrics = []
    for key, stats in global_stats.items():
        if stats["count"] > 0:
            avg_wer = stats["total_wer"] / stats["total_edits"] if stats["total_edits"] > 0 else 0
            avg_cer = stats["total_cer"] / stats["total_chars"] if stats["total_chars"] > 0 else 0
            audio_type, model = key.split("_")
            global_metrics.append({
                "audio_type": audio_type,
                "model": model,
                "wer": round(avg_wer, 4),
                "cer": round(avg_cer, 4),
                "n_samples": stats["count"],
            })

    # Save global metrics
    metrics_df = pd.DataFrame(global_metrics)
    metrics_path = Path("output") / f"{split}_global_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Global metrics: {metrics_path}")

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Processed:       {len(df)} utterances")
    print(f"Time:            {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Missing resynth: {missing_resynth}")
    print("\nGlobal Metrics (weighted):")
    print(metrics_df.to_string(index=False))
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="ASR evaluation pipeline for LibriSpeech + STS resynthesis"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to LibriSpeech manifest (CSV)",
    )

    args = parser.parse_args()

    print_device_info()

    try:
        process_manifest(args.manifest)
    except Exception as e:
        print(f"\nERROR: Pipeline failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()