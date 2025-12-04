#!/usr/bin/env python3
"""
ASR Evaluation for audio files from a manifest.

Evaluates multiple ASR models on audio files.
Computes WER and CER with standard normalization.

Usage:
    python asr_eval.py <manifest> <output_csv> <dataset_name>

Example:
    python asr_eval.py metadata/ls-clean_25-11-25.csv output/asr_metrics.csv ls-clean-raw
    python asr_eval.py output/ls-clean/metadata.csv output/asr_metrics.csv ls-clean-syn
"""

import argparse
import gc
import logging
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd  # type: ignore
import torch  # type: ignore
from jiwer import cer, mer, wer  # type: ignore
from nemo.collections.speechlm2.models import SALM  # type: ignore
from transformers import pipeline  # type: ignore
from whisper_normalizer.english import EnglishTextNormalizer  # type: ignore

from utils import print_device_info

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("nemo_logger").setLevel(logging.WARNING)

# Batch sizes
WHISPER_BATCH_SIZE = 24
CANARY_BATCH_SIZE = 24


# ============================================================================
# Model Configurations
# ============================================================================

@dataclass
class ASRModel:
    """Configuration for an ASR model."""
    name: str
    batch_size: int
    load_fn: Callable[[], Any]
    transcribe_fn: Callable[[Any, list[str]], list[str]]


def load_whisper():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        device=0 if torch.cuda.is_available() else -1,
        return_timestamps=True,
        batch_size=WHISPER_BATCH_SIZE,
        generate_kwargs={"language": "en"},
    )


def transcribe_whisper(model, audio_paths: list[str]) -> list[str]:
    results = model(audio_paths)
    return [r["text"] for r in results]


def load_canary():
    model = SALM.from_pretrained("nvidia/canary-qwen-2.5b")
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


def transcribe_canary(model, audio_paths: list[str]) -> list[str]:
    prompts = [
        [{"role": "user", "content": f"Transcribe the following: {model.audio_locator_tag}", "audio": [path]}]
        for path in audio_paths
    ]
    with torch.inference_mode():
        answer_ids = model.generate(prompts=prompts, max_new_tokens=1000)
    return [model.tokenizer.ids_to_text(ids.cpu()) for ids in answer_ids]


# Define available models
MODELS = [
    ASRModel(name="whisper", batch_size=WHISPER_BATCH_SIZE, load_fn=load_whisper, transcribe_fn=transcribe_whisper),
    ASRModel(name="canary", batch_size=CANARY_BATCH_SIZE, load_fn=load_canary, transcribe_fn=transcribe_canary),
]


# ============================================================================
# Core Functions
# ============================================================================

def transcribe_with_model(model_config: ASRModel, audio_paths: list[str]) -> list[str]:
    """Load model, transcribe all files in batches, unload model."""
    n_samples = len(audio_paths)
    batch_size = model_config.batch_size

    print(f"\nLoading {model_config.name}...")
    model = model_config.load_fn()
    print(f"{model_config.name} loaded (batch_size={batch_size})...")

    print(f"\n[{time.strftime('%H:%M:%S')}] Transcribing with {model_config.name} ({n_samples} files)...")
    transcriptions = []
    last_pct_logged = -5

    for i in range(0, n_samples, batch_size):
        batch_paths = audio_paths[i:i + batch_size]
        try:
            batch_results = model_config.transcribe_fn(model, batch_paths)
            transcriptions.extend(batch_results)
        except Exception as e:
            print(f"  ERROR {model_config.name} batch {i}: {e}")
            transcriptions.extend([None] * len(batch_paths))

        pct = (i / n_samples) * 100
        if pct >= last_pct_logged + 5:
            print(f"[{time.strftime('%H:%M:%S')}] {model_config.name}: [{i}/{n_samples} - {pct:.0f}%]")
            last_pct_logged = pct

    print(f"[{time.strftime('%H:%M:%S')}] {model_config.name}: [{n_samples}/{n_samples} - 100%] Done")

    # Unload model to free GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"{model_config.name} unloaded, GPU memory freed")

    return transcriptions


def compute_metrics(prediction: str, reference: str, normalizer) -> tuple[float, float, float]:
    """Returns (wer, mer, cer) scores."""
    norm_pred = normalizer(prediction)
    norm_ref = normalizer(reference)
    return wer(norm_ref, norm_pred), mer(norm_ref, norm_pred), cer(norm_ref, norm_pred)


def process_manifest(manifest_path: str, output_csv: str, dataset_name: str):
    """
    Evaluate ASR models on audio files from a manifest.

    Args:
        manifest_path: Path to manifest (CSV or JSONL) with audio_filepath and transcription
        output_csv: Path to output CSV for global metrics (appends if exists)
        dataset_name: Name of the dataset (first column in output CSV)
    """
    # Load manifest as DataFrame to allow adding columns
    df = pd.read_csv(manifest_path)

    print("=" * 60)
    print(f"ASR Evaluation")
    print(f"Manifest: {manifest_path}")
    print(f"Dataset: {dataset_name}")
    print(f"Audio files: {len(df)}")
    print(f"Output CSV: {output_csv}")
    print(f"Models: {', '.join(m.name for m in MODELS)}")
    print("=" * 60)

    # Initialize result columns
    for model_config in MODELS:
        df[f"{model_config.name}_transcription"] = None
        df[f"{model_config.name}_wer_pct"] = None
        df[f"{model_config.name}_mer_pct"] = None
        df[f"{model_config.name}_cer_pct"] = None

    normalizer = EnglishTextNormalizer()
    start_time = time.time()
    audio_paths = df["audio_filepath"].tolist()
    references = df["transcription"].tolist()
    n_samples = len(df)

    # Tracking for global metrics
    global_stats = {m.name: {"total_wer": 0, "total_mer": 0, "total_cer": 0, "total_words": 0, "total_chars": 0, "count": 0} for m in MODELS}

    # Process each model sequentially (one at a time to save GPU memory)
    all_transcriptions = {}
    for model_config in MODELS:
        all_transcriptions[model_config.name] = transcribe_with_model(model_config, audio_paths)

    # Compute metrics and store results
    print(f"\n[{time.strftime('%H:%M:%S')}] Computing metrics...")
    for idx in range(n_samples):
        reference = references[idx]
        ref_words = len(reference.split())
        ref_chars = len(reference)

        for model_config in MODELS:
            prediction = all_transcriptions[model_config.name][idx]
            if prediction is None:
                continue

            try:
                wer_score, mer_score, cer_score = compute_metrics(prediction, reference, normalizer)

                # Store results in DataFrame (as percentages)
                df.at[idx, f"{model_config.name}_transcription"] = prediction
                df.at[idx, f"{model_config.name}_wer_pct"] = round(wer_score * 100, 2)
                df.at[idx, f"{model_config.name}_mer_pct"] = round(mer_score * 100, 2)
                df.at[idx, f"{model_config.name}_cer_pct"] = round(cer_score * 100, 2)

                # Accumulate weighted metrics (wer_score * ref_words = number of word errors)
                stats = global_stats[model_config.name]
                stats["total_wer"] += wer_score * ref_words
                stats["total_mer"] += mer_score * ref_words  # weight by ref_words for consistency
                stats["total_cer"] += cer_score * ref_chars
                stats["total_words"] += ref_words
                stats["total_chars"] += ref_chars
                stats["count"] += 1
            except Exception as e:
                print(f"  ERROR metrics idx={idx} {model_config.name}: {e}")

    # Save updated manifest with transcriptions
    df.to_csv(manifest_path, index=False)
    print(f"\nUpdated manifest: {manifest_path}")

    # Compute weighted global metrics
    global_metrics = []
    for model_config in MODELS:
        stats = global_stats[model_config.name]
        if stats["count"] > 0:
            avg_wer = stats["total_wer"] / stats["total_words"] if stats["total_words"] > 0 else 0
            avg_mer = stats["total_mer"] / stats["total_words"] if stats["total_words"] > 0 else 0
            avg_cer = stats["total_cer"] / stats["total_chars"] if stats["total_chars"] > 0 else 0
            global_metrics.append({
                "dataset": dataset_name,
                "model": model_config.name,
                "wer_pct": round(avg_wer * 100, 2),
                "mer_pct": round(avg_mer * 100, 2),
                "cer_pct": round(avg_cer * 100, 2),
                "n_samples": stats["count"],
            })

    metrics_df = pd.DataFrame(global_metrics)

    # Append to output CSV (create if doesn't exist)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        metrics_df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        metrics_df.to_csv(output_path, index=False)
    print(f"Metrics appended to: {output_path}")

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Dataset:      {dataset_name}")
    print(f"Processed:    {len(df)} files")
    print(f"Time:         {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("\nGlobal Metrics (weighted):")
    print(metrics_df.to_string(index=False))
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="ASR evaluation for manifest")
    parser.add_argument("manifest", help="Path to manifest (CSV or JSONL) with audio_filepath and transcription")
    parser.add_argument("output_csv", help="Path to output CSV for global metrics (appends if exists)")
    parser.add_argument("dataset_name", help="Name of the dataset (first column in output CSV)")

    args = parser.parse_args()

    print_device_info()

    try:
        process_manifest(args.manifest, args.output_csv, args.dataset_name)
    except Exception as e:
        print(f"\nERROR: Pipeline failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
