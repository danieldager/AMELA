#!/usr/bin/env python3
"""Tokenize audio files using mHuBERT + k-means with optional splitting."""

import argparse
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import soundfile as sf  # type: ignore
import torch
import torchaudio  # type: ignore

warnings.filterwarnings("ignore")

# Monkey-patching
sys.path.insert(0, str(Path(__file__).parent))
from sts import *
from textless.data.speech_encoder import SpeechEncoder  # type: ignore

FRAME_RATE = 50  # mHuBERT: 50 Hz = 20ms hop


def tokenize_manifest(
    manifest_path: str,
    dense_model: str,
    quantizer: str,
    vocab_size: int,
    task_id: int = 0,
    num_tasks: int = 1,
    deduplicate: bool = True,
    overwrite: bool = False,
    device: str = "cuda",
    use_splits: bool = True,
):
    """Tokenize audio files, with optional splitting on frame boundaries."""

    print(f"\n{'='*50}")
    print(f"Tokenization Task {task_id}/{num_tasks}")
    print(f"{'='*50}")
    print(f"Manifest: {manifest_path}")
    print(f"Model: {dense_model} + {quantizer} (vocab={vocab_size})")
    print(f"Splits: {use_splits} | Dedup: {deduplicate} | Device: {device}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}\n")

    df = pd.read_csv(manifest_path)
    if "audio_filepath" not in df.columns or "file_id" not in df.columns:
        print("ERROR: Manifest must have 'audio_filepath' and 'file_id' columns")
        return

    # Distribute files across tasks
    df = df.iloc[task_id::num_tasks].reset_index(drop=True)
    print(f"Processing {len(df)} files (task {task_id})\n")

    # Load encoder
    print("Loading encoder...")
    encoder = SpeechEncoder.by_name(
        dense_model_name=dense_model,
        quantizer_model_name=quantizer,
        vocab_size=vocab_size,
        deduplicate=deduplicate,
        need_f0=False,
    )
    if device == "cuda" and torch.cuda.is_available():
        encoder = encoder.cuda()
    print(f"Encoder on {device}\n")

    # Setup output
    manifest_path_obj = Path(manifest_path)
    dataset_name = manifest_path_obj.stem.split("_")[0]
    model_name = f"{dense_model.split('-')[0]}_{quantizer.split('-')[-1]}_{vocab_size}"
    output_dir = (
        manifest_path_obj.parent.parent / "tokens" / f"{dataset_name}_{model_name}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}\n")

    processed = 0
    skipped = 0
    errors = []
    timings = {"load": [], "encode": [], "write": []}

    # iterate over df rows
    for idx, row in df.iterrows():
        audio_path = row["audio_filepath"]
        file_id = str(row["file_id"])

        # Parse splits: handle NaN, strings, and lists
        splits = None
        if use_splits and "splits" in row and row["splits"] is not None:
            splits_val = row["splits"]
            # Check if it's NaN (float)
            if isinstance(splits_val, float):
                if splits_val != splits_val:  # NaN check
                    splits = None
            elif isinstance(splits_val, str) and splits_val.strip():
                # Parse string representation of list
                try:
                    splits = [int(x) for x in eval(splits_val)]
                except:
                    splits = None
            elif isinstance(splits_val, list):
                splits = [int(x) for x in splits_val]

        try:
            t0 = time.time()

            # Load and preprocess audio
            waveform, sr = sf.read(audio_path, dtype="float32")
            waveform = torch.from_numpy(waveform)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.ndim == 2:
                waveform = waveform.T
            if waveform.shape[0] > 1:
                waveform = waveform[0:1, :]

            t_load = time.time() - t0

            # Resample if needed
            if sr != 16000:
                waveform = torchaudio.functional.resample(
                    waveform, orig_freq=sr, new_freq=16000
                )
                t_load = time.time() - t0

            # Determine segments: either from splits or full audio
            if splits and len(splits) > 0:
                # Frame indices → sample indices
                splits_expanded = [0] + list(splits) + [waveform.shape[1]]
                segments = [
                    (splits_expanded[i], splits_expanded[i + 1])
                    for i in range(len(splits_expanded) - 1)
                ]
            else:
                segments = [(0, waveform.shape[1])]

            # Encode each segment
            for split_idx, (start_sample, end_sample) in enumerate(segments):
                output_path = output_dir / f"{file_id}_split_{split_idx}.pt"
                if not overwrite and output_path.exists():
                    skipped += 1
                    continue

                segment = waveform[:, start_sample:end_sample]
                if device == "cuda":
                    segment = segment.cuda()

                with torch.no_grad():
                    tokens = encoder(segment)["units"].cpu()

                t_encode = time.time() - t0 - t_load
                torch.save(tokens, output_path)
                processed += 1
                t_write = time.time() - t0 - t_load - t_encode

                # Track timing
                timings["load"].append(t_load)
                timings["encode"].append(t_encode)
                timings["write"].append(t_write)
                if len(timings["load"]) > 1000:
                    for k in timings:
                        timings[k] = timings[k][-1000:]

                if processed % 1000 == 0:
                    window = min(1000, len(timings["load"]))
                    avg_load_s = sum(timings["load"][-1000:]) / window
                    avg_encode_s = sum(timings["encode"][-1000:]) / window
                    print(
                        f"{idx}/{len(df)} | "
                        f"load={avg_load_s:.3f}s encode={avg_encode_s:.3f}s"
                    )

        except Exception as e:
            error_str = str(e)
            if "Cannot subsample F0" not in error_str:
                if len(errors) < 100:
                    errors.append((file_id, audio_path, error_str))
                print(f"ERROR [{file_id}]: {error_str[:100]}")

    print(f"\n{'='*50}")
    print(f"Complete: {processed} files, {skipped} skipped")
    if errors:
        print(f"Errors: {len(errors)}")
        for file_id, path, err in errors[:5]:
            print(f"  {file_id}: {err[:80]}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    print(f"Task {task_id} done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize audio files using speech encoders"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="CSV manifest with audio_filepath and file_id",
    )
    parser.add_argument(
        "--dense-model",
        type=str,
        default="mhubert-base-vp_mls_cv_8lang",
        help="Dense model name",
    )
    parser.add_argument(
        "--quantizer", type=str, default="kmeans", help="Quantizer name"
    )
    parser.add_argument("--vocab-size", type=int, default=2000, help="Vocabulary size")
    parser.add_argument("--task-id", type=int, default=0, help="Task ID for array job")
    parser.add_argument(
        "--num-tasks", type=int, default=1, help="Total number of parallel tasks"
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        default=True,
        help="Remove consecutive duplicate tokens",
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_false",
        dest="deduplicate",
        help="Keep consecutive duplicate tokens",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing token files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--no-splits",
        action="store_false",
        dest="use_splits",
        default=True,
        help="Disable splitting on 'splits' column",
    )

    args = parser.parse_args()
    tokenize_manifest(
        manifest_path=args.manifest,
        dense_model=args.dense_model,
        quantizer=args.quantizer,
        vocab_size=args.vocab_size,
        task_id=args.task_id,
        num_tasks=args.num_tasks,
        deduplicate=args.deduplicate,
        overwrite=args.overwrite,
        device=args.device,
        use_splits=args.use_splits,
    )


if __name__ == "__main__":
    main()
