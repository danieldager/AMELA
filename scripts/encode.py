#!/usr/bin/env python3
"""
Tokenize audio files using speech encoders (e.g., mHuBERT + k-means).

Outputs individual .pt files (one per audio file) for parallel processing.

Usage:
    python encode.py --manifest metadata/expresso.csv --task-id 0 --num-tasks 6
"""

import argparse
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import soundfile as sf
import torch
import torchaudio

warnings.filterwarnings("ignore")

# Import compatibility fixes from sts.py
sys.path.insert(0, str(Path(__file__).parent))
from sts import *  # Imports monkey patches and encoder setup

from textless.data.speech_encoder import SpeechEncoder  # type: ignore


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
):
    """Tokenize audio files and save individual .pt files."""
    
    print(f"========================================")
    print(f"Tokenization Task {task_id}/{num_tasks}")
    print(f"========================================")
    print(f"Manifest: {manifest_path}")
    print(f"Model: {dense_model} + {quantizer} (vocab={vocab_size})")
    print(f"Deduplicate: {deduplicate}")
    print(f"Overwrite: {overwrite}")
    print(f"Device: {device}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"========================================\n")
    
    # Read manifest (CSV format)
    df = pd.read_csv(manifest_path)
    
    # Validate required columns
    if "audio_filepath" not in df.columns:
        print(f"ERROR: Manifest must have 'audio_filepath' column")
        print(f"Found columns: {list(df.columns)}")
        return
    
    if "file_id" not in df.columns:
        print(f"ERROR: Manifest must have 'file_id' column")
        print(f"Found columns: {list(df.columns)}")
        return
    
    # Slice for this task (round-robin distribution)
    df = df.iloc[task_id::num_tasks].reset_index(drop=True)
    print(f"Processing {len(df)} files (task {task_id})\n")
    
    if len(df) == 0:
        print("No files to process for this task")
        return
    
    # Initialize encoder
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
    print(f"Encoder loaded on {device}\n")
    
    # Setup output directory
    manifest_path_obj = Path(manifest_path)
    dataset_name = manifest_path_obj.stem.split('_')[0]
    dense_model_name = dense_model.split('-')[0]
    quantizer_name = quantizer.split('-')[-1]
    model_name = f"{dense_model_name}_{quantizer_name}_{vocab_size}"

    output_dir = manifest_path_obj.parent / f"{dataset_name}_{model_name}"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Output: {output_dir}/\n")
    
    # Process files and save tokens
    ENCODER_SAMPLE_RATE = 16000
    processed = 0
    skipped = 0
    errors = []
    
    # Timing diagnostics
    clock = {'load': [], 'encode': [], 'write': []}
    TIMING_WINDOW = 1000  # Keep last 1000 samples to avoid memory leak
    
    for idx, row_dict in enumerate(df.to_dict('records')):
        audio_path = row_dict["audio_filepath"]
        file_id = str(row_dict["file_id"])
        
        # Check if tokens already exist (skip if not overwriting)
        output_path = output_dir / f"{file_id}.pt"
        if not overwrite and output_path.exists():
            skipped += 1
            continue
        
        try:
            t_start = time.time()
            
            # Load audio with soundfile (faster than torchaudio for WAV)
            waveform, sr = sf.read(audio_path, dtype='float32')
            
            # Convert to torch tensor and ensure correct shape
            waveform = torch.from_numpy(waveform)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.T
            
            # Ensure mono audio (take first channel if multi-channel)
            if waveform.shape[0] > 1:
                waveform = waveform[0:1, :]
            
            t_load = time.time() - t_start
            
            # Resample to 16kHz if needed
            if sr != ENCODER_SAMPLE_RATE:
                waveform = torchaudio.functional.resample(
                    waveform, orig_freq=sr, new_freq=ENCODER_SAMPLE_RATE
                )
                t_load = time.time() - t_start
            
            # Encode: Audio → Discrete Units
            if device == "cuda":
                waveform = waveform.cuda()
            
            with torch.no_grad():
                encoded = encoder(waveform)
                tokens = encoded["units"].cpu()
            
            t_encode = time.time() - t_start - t_load
            
            # Save tokens as .pt file
            torch.save(tokens, output_path)
            
            t_write = time.time() - t_start - t_load - t_encode
            
            # Collect timing stats
            clock['load'].append(t_load)
            clock['encode'].append(t_encode)
            clock['write'].append(t_write)
            
            # Trim to window size
            if len(clock['load']) > TIMING_WINDOW:
                clock['load'] = clock['load'][-TIMING_WINDOW:]
                clock['encode'] = clock['encode'][-TIMING_WINDOW:]
                clock['write'] = clock['write'][-TIMING_WINDOW:]
            
            processed += 1
            
            x = 100
            if processed % x == 0:
                # Print timing diagnostics every 100 files
                l = f"{sum(clock['load'][-x:]) / min(x, len(clock['load'])) * 1000:.1f}"
                e = f"{sum(clock['encode'][-x:]) / min(x, len(clock['encode'])) * 1000:.1f}"
                w = f"{sum(clock['write'][-x:]) / min(x, len(clock['write'])) * 1000:.1f}"
                print(f"{idx + 1}/{len(df)} | Avg (ms): load={l}, encode={e}, write={w}")

        except Exception as e:
            # Skip F0 subsampling errors (non-critical)
            error_str = str(e)
            if "Cannot subsample F0" not in error_str:
                if len(errors) < 1000:  # Cap at 1000 errors
                    errors.append({
                        "file_id": file_id,
                        "audio_path": audio_path,
                        "error": error_str,
                    })
                print(f"ERROR [{file_id}]: {audio_path}")
                print(f"  {error_str}\n")
        
        finally:
            # Clear GPU memory periodically
            if torch.cuda.is_available() and processed % 100 == 0:
                torch.cuda.empty_cache()
    
    print(f"\n========================================")
    print(f"Processing complete: {processed}/{len(df)} successful")
    if skipped > 0:
        print(f"Skipped: {skipped} (already existed)")
    if errors:
        print(f"Errors: {len(errors)}")
        print(f"See details above")
    print(f"========================================\n")
    
    # Final error report
    if errors:
        print(f"========================================")
        print(f"ERROR REPORT ({len(errors)} failed files)")
        print(f"========================================")
        for err in errors[:10]:  # Show first 10
            print(f"  {err['file_id']}: {err['audio_path']}")
            print(f"    → {err['error']}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        print(f"========================================\n")
    
    print(f"Task {task_id} completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize audio files using speech encoders"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to CSV manifest with audio_filepath column",
    )
    parser.add_argument(
        "--dense-model",
        type=str,
        default="mhubert-base-vp_mls_cv_8lang",
        help="Dense model name (default: mhubert-base-vp_mls_cv_8lang)",
    )
    parser.add_argument(
        "--quantizer",
        type=str,
        default="kmeans-expresso",
        help="Quantizer name (default: kmeans-expresso)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=2000,
        help="Vocabulary size (default: 2000)",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=0,
        help="Task ID for array job (0-indexed)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=1,
        help="Total number of parallel tasks",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        default=True,
        help="Remove consecutive duplicate tokens (default: True)",
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
        help="Overwrite existing token files (default: False, skip existing)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)",
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
    )


if __name__ == "__main__":
    main()
