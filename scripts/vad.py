#!/usr/bin/env python3
"""
VAD Pipeline for audio processing with multiprocessing support.
Dataset-agnostic: processes any directory structure, stores absolute paths.
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from ten_vad import TenVad  # type: ignore


def get_runs(flags):
    """Return (start, end) pairs for periods of speech and non-speech."""
    if len(flags) == 0:
        return np.array([]), np.array([])

    first_flag = flags[0]
    last_index = len(flags)
    arr = np.flatnonzero(np.diff(flags))
    arr = np.r_[0, arr + 1, last_index]

    pairs = np.column_stack((arr[:-1], arr[1:]))
    odd = pairs[::2]
    even = pairs[1::2]

    if first_flag == 1:
        ones = odd
        zeros = even
    else:
        ones = even
        zeros = odd

    return ones, zeros


def runs_to_secs(runs, hop_size, sr):
    """Convert runs of speech and non-speech from frames to seconds."""
    if runs is None or len(runs) == 0:
        return np.array([], dtype=np.float32)

    frame_lengths = runs[:, 1] - runs[:, 0]
    return (frame_lengths * (hop_size / sr)).astype(np.float32, copy=False)


def find_splits(flags, hop_size, sr, target_interval=30.0):
    """
    Find optimal split points for long audio files.

    Looks for non-speech runs of at least 300ms starting around target_interval,
    and places split points at the middle of suitable non-speech segments.

    Args:
        flags: Array of VAD flags (0=non-speech, 1=speech)
        hop_size: Number of samples per frame
        sr: Sample rate

    Returns:
        List of frame indices where splits should occur
    """
    splits = []

    target_interval_frames = int(target_interval * sr / hop_size)
    min_silence_frames = int(0.3 * sr / hop_size)

    # Start looking for splits after the first 30 seconds
    current_pos = target_interval_frames
    total_frames = len(flags)

    while current_pos < total_frames - target_interval_frames:
        # Look for a suitable non-speech run starting from current_pos
        split_found = False

        # Search window: look ahead up to 10 seconds for a good split point
        search_end = min(current_pos + int(10.0 * sr / hop_size), total_frames)

        i = current_pos
        while i < search_end:
            if flags[i] == 0:  # Found start of non-speech
                # Check how long this non-speech run is
                silence_start = i
                while i < total_frames and flags[i] == 0:
                    i += 1
                silence_end = i
                silence_length = silence_end - silence_start

                # If silence is long enough (>=300ms), place split in the middle
                if silence_length >= min_silence_frames:
                    split_frame = silence_start + silence_length // 2
                    splits.append(split_frame)

                    # Move to next target position (30 seconds after this split)
                    current_pos = split_frame + target_interval_frames
                    split_found = True
                    break
            else:
                i += 1

        # If no suitable split found, move forward and try again
        if not split_found:
            current_pos += int(10.0 * sr / hop_size)  # Skip ahead 10 seconds

    return splits


def process_single_wav(args):
    """Process a single WAV file - designed for multiprocessing."""
    wav_path, hop_size, threshold = args

    try:
        # Each process gets its own instance
        TV = TenVad(hop_size=hop_size, threshold=threshold)
    except Exception as e:
        return {
            "audio_filepath": str(wav_path),
            "error": f"TenVad initialization failed: {str(e)}",
        }

    try:
        # Read audio file and convert to mono if needed
        data, sr = sf.read(str(wav_path), dtype='float32')
        if len(data.shape) > 1:
            data = data.mean(axis=1)

        # Resample to 16kHz if needed (TenVAD expects 16kHz)
        TARGET_SR = 16000
        if sr != TARGET_SR:
            data_tensor = torch.from_numpy(data).unsqueeze(0)
            data_tensor = torchaudio.functional.resample(
                data_tensor, orig_freq=sr, new_freq=TARGET_SR
            )
            data = data_tensor.squeeze(0).numpy()
            sr = TARGET_SR

        # Convert to int16 for TenVAD
        data = (data * 32767).astype(np.int16)
        duration = len(data) / sr

        # Process frames
        num_frames = len(data) // hop_size
        if num_frames == 0:
            raise ValueError(f"Audio too short for hop_size {hop_size}: {len(data)} samples")
        
        frames = data[: num_frames * hop_size].reshape(-1, hop_size)
        flags = np.empty(num_frames, dtype=np.uint8)

        process_func = TV.process
        for i in range(num_frames):
            _, flags[i] = process_func(frames[i])

        spch_ratio = float(flags.mean())

        # Calculate runs and durations
        ones, zeros = get_runs(flags)
        spoken_secs = runs_to_secs(ones, hop_size, sr)
        nospch_secs = runs_to_secs(zeros, hop_size, sr)

        # Find splits for long files
        splits = find_splits(flags, hop_size, sr) if duration >= 30.0 else ""

        return {
            "audio_filepath": str(wav_path),
            "duration": duration,
            "max-spoken": float(spoken_secs.max()) if spoken_secs.size else 0.0,
            "min-spoken": float(spoken_secs.min()) if spoken_secs.size else 0.0,
            "avg-spoken": float(spoken_secs.mean()) if spoken_secs.size else 0.0,
            "max-nospch": float(nospch_secs.max()) if nospch_secs.size else 0.0,
            "min-nospch": float(nospch_secs.min()) if nospch_secs.size else 0.0,
            "avg-nospch": float(nospch_secs.mean()) if nospch_secs.size else 0.0,
            "spch-ratio": spch_ratio,
            "splits": splits,
        }

    except Exception as e:
        return {
            "audio_filepath": str(wav_path),
            "error": str(e),
        }


def process_wavs_parallel(wavs, hop_size, threshold, max_workers):
    """Process WAV files in parallel across multiple workers."""
    
    args_list = [(wav, hop_size, threshold) for wav in wavs]
    
    results = []
    completed = 0
    errors = 0
    total = len(wavs)
    hundredth = max(1, total // 100)
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_wav = {
            executor.submit(process_single_wav, args): args[0] for args in args_list
        }

        # Collect results as they complete
        for future in as_completed(future_to_wav):
            wav_path = future_to_wav[future]
            completed += 1

            if completed % hundredth == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                eta = (total - completed) / rate if rate > 0 else 0
                print(
                    f"Progress: {completed}/{total} ({completed/total*100:.1f}%) "
                    f"Rate: {rate:.1f} files/sec ETA: {eta:.0f}s"
                )

            try:
                result = future.result()
                if result is not None:
                    if "error" in result:
                        errors += 1
                        print(f"WARNING: Error processing {wav_path.name}: {result['error']}", file=sys.stderr)
                    results.append(result)
            except Exception as e:
                errors += 1
                print(f"ERROR: Exception with {wav_path}: {e}", file=sys.stderr)

    elapsed = time.time() - start_time
    print(f"Completed processing {len(results)}/{total} files in {elapsed:.1f}s")

    if errors > 0:
        print(f"WARNING: Encountered {errors} errors during processing", file=sys.stderr)

    return results


def main():
    parser = argparse.ArgumentParser(description="VAD Pipeline for audio processing")
    parser.add_argument(
        "dataset",
        type=str,
        help="Directory containing WAV files (searches recursively)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output base path (default: metadata/<dirname>_<timestamp>). Creates .csv and .json",
    )
    parser.add_argument(
        "--hop_size",
        type=int,
        default=256,
        help="Hop size for VAD processing (default: 256)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="VAD threshold (default: 0.5)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto-detect from CPUs)",
    )
    args = parser.parse_args()

    # Validate dataset directory
    dataset = Path(args.dataset).resolve()
    if not dataset.exists():
        print(f"ERROR: Dataset does not exist: {dataset}", file=sys.stderr)
        sys.exit(1)

    # Find WAV files recursively
    wavs = list(dataset.rglob("*.wav"))
    if not wavs:
        print(f"ERROR: No WAV files found in {dataset}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(wavs)} WAV files in {dataset}")

    # Auto-detect workers
    if args.workers is None:
        args.workers = mp.cpu_count()
    print(f"Using {args.workers} parallel workers")

    # Determine output files
    if args.output is None:
        timestamp = datetime.now().strftime("%d-%m-%y")
        output_dir = Path("metadata")
        output_dir.mkdir(exist_ok=True)
        dirname = dataset.name
        output_base = output_dir / f"{dirname}_{timestamp}"
    else:
        output_base = Path(args.output)
        output_base.parent.mkdir(parents=True, exist_ok=True)
    
    output_csv = output_base.with_suffix('.csv')
    output_jsonl = output_base.with_suffix('.jsonl')

    # Process files
    try:
        results = process_wavs_parallel(
            wavs,
            args.hop_size,
            args.threshold,
            args.workers,
        )

        if not results:
            print("ERROR: No results generated - all files failed processing", file=sys.stderr)
            sys.exit(1)

        df = pd.DataFrame(results)
        
        # Save full CSV
        df.to_csv(output_csv, index=False)
        print(f"Full results saved to {output_csv}")

        # Save JSONL with only audio_filepath and duration
        with open(output_jsonl, 'w') as f:
            for _, row in df.iterrows():
                if pd.isna(row.get('error')):
                    entry = {
                        "audio_filepath": row["audio_filepath"],
                        "duration": row["duration"]
                    }
                    f.write(json.dumps(entry) + '\n')
        
        print(f"JSONL manifest saved to {output_jsonl}")

        # Report statistics
        if "error" in df.columns:
            successful = df[df["error"].isna()]
            errors_df = df[df["error"].notna()]
            print(f"Successfully processed: {len(successful)}/{len(df)} files")
            if len(errors_df) > 0:
                print(f"WARNING: Failed files: {len(errors_df)}", file=sys.stderr)
                for _, row in errors_df.head(5).iterrows():
                    print(f"  {Path(row['audio_filepath']).name}: {row['error']}", file=sys.stderr)
        else:
            print(f"Successfully processed: {len(df)} files")

    except Exception as e:
        print(f"ERROR: Pipeline failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
