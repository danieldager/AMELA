#!/usr/bin/env python3
"""
VAD Pipeline for audio processing with multiprocessing support.
Dataset-agnostic: processes any directory structure, stores absolute paths.
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from datetime import datetime
from pathlib import Path

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import soundfile as sf  # type: ignore
import torch  # type: ignore
import torchaudio  # type: ignore
from ten_vad import TenVad  # type: ignore


def get_runs(flags, hop_size, sr, max_thresh, min_thresh, decay_rate):
    """
    Merge speech segments based on dynamic thresholding.
    Returns merged flags and the runs (start, end) for speech and silence.
    """
    if len(flags) == 0:
        return flags.copy(), np.array([]), np.array([])

    # Identify flags
    diffs = np.diff(flags)
    changes = np.flatnonzero(diffs) + 1
    boundaries = np.r_[0, changes, len(flags)]
    new_flags = flags.copy()

    # Initial state
    is_speech = new_flags[0] == 1
    speech_dur = 0.0

    # Merge speech segments with short silences
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        dur = (end - start) * hop_size / sr

        if is_speech:
            speech_dur += dur  # Continue
        else:
            thresh = max(min_thresh, max_thresh - (decay_rate * speech_dur))
            if dur < thresh:
                new_flags[start:end] = 1  # Merge
                speech_dur += dur
            else:
                speech_dur = 0.0  # Reset

        is_speech = not is_speech  # Alternate

    # Extract runs from the merged result
    diffs = np.diff(new_flags)
    changes = np.flatnonzero(diffs) + 1
    boundaries = np.r_[0, changes, len(new_flags)]
    runs = np.column_stack((boundaries[:-1], boundaries[1:]))

    if new_flags[0] == 1:
        ones, zeros = runs[::2], runs[1::2]
    else:
        ones, zeros = runs[1::2], runs[::2]

    return new_flags, ones, zeros


def runs_to_secs(runs, hop_size, sr):
    """Convert frame runs to seconds."""
    if len(runs) == 0:
        return np.array([])
    return (runs[:, 1] - runs[:, 0]) * hop_size / sr


def find_splits(zeros, total_frames, hop_size, sr, target_interval=30.0):
    """
    Find optimal split points for long audio files.

    Uses precomputed silence runs. For each target, finds silences
    overlapping the search window and picks the longest one (full duration).
    """
    if len(zeros) == 0:
        return []

    splits = []

    # Constants
    target_frames = int(target_interval * sr / hop_size)
    half_window = int(5.0 * sr / hop_size)  # +/- 5s
    min_silence_frames = int(0.3 * sr / hop_size)  # 300ms minimum

    current_start = 0

    while current_start + target_frames < total_frames:
        # Search window centered on target
        center = current_start + target_frames
        win_start = max(current_start, center - half_window)
        win_end = min(total_frames, center + half_window)

        # Find silences overlapping the window (use full duration for ranking)
        candidates = []
        for run_start, run_end in zeros:
            # Check for overlap with window
            if run_end <= win_start or run_start >= win_end:
                continue

            dur = run_end - run_start
            if dur >= min_silence_frames:
                split_point = run_start + dur // 2
                candidates.append((dur, split_point))

        if candidates:
            # Pick longest silence that touches the window
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_split = candidates[0][1]
            splits.append(best_split)
            current_start = best_split
        else:
            # No valid silence in window - find next valid silence after window
            found = False
            for run_start, run_end in zeros:
                if run_start >= win_end:
                    dur = run_end - run_start
                    if dur >= min_silence_frames:
                        split_point = run_start + dur // 2
                        splits.append(split_point)
                        current_start = split_point
                        found = True
                        break

            if not found:
                break

    return splits


def process_single_wav(
    wav_path, hop_size, threshold, merge_max, merge_min, merge_decay, flags_dir
):
    """Process a single WAV file."""
    try:
        TV = TenVad(hop_size=hop_size, threshold=threshold)

        # Read and preprocess
        data, sr = sf.read(str(wav_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)

        TARGET_SR = 16000
        if sr != TARGET_SR:
            data = torch.from_numpy(data).unsqueeze(0)
            data = torchaudio.functional.resample(data, sr, TARGET_SR)
            data = data.squeeze(0).numpy()
            sr = TARGET_SR

        # Convert to int16
        data = (data * 32767).astype(np.int16)
        duration = len(data) / sr

        num_frames = len(data) // hop_size
        if num_frames == 0:
            raise ValueError(f"Audio too short: {len(data)} samples")

        # Process VAD
        frames = data[: num_frames * hop_size].reshape(-1, hop_size)
        flags = np.empty(num_frames, dtype=np.uint8)

        process_func = TV.process
        for i in range(num_frames):
            ret, flag = process_func(frames[i])
            # Debug: catch TenVad errors
            if flag < 0 or flag > 1:
                raise RuntimeError(
                    f"TenVad returned invalid flag={flag} (ret={ret}) at frame {i}/{num_frames} | "
                    f"file={wav_path.name}, sr={sr}, duration={duration:.2f}s, "
                    f"frame_min={frames[i].min()}, frame_max={frames[i].max()}"
                )
            flags[i] = flag

        # Merge flags
        merged_flags, ones, zeros = get_runs(
            flags, hop_size, sr, merge_max, merge_min, merge_decay
        )

        # Save flags
        flags_path = flags_dir / f"{wav_path.stem}.npy"
        np.save(flags_path, merged_flags)

        # Metrics
        speech_secs = runs_to_secs(ones, hop_size, sr)
        nospch_secs = runs_to_secs(zeros, hop_size, sr)

        splits = (
            find_splits(zeros, len(merged_flags), hop_size, sr)
            if duration >= 30.0
            else []
        )

        return {
            "audio_filepath": str(wav_path),
            "duration": duration,
            "max-speech": float(speech_secs.max()) if speech_secs.size else 0.0,
            "min-speech": float(speech_secs.min()) if speech_secs.size else 0.0,
            "avg-speech": float(speech_secs.mean()) if speech_secs.size else 0.0,
            "total-speech": float(speech_secs.sum()) if speech_secs.size else 0.0,
            "count-speech": int(speech_secs.size),
            "max-nospch": float(nospch_secs.max()) if nospch_secs.size else 0.0,
            "min-nospch": float(nospch_secs.min()) if nospch_secs.size else 0.0,
            "avg-nospch": float(nospch_secs.mean()) if nospch_secs.size else 0.0,
            "total-nospch": float(nospch_secs.sum()) if nospch_secs.size else 0.0,
            "count-nospch": int(nospch_secs.size),
            "spch-ratio": float(merged_flags.mean()),
            "splits": splits,
            "flags_path": str(flags_path),
            "speech_durations": speech_secs.tolist(),
            "nospch_durations": nospch_secs.tolist(),
        }

    except Exception as e:
        return {"audio_filepath": str(wav_path), "error": str(e)}


def discover_files(dataset):
    """Generator that yields audio file paths as they're discovered."""
    EXTENSIONS = {".wav", ".flac"}
    for root, _, files in os.walk(dataset):
        for f in files:
            if os.path.splitext(f)[1].lower() in EXTENSIONS:
                yield Path(root) / f


def process_wavs_streaming(
    dataset, hop_size, threshold, max_workers, merge_params, flags_dir
):
    """Process WAV files with lazy discovery via generator."""
    merge_max, merge_min, merge_decay = merge_params

    results = []
    completed = 0
    errors = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        file_iter = discover_files(dataset)
        exhausted = False

        while not exhausted or futures:
            # Submit new tasks while we have capacity
            while len(futures) < max_workers * 2 and not exhausted:
                try:
                    wav_path = next(file_iter)
                    future = executor.submit(
                        process_single_wav,
                        wav_path,
                        hop_size,
                        threshold,
                        merge_max,
                        merge_min,
                        merge_decay,
                        flags_dir,
                    )
                    futures[future] = wav_path
                except StopIteration:
                    exhausted = True
                    break

            # Process completed tasks
            if futures:
                done, _ = wait(
                    list(futures.keys()), timeout=0.1, return_when=FIRST_COMPLETED
                )

                for future in done:
                    wav_path = futures.pop(future)
                    completed += 1

                    if completed % 2000 == 0 or (exhausted and not futures):
                        elapsed = time.time() - start_time
                        rate = completed / elapsed
                        print(
                            f"[{completed:>6}] {rate:.1f} files/s | {elapsed:.0f}s elapsed"
                        )

                    try:
                        result = future.result()
                        results.append(result)
                        if "error" in result:
                            errors += 1
                            print(
                                f"FAIL {wav_path.name}: {result['error']}",
                                file=sys.stderr,
                            )
                    except Exception as e:
                        errors += 1
                        print(f"CRASH {wav_path.name}: {e}", file=sys.stderr)
                        results.append(
                            {"audio_filepath": str(wav_path), "error": f"CRASH: {e}"}
                        )

    print(f"Completed {len(results)} files in {time.time() - start_time:.1f}s")
    if errors > 0:
        print(f"WARNING: {errors} errors encountered", file=sys.stderr)

    return results


def main():
    parser = argparse.ArgumentParser(description="VAD Pipeline for audio processing")
    parser.add_argument("dataset", type=str, help="Directory containing audio files")
    parser.add_argument(
        "--hop_size", type=int, default=256, help="Hop size (default: 256)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="VAD threshold (default: 0.5)"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=None, help="Parallel workers"
    )
    parser.add_argument(
        "--merge_max", type=float, default=0.3, help="Max merge threshold (s)"
    )
    parser.add_argument(
        "--merge_min", type=float, default=0.1, help="Min merge threshold (s)"
    )
    parser.add_argument(
        "--merge_decay", type=float, default=0.1, help="Merge decay rate"
    )
    args = parser.parse_args()

    # Validate dataset directory
    dataset = Path(args.dataset).resolve()
    if not dataset.exists():
        print(f"ERROR: Dataset does not exist: {dataset}", file=sys.stderr)
        sys.exit(1)

    # Auto-detect workers
    if args.workers is None:
        args.workers = mp.cpu_count()
    print(f"Using {args.workers} parallel workers")

    timestamp = datetime.now().strftime("%d%m%y")
    output_dir = Path("metadata") / f"{dataset.name}_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    flags_dir = output_dir / "flags"
    flags_dir.mkdir(exist_ok=True)

    durations_dir = output_dir / "durations"
    durations_dir.mkdir(exist_ok=True)

    metadata_csv = output_dir / "metadata.csv"
    global_csv = output_dir / "global.csv"
    speech_durations_npy = durations_dir / "speech_durations.npy"
    nospch_durations_npy = durations_dir / "nospch_durations.npy"

    print(f"Output directory: {output_dir}")
    print(f"Saving flags to: {flags_dir}")
    print(f"Discovering files in {dataset}...")

    results = process_wavs_streaming(
        dataset,
        args.hop_size,
        args.threshold,
        args.workers,
        (args.merge_max, args.merge_min, args.merge_decay),
        flags_dir,
    )
    df = pd.DataFrame(results)
    df.to_csv(metadata_csv, index=False)
    print(f"Per-file metadata saved to {metadata_csv}")

    # Collect all segment durations for histograms
    all_speech = []
    all_nospch = []
    for result in results:
        if "error" not in result or pd.isna(result.get("error")):
            all_speech.extend(result.get("speech_durations", []))
            all_nospch.extend(result.get("nospch_durations", []))

    if all_speech:
        np.save(speech_durations_npy, np.array(all_speech, dtype=np.float32))
        print(
            f"Speech segment durations saved to {speech_durations_npy} ({len(all_speech)} segments)"
        )

    if all_nospch:
        np.save(nospch_durations_npy, np.array(all_nospch, dtype=np.float32))
        print(
            f"Non-speech segment durations saved to {nospch_durations_npy} ({len(all_nospch)} segments)"
        )

    # Global metrics
    df = pd.DataFrame(results)
    df.to_csv(metadata_csv, index=False)
    print(f"Per-file metadata saved to {metadata_csv}")

    # Global metrics
    if "error" not in df.columns or df["error"].isna().all():
        valid_df = df
    else:
        valid_df = df[df["error"].isna()]

    if not valid_df.empty:
        total_speech_dur = valid_df["total-speech"].sum()
        total_speech_count = valid_df["count-speech"].sum()
        total_nospch_dur = valid_df["total-nospch"].sum()
        total_nospch_count = valid_df["count-nospch"].sum()
        total_duration = valid_df["duration"].sum()

        avg_speech = (
            total_speech_dur / total_speech_count if total_speech_count > 0 else 0.0
        )
        avg_nospch = (
            total_nospch_dur / total_nospch_count if total_nospch_count > 0 else 0.0
        )
        avg_file = valid_df["duration"].mean()
        speech_ratio = total_speech_dur / total_duration if total_duration > 0 else 0.0

        global_metrics = {
            "avg_speech_duration": avg_speech,
            "avg_nospch_duration": avg_nospch,
            "avg_file_duration": avg_file,
            "global_speech_ratio": speech_ratio,
        }

        pd.DataFrame([global_metrics]).to_csv(global_csv, index=False)

        print(f"\n{'='*40}")
        print(
            f"Global: {total_duration/3600:.1f}h total | {speech_ratio*100:.1f}% speech"
        )
        print(f"Avg segment: {avg_speech:.2f}s speech | {avg_nospch:.2f}s silence")
        print(f"{'='*40}")

    if "error" in df.columns:
        errors_df = df[df["error"].notna()]
        if len(errors_df) > 0:
            print(f"WARNING: {len(errors_df)} failed files", file=sys.stderr)


if __name__ == "__main__":
    main()
