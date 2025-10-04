#!/usr/bin/env python3
"""
Enhanced VAD Pipeline for HPC/SLURM environments
Adds command-line arguments, better logging, and progress tracking
"""

import argparse
import logging
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io.wavfile as Wavfile
from ten_vad import TenVad


def setup_logging(log_name: str, log_level="INFO"):
    """Setup logging configuration."""

    # Create log file
    logs_dir = Path("logs")
    log_file = logs_dir / f"{log_name}.log"

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    return logging.getLogger(__name__)


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


def find_splits(flags, hop_size, sr):
    """
    Find optimal split points for long audio files.

    Looks for non-speech runs of at least 300ms starting around 30-second intervals,
    and places split points at the middle of suitable non-speech segments.

    Args:
        flags: Array of VAD flags (0=non-speech, 1=speech)
        hop_size: Number of samples per frame
        sr: Sample rate

    Returns:
        List of frame indices where splits should occur
    """
    splits = []

    # Convert timing constants to frames
    target_interval_frames = int(30.0 * sr / hop_size)  # ~30 seconds in frames
    min_silence_frames = int(0.3 * sr / hop_size)  # 300ms in frames

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
        # Initialize model (each process gets its own)
        # This can fail if TEN-VAD isn't properly installed
        TV = TenVad(hop_size=hop_size, threshold=threshold)
    except Exception as e:
        return {
            "filename": wav_path.name,
            "error": f"TenVad initialization failed: {str(e)}",
            "top-file": None,
            "mid-file": None,
            "sequence": None,
            "duration": None,
            "max-spoken": None,
            "min-spoken": None,
            "avg-spoken": None,
            "max-nospch": None,
            "min-nospch": None,
            "avg-nospch": None,
            "spch-ratio": None,
            "flagged_ns": None,
            "flagged_1m": None,
            "splits": "",
        }

    try:
        # Decompose filename (specific to our naming convention)
        top, _, seq = wav_path.stem.split("-")
        mid, _, _, _, seq = seq.split("_")

        # Read wav file - handle both absolute paths and symlinks
        sr, data = Wavfile.read(str(wav_path))

        # Ensure data is in the right format (mono, int16/float32)
        if len(data.shape) > 1:
            data = data[:, 0]  # Take first channel if stereo

        duration = len(data) / sr

        # Vectorized frame extraction
        num_frames = len(data) // hop_size
        if num_frames == 0:
            raise ValueError(
                f"Audio too short for hop_size {hop_size}: {len(data)} samples"
            )
        frames = data[: num_frames * hop_size].reshape(-1, hop_size)

        # Pre-allocate arrays for better memory efficiency
        # probs = np.empty(num_frames, dtype=np.float32)
        flags = np.empty(num_frames, dtype=np.uint8)

        # Process frames - cache the process method for speed
        process_func = TV.process
        for i in range(num_frames):
            _, flags[i] = process_func(frames[i])

        # Calculate speech ratio
        spch_ratio = float(flags.mean())

        # Calculate runs and durations
        ones, zeros = get_runs(flags)
        spoken_secs = runs_to_secs(ones, hop_size, sr)
        nospch_secs = runs_to_secs(zeros, hop_size, sr)

        # Calculate statistics
        max_spoken = float(spoken_secs.max()) if spoken_secs.size else 0.0
        min_spoken = float(spoken_secs.min()) if spoken_secs.size else 0.0
        avg_spoken = float(spoken_secs.mean()) if spoken_secs.size else 0.0
        max_nospch = float(nospch_secs.max()) if nospch_secs.size else 0.0
        min_nospch = float(nospch_secs.min()) if nospch_secs.size else 0.0
        avg_nospch = float(nospch_secs.mean()) if nospch_secs.size else 0.0
        flagged_ns = bool(spch_ratio < 0.01) or bool(max_spoken < 0.3)
        flagged_1m = bool(duration >= 60.0)

        # Calculate splits for long audio files (>60 seconds)
        splits = find_splits(flags, hop_size, sr) if duration >= 60.0 else ""

        return {
            "filename": wav_path.name,
            "top-file": top,
            "mid-file": mid,
            "sequence": seq,
            "duration": duration,
            "max-spoken": max_spoken,
            "min-spoken": min_spoken,
            "avg-spoken": avg_spoken,
            "max-nospch": max_nospch,
            "min-nospch": min_nospch,
            "avg-nospch": avg_nospch,
            "spch-ratio": spch_ratio,
            "flagged_ns": flagged_ns,
            "flagged_1m": flagged_1m,
            "splits": splits,
        }

    except Exception as e:
        return {
            "filename": wav_path.name,
            "error": str(e),
            "top-file": None,
            "mid-file": None,
            "sequence": None,
            "duration": None,
            "max-spoken": None,
            "min-spoken": None,
            "avg-spoken": None,
            "max-nospch": None,
            "min-nospch": None,
            "avg-nospch": None,
            "spch-ratio": None,
            "flagged_ns": None,
            "flagged_1m": None,
            "splits": "",
        }


def process_wavs_optimized(
    WAVS, hop_size=256, threshold=0.5, max_workers=None, logger=None
):
    """
    Optimized VAD processing with multiple performance improvements.

    Args:
        WAVS: List of wav file paths
        hop_size: Hop size for processing
        threshold: VAD threshold
        max_workers: Number of processes (None for auto-detection)
        logger: Logger instance

    Returns:
        List of results dictionaries
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if max_workers is None:
        max_workers = mp.cpu_count()

    logger.info(f"Using {max_workers} parallel workers for {len(WAVS)} files")

    # Prepare arguments for multiprocessing
    args_list = [(wav, hop_size, threshold) for wav in WAVS]

    results = []
    completed = 0
    errors = 0
    total = len(WAVS)
    hundredth = max(1, total // 100)
    start_time = time.time()

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
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
                logger.info(
                    f"Progress: {completed}/{total} ({completed/total*100:.1f}%) "
                    f"Rate: {rate:.1f} files/sec ETA: {eta:.0f}s"
                )

            try:
                result = future.result()
                if result is not None:
                    if "error" in result:
                        errors += 1
                        logger.warning(
                            f"Error processing {wav_path.name}: {result['error']}"
                        )
                    results.append(result)
            except Exception as e:
                errors += 1
                logger.error(f"Exception with {wav_path}: {e}")

    elapsed = time.time() - start_time
    logger.info(f"Completed processing {len(results)}/{total} files in {elapsed:.1f}s")

    if errors > 0:
        logger.warning(f"Encountered {errors} errors during processing")

    return results


def main():
    parser = argparse.ArgumentParser(description="VAD Pipeline for audio processing")
    parser.add_argument(
        "--data_dir",
        "-i",
        default="data/EN_flat",
        help="Input directory containing WAV files (default: data/EN_flat)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output CSV file (default: auto-generated in output/ folder)",
    )
    parser.add_argument(
        "--hop_size",
        type=int,
        default=256,
        help="Hop size for VAD processing (default: 256)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="VAD threshold (default: 0.5)"
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto)",
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log_name",
        type=str,
        default=None,
        help="Log filename (default: auto-generated by slurm)",
    )
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_name, args.log_level)

    # Create output directory and filename
    if args.output is None:
        timestamp = datetime.now().strftime("%H-%d-%m")
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"vad_results_{timestamp}.csv"
    else:
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)

    # Validate input directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Input directory does not exist: {data_dir}")
        sys.exit(1)

    # Find WAV files
    wavs = list(data_dir.rglob("*.wav"))
    if not wavs:
        logger.error(f"No WAV files found in {data_dir}")
        sys.exit(1)
    logger.info(f"Found {len(wavs)} WAV files in {data_dir}")

    # Process files
    try:
        results = process_wavs_optimized(
            wavs,
            hop_size=args.hop_size,
            threshold=args.threshold,
            max_workers=args.workers,
            logger=logger,
        )

        # Save results
        if not results:
            logger.error("No results generated - all files failed processing")
            sys.exit(1)

        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")

        # Calculate success rate
        if "error" in df.columns:
            successful = df[df["error"].isna()]
            errors = df[df["error"].notna()]
            logger.info(f"Successfully processed: {len(successful)}/{len(df)} files")
            if len(errors) > 0:
                logger.warning(f"Failed files: {len(errors)}")
                # Log a few examples of errors
                for _, row in errors.head(3).iterrows():
                    logger.warning(f"  {row['filename']}: {row['error']}")
        else:
            logger.info(f"Successfully processed: {len(df)} files")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
