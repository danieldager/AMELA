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
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io.wavfile as Wavfile
from ten_vad import TenVad


def setup_logging(log_level="INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("vad_pipeline.log"),
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


def process_single_wav(args):
    """Process a single WAV file - designed for multiprocessing."""
    wav_path, hop_size, threshold = args

    # Initialize model (each process gets its own)
    TV = TenVad(hop_size=hop_size, threshold=threshold)

    try:
        # Decompose filename (specific to our naming convention)
        parts = wav_path.stem.split("-")
        if len(parts) < 3:
            raise ValueError(f"Invalid filename format: {wav_path.stem}")

        top, _, seq = parts[0], parts[1], parts[2]
        seq_parts = seq.split("_")
        if len(seq_parts) >= 5:
            mid = seq_parts[1]
        else:
            mid = "unknown"

        # Read wav file
        sr, data = Wavfile.read(wav_path)
        duration = len(data) / sr

        # Vectorized frame extraction
        num_frames = len(data) // hop_size
        if num_frames == 0:
            raise ValueError("Audio too short for given hop size")
        frames = data[: num_frames * hop_size].reshape(-1, hop_size)

        # Pre-allocate arrays
        probs = np.empty(num_frames, dtype=np.float32)
        flags = np.empty(num_frames, dtype=np.uint8)

        process = TV.process
        for i in range(num_frames):
            probs[i], flags[i] = process(frames[i])

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
        max_workers = min(mp.cpu_count(), len(WAVS))

    logger.info(f"Using {max_workers} parallel workers for {len(WAVS)} files")

    # Prepare arguments for multiprocessing
    args_list = [(wav, hop_size, threshold) for wav in WAVS]

    results = []
    completed = 0
    errors = 0
    total = len(WAVS)
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

            if completed % 100 == 0 or completed == total:
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
        "--input-dir",
        "-i",
        default="EN_flat",
        help="Input directory containing WAV files (default: EN_flat)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="vad_results.csv",
        help="Output CSV file (default: vad_results.csv)",
    )
    parser.add_argument(
        "--hop-size",
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
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Find WAV files
    wavs = list(input_dir.rglob("*.wav"))
    if not wavs:
        logger.error(f"No WAV files found in {input_dir}")
        sys.exit(1)
    logger.info(f"Found {len(wavs)} WAV files in {input_dir}")

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
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        logger.info(f"Results saved to {args.output}")

        successful = df[df["error"].isna()] if "error" in df.columns else df
        logger.info(f"Successfully processed: {len(successful)}/{len(df)} files")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
