#!/usr/bin/env python3
"""Tokenize audio files using mHuBERT + k-means with optional splitting."""

import argparse
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, cast

import pandas as pd
import numpy as np
import torch
import torchaudio  # type: ignore
import webdataset as wds  # type: ignore

warnings.filterwarnings("ignore")

# Monkey-patching
sys.path.insert(0, str(Path(__file__).parent))
from sts import *
from textless.data.speech_encoder import SpeechEncoder  # type: ignore

# Constants
FRAME_RATE = 50  # mHuBERT: 50 Hz = 20ms hop
VAD_HOP_SIZE = 256
SAMPLE_RATE = 16000
TIMING_WINDOW = 1000
MAX_SHARD_SIZE = 1 * 1024**3  # 1GB per shard
MAX_SHARD_COUNT = 10000  # Max samples per shard


@dataclass
class Config:
    """Configuration for tokenization."""

    manifest: str  # Path to manifest CSV
    model: str  # e.g., "mhubert-base-vp_mls_cv_8lang/kmeans/2000"
    model_name: str  # e.g., "mhubert"
    task_id: int = 0
    num_tasks: int = 1
    overwrite: bool = False
    device: str = "cuda"


class ProgressTracker:
    """Track timing and summary statistics."""

    def __init__(self, window_size: int = TIMING_WINDOW):
        self.window_size = window_size
        self.timings = {"load": [], "encode": [], "write": []}
        self.processed = 0
        self.short_segments = 0  # < 3s
        self.long_segments = 0  # > 30s
        self.start_time = time.time()

    def add_timing(self, stage: str, duration: float) -> None:
        """Record timing for a stage."""
        self.timings[stage].append(duration)
        if len(self.timings[stage]) > self.window_size:
            self.timings[stage] = self.timings[stage][-self.window_size :]

    def get_avg_timing(self, stage: str) -> float:
        """Get average timing for a stage (last window)."""
        times = self.timings[stage]
        if not times:
            return 0.0
        return sum(times[-self.window_size :]) / min(self.window_size, len(times))

    def get_throughput(self) -> float:
        """Get samples per second."""
        elapsed = time.time() - self.start_time
        return self.processed / elapsed if elapsed > 0 else 0

    def get_elapsed_minutes(self) -> float:
        """Get elapsed time in minutes."""
        return (time.time() - self.start_time) / 60


# =============================================================================
# Manifest Processing
# =============================================================================


def parse_splits(splits_val: Optional[object]) -> Optional[List[int]]:
    """Parse splits column value (handle NaN, strings, and lists)."""
    if splits_val is None:
        return None

    if isinstance(splits_val, float):
        return None if splits_val != splits_val else None  # NaN check

    if isinstance(splits_val, str) and splits_val.strip():
        try:
            return [int(x) for x in eval(splits_val)]
        except Exception:
            return None

    if isinstance(splits_val, list):
        return [int(x) for x in splits_val]

    return None


def create_segments(row: pd.Series) -> List[Tuple[int, int]]:
    """Convert VAD split points into segment boundaries (in samples)."""
    total_samples = int(row.duration * SAMPLE_RATE)
    splits = row.splits

    if splits is None or not isinstance(splits, list) or len(splits) == 0:
        return [(0, total_samples)]

    # Convert VAD frame indices to audio sample indices
    boundaries = [0] + [int(s * VAD_HOP_SIZE) for s in splits] + [total_samples]
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def load_manifest(manifest_path: str) -> pd.DataFrame:
    """Load and validate manifest CSV."""
    df = pd.read_csv(manifest_path)

    # Validate required columns
    required_cols = {"file_id", "audio_filepath", "duration"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # Parse splits if present
    if "splits" in df.columns:
        df["splits"] = df["splits"].apply(parse_splits)
    else:
        df["splits"] = None

    # Create segment boundaries
    df["segments"] = df.apply(create_segments, axis=1)

    return df


# =============================================================================
# Audio Processing
# =============================================================================


def load_and_prepare_audio(audio_path: str) -> Tuple[torch.Tensor, float]:
    """Load audio file, ensure mono, and resample to target SR."""
    t0 = time.time()
    waveform, sr = torchaudio.load(audio_path)

    # Convert to mono (take first channel)
    if waveform.shape[0] > 1:
        waveform = waveform[0:1, :]

    # Resample if needed
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=SAMPLE_RATE
        )

    return waveform, time.time() - t0


# =============================================================================
# Encoding & Writing
# =============================================================================


def encode_segment(
    encoder: SpeechEncoder, segment: torch.Tensor, device: str
) -> np.ndarray:
    """Encode audio segment to tokens."""
    if device == "cuda":
        segment = segment.cuda()

    with torch.no_grad():
        tokens = encoder(segment)["units"].cpu().numpy().astype(np.int32)

    return tokens


def write_sample_to_sink(
    sink: wds.ShardWriter,  # type: ignore
    file_id: str,
    segment_id: int,
    tokens: np.ndarray,
    audio_filepath: str,
) -> None:
    """Write tokenized sample to WebDataset shard."""
    file_stem = Path(file_id).stem
    key = f"{file_stem}_s{segment_id:03d}"

    sample = {
        "__key__": key,
        "tokens.npy": tokens.astype(np.int16),
        "json": {
            "file_id": file_stem,
            "segment_id": segment_id,
            "token_count": len(tokens),
            "audio_filepath": str(audio_filepath),
        },
    }

    sink.write(sample)


# =============================================================================
# Setup Helpers
# =============================================================================


def load_encoder(config: Config) -> SpeechEncoder:
    """Load and initialize mHuBERT encoder."""
    print("Loading encoder...")
    dense_model, quantizer, vocab_size = config.model.split("/")
    encoder = SpeechEncoder.by_name(
        dense_model_name=dense_model,
        quantizer_model_name=quantizer,
        vocab_size=int(vocab_size),
        deduplicate=True,
        need_f0=False,
    )
    if config.device == "cuda" and torch.cuda.is_available():
        encoder = encoder.cuda()
    print(f"Encoder loaded on {config.device}\n")
    return encoder


def setup_writer(manifest: str, model_name: str, task_id: int) -> wds.ShardWriter:  # type: ignore
    """Create output directory and WebDataset writer."""
    manifest_path_obj = Path(manifest)
    root_path = manifest_path_obj.parent.parent.parent
    dataset_name = manifest_path_obj.stem.split("_")[0]
    output_dir = root_path / "tokens" / f"{dataset_name}_{model_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}\n")

    shard_pattern = str(output_dir / f"task{task_id:03d}-shard%03d.tar")
    return wds.ShardWriter(shard_pattern, maxsize=MAX_SHARD_SIZE, maxcount=MAX_SHARD_COUNT)  # type: ignore


def log_error(file_id: str, audio_filepath: str, message: str) -> None:
    """Log error to stderr."""
    print(f"  ERROR [{audio_filepath}]: {message}", flush=True, file=sys.stderr)


# =============================================================================
# Main Processing
# =============================================================================


def process_audio_file(
    file_id: str,
    audio_filepath: str,
    segments: List[Tuple[int, int]],
    encoder: SpeechEncoder,
    sink: wds.ShardWriter,  # type: ignore
    config: Config,
    tracker: ProgressTracker,
) -> None:
    """Process a single audio file and encode all segments."""

    # Load audio
    waveform, load_time = load_and_prepare_audio(audio_filepath)
    tracker.add_timing("load", load_time)

    if waveform.shape[1] == 0:
        log_error(file_id, audio_filepath, "Invalid audio: zero-length waveform")
        return

    # Fix last segment boundary to match actual waveform length
    if segments:
        segments = list(segments)
        segments[-1] = (segments[-1][0], waveform.shape[1])

    # Process each segment
    for segment_id, (start, end) in enumerate(segments):
        t_encode_start = time.time()
        segment = waveform[:, start:end]
        seg_duration = segment.shape[1] / SAMPLE_RATE

        # Validate segment length
        if seg_duration < 3:
            tracker.short_segments += 1
            log_error(
                file_id, audio_filepath, f"Segment {segment_id} is under 3 seconds"
            )
            continue
        if seg_duration > 30:
            tracker.long_segments += 1

        # Encode
        tokens = encode_segment(encoder, segment, config.device)
        tracker.add_timing("encode", time.time() - t_encode_start)

        if len(tokens) == 0:
            log_error(
                file_id,
                audio_filepath,
                f"Zero-length token sequence at segment {segment_id}",
            )
            continue

        # Write
        t_write_start = time.time()
        write_sample_to_sink(sink, file_id, segment_id, tokens, audio_filepath)
        tracker.add_timing("write", time.time() - t_write_start)
        tracker.processed += 1


def tokenize_manifest(config: Config) -> None:
    """Main tokenization pipeline."""

    # Load and prepare data
    df = load_manifest(config.manifest)
    df = df.iloc[config.task_id :: config.num_tasks].reset_index(drop=True)
    print(f"Processing {len(df)} files\n")

    # Setup
    tracker = ProgressTracker()
    encoder = load_encoder(config)

    with setup_writer(config.manifest, config.model_name, config.task_id) as sink:
        for counter, row in enumerate(df.itertuples(index=False)):
            file_id = str(row.file_id)
            audio_filepath = str(row.audio_filepath)
            segments = cast(List[Tuple[int, int]], row.segments)

            try:
                process_audio_file(
                    file_id=file_id,
                    audio_filepath=audio_filepath,
                    segments=segments,
                    encoder=encoder,
                    sink=sink,
                    config=config,
                    tracker=tracker,
                )
            except Exception as e:
                error_msg = str(e)
                # Skip F0 extraction errors (expected for some files)
                if "Cannot subsample F0" not in error_msg:
                    log_error(file_id, audio_filepath, error_msg[:100])

            # Progress update every 1000 files
            if counter % 1000 == 0 or counter == 100:
                rate = tracker.get_throughput()
                avg_load = tracker.get_avg_timing("load")
                avg_encode = tracker.get_avg_timing("encode")
                print(
                    f"  [{counter:5d}/{len(df)}] | "
                    f"{tracker.processed} samples | "
                    f"{rate:.1f} s/sec | "
                    f"load={avg_load:.3f}s encode={avg_encode:.3f}s",
                    flush=True,
                )

    # Print summary
    print(f"\n{'='*60}")
    print(f"Completed: {tracker.processed} samples processed")
    print(
        f"Short segments (<3s): {tracker.short_segments}, "
        f"Long segments (>30s): {tracker.long_segments}"
    )
    print(f"Task {config.task_id} finished: {datetime.now().strftime('%H:%M:%S')}")
    print(
        f"Total time: {tracker.get_elapsed_minutes():.1f} min "
        f"({tracker.get_throughput():.1f} samples/sec)"
    )
    print(f"{'='*60}\n")


# =============================================================================
# CLI Entry Point
# =============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize audio files and write to webdataset tar shards"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="CSV manifest with file_id, audio_filepath, and duration columns",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Full model name (e.g., mhubert-base-vp_mls_cv_8lang/kmeans/2000)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Custom name for output folder (e.g., mhubert)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for inference",
    )
    parser.add_argument(
        "--task-id", type=int, default=0, help="Task ID for distributed processing"
    )
    parser.add_argument(
        "--num-tasks", type=int, default=1, help="Total number of parallel tasks"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing outputs"
    )

    args = parser.parse_args()
    config = Config(**vars(args))
    tokenize_manifest(config)
