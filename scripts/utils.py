#!/usr/bin/env python3
"""
Shared utility functions for AMELA pipeline scripts.
Consolidates common patterns across generate, synthesize, VAD, ASR, etc.

Import and use functions directly:
    from utils import load_manifest, distribute_tasks, print_device_info
"""

import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ==========================================
# Manifest I/O
# ==========================================


def load_manifest(manifest_path: str) -> list[dict]:
    """
    Load manifest from CSV or JSONL file.
    Automatically converts CSV to appropriate format for downstream tasks.

    Args:
        manifest_path: Path to .csv or .jsonl file

    Returns:
        List of dict entries
    """
    path = Path(manifest_path)

    if path.suffix in [".jsonl", ".json"]:
        with open(path, "r") as f:
            return [json.loads(line) for line in f]
    elif path.suffix == ".csv":
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            entries = []
            for row in reader:
                # Convert duration to float if present
                if "duration" in row:
                    row["duration"] = float(row["duration"])
                entries.append(row)
            return entries
    else:
        raise ValueError(f"Unsupported manifest format: {path.suffix}")


def csv_to_jsonl(csv_path: str, jsonl_path: str, fields: Optional[list] = None):
    """Convert CSV to JSONL, optionally selecting specific fields."""
    if fields is None:
        fields = ["audio_filepath", "duration"]

    with open(csv_path, "r") as csvfile, open(jsonl_path, "w") as jsonlfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            entry = {}
            for field in fields:
                if field in row:
                    # Convert duration to float
                    if field == "duration":
                        entry[field] = float(row[field])
                    else:
                        entry[field] = row[field]
            jsonlfile.write(json.dumps(entry) + "\n")

    print(f"Converted {csv_path} → {jsonl_path}")


def merge_transcriptions_to_csv(csv_path: str, transcriptions_path: str):
    """Merge transcriptions JSON into CSV, adding 'text' column."""
    # Read transcriptions
    with open(transcriptions_path, "r") as f:
        transcriptions = json.load(f)

    # Read CSV and add text column
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames) if reader.fieldnames else []

        # Add 'text' field if not present
        if "text" not in fieldnames:
            fieldnames.append("text")

        for row in reader:
            row["text"] = transcriptions.get(row["audio_filepath"], "")
            rows.append(row)

    # Write updated CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Merged {len(transcriptions)} transcriptions into {csv_path}")


def merge_asr_task_outputs(manifest_path: str, output_path: Optional[str] = None):
    """
    Merge distributed ASR task outputs into final JSONL manifest.

    Args:
        manifest_path: Path to original manifest (to find transcriptions dir)
        output_path: Optional output path (default: output/{dataset}.jsonl)

    Returns:
        True if successful, False otherwise
    """
    manifest_path_obj = Path(manifest_path)
    transcriptions_dir = (
        manifest_path_obj.parent / f".{manifest_path_obj.stem}_transcriptions"
    )

    if not transcriptions_dir.exists():
        print(f"ERROR: Transcriptions directory not found: {transcriptions_dir}")
        return False

    # Extract dataset name
    dataset_name = manifest_path_obj.stem.split("_")[0]

    if output_path is None:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_path = str(output_dir / f"{dataset_name}.jsonl")

    print("=" * 60)
    print(f"Merging ASR task outputs")
    print(f"Transcriptions: {transcriptions_dir}")
    print(f"Output: {output_path}")
    print("=" * 60)

    # Load all task files
    all_transcriptions = {}
    task_files = sorted(transcriptions_dir.glob("task_*.json"))

    if not task_files:
        print("ERROR: No task files found")
        return False

    print(f"Found {len(task_files)} task files")
    for task_file in task_files:
        with open(task_file) as f:
            content = f.read().strip()
            if content:
                task_data = json.loads(content)
                all_transcriptions.update(task_data)
                print(f"  {task_file.name}: {len(task_data)} entries")

    print(f"Total: {len(all_transcriptions)} transcriptions")

    # Write merged manifest
    with open(output_path, "w") as f:
        for audio_path, text in all_transcriptions.items():
            f.write(json.dumps({"audio_filepath": audio_path, "text": text}) + "\n")

    print(f"✓ Written: {output_path} ({len(all_transcriptions)} entries)")

    # Clean up
    shutil.rmtree(transcriptions_dir)
    print(f"✓ Cleaned up: {transcriptions_dir}")

    return True


# ==========================================
# Audio Processing
# ==========================================


def load_audio_mono(audio_path: str, backend: str = "soundfile"):
    """
    Load audio and convert to mono if needed.

    Args:
        audio_path: Path to audio file
        backend: 'soundfile' or 'torchaudio'

    Returns:
        Tuple of (waveform, sample_rate)
    """
    if backend == "soundfile":
        import soundfile as sf

        waveform, sr = sf.read(audio_path, dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        return waveform, sr

    elif backend == "torchaudio":
        import torchaudio

        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform[0:1, :]
        return waveform, sr
    else:
        raise ValueError(f"Unknown backend: {backend}")


def resample_to_16khz(waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
    """Resample audio to 16kHz if needed."""
    import torchaudio

    TARGET_SR = 16000

    if orig_sr != TARGET_SR:
        return torchaudio.functional.resample(
            waveform, orig_freq=orig_sr, new_freq=TARGET_SR
        )
    return waveform


# ==========================================
# Task Distribution
# ==========================================


def distribute_tasks(items: list, task_id: int, num_tasks: int) -> list:
    """Distribute items across parallel tasks using round-robin."""
    return [item for i, item in enumerate(items) if i % num_tasks == task_id]


# ==========================================
# Timestamp Utilities
# ==========================================


def timestamp_now(fmt: str = "full") -> str:
    """
    Get formatted timestamp.

    Args:
        fmt: Format type - 'full', 'time', 'date', 'iso'
    """
    formats = {
        "full": "%Y-%m-%d %H:%M:%S",
        "time": "%H:%M:%S",
        "date": "%d-%m-%y",
    }

    if fmt == "iso":
        return datetime.now().isoformat()
    return datetime.now().strftime(formats.get(fmt, formats["full"]))


# ==========================================
# Device/CUDA Setup
# ==========================================


def get_device(prefer_cuda: bool = True) -> str:
    """Get available device, preferring CUDA if available."""
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def print_device_info():
    """Print GPU information if available."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("Device: CPU")


# ==========================================
# Path Utilities
# ==========================================


def extract_dataset_name(manifest_path: str) -> str:
    """Extract dataset name from manifest filename (before first underscore)."""
    return Path(manifest_path).stem.split("_")[0]


def should_skip_existing(output_path: Path, overwrite: bool = False) -> bool:
    """Check if file should be skipped (exists and not overwriting)."""
    return output_path.exists() and not overwrite
