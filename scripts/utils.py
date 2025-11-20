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
from datetime import datetime
from pathlib import Path

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


def save_manifest(entries: list[dict], manifest_path: str):
    """
    Save manifest to CSV or JSONL file.

    Args:
        entries: List of dict entries
        manifest_path: Path to .csv or .jsonl file
    """
    path = Path(manifest_path)

    if path.suffix in [".jsonl", ".json"]:
        with open(path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
    elif path.suffix == ".csv":
        if not entries:
            return

        # Collect all unique field names
        fieldnames = []
        for entry in entries:
            for key in entry.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(entries)
    else:
        raise ValueError(f"Unsupported manifest format: {path.suffix}")


def merge_asr_task_outputs(manifest_path: str):
    """
    Merge distributed ASR task outputs and update the original manifest.
    
    Reads intermediate JSON files from .{manifest_stem}_transcriptions/ directory,
    merges them, updates the manifest with transcriptions, and cleans up.

    Args:
        manifest_path: Path to original manifest (CSV or JSONL)
    
    Returns:
        True if successful, False otherwise
    """
    manifest_path_obj = Path(manifest_path)
    transcriptions_dir = (
        manifest_path_obj.parent / f".{manifest_path_obj.stem}_transcriptions"
    )

    if not transcriptions_dir.exists():
        print(f"ERROR: Transcriptions directory not found: {transcriptions_dir}")
        print(f"Expected: {transcriptions_dir}")
        return False

    print("=" * 60)
    print(f"Merging ASR task outputs")
    print(f"Transcriptions: {transcriptions_dir}")
    print(f"Manifest: {manifest_path}")
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

    print(f"Total transcriptions: {len(all_transcriptions)}")

    # Load manifest and update with transcriptions
    print(f"Updating manifest: {manifest_path}")
    entries = load_manifest(manifest_path)
    
    updated_count = 0
    for entry in entries:
        audio_path = entry["audio_filepath"]
        if audio_path in all_transcriptions:
            entry["text"] = all_transcriptions[audio_path]
            updated_count += 1
    
    # Save updated manifest
    save_manifest(entries, manifest_path)
    print(f"✓ Updated {updated_count} entries in manifest")

    # Clean up intermediate files
    shutil.rmtree(transcriptions_dir)
    print(f"✓ Cleaned up: {transcriptions_dir}")
    print("=" * 60)

    return True


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


def print_device_info():
    """Print GPU information if available."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("Device: CPU")
