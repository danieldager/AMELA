#!/usr/bin/env python3
"""
Merge ASR task outputs back into original manifest.

Usage:
    python merge_manifest.py --manifest metadata/expresso.json
"""

import argparse
import json
from pathlib import Path


def merge_transcriptions(manifest_path: str):
    """Merge per-task transcription files into manifest."""
    
    manifest_path_obj = Path(manifest_path)
    transcriptions_dir = manifest_path_obj.parent / f".{manifest_path_obj.stem}_transcriptions"
    
    if not transcriptions_dir.exists():
        print(f"ERROR: No transcriptions directory found: {transcriptions_dir}")
        return
    
    # Load all task outputs
    print(f"Loading task outputs from {transcriptions_dir}")
    all_transcriptions = {}
    
    task_files = sorted(transcriptions_dir.glob("task_*.json"))
    if not task_files:
        print(f"ERROR: No task files found in {transcriptions_dir}")
        return
    
    print(f"Found {len(task_files)} task files")
    for task_file in task_files:
        with open(task_file, 'r') as f:
            content = f.read().strip()
            if content:  # Skip empty files
                task_transcriptions = json.loads(content)
                all_transcriptions.update(task_transcriptions)
    
    print(f"Loaded {len(all_transcriptions)} transcriptions")
    
    # Read original manifest
    with open(manifest_path, 'r') as f:
        entries = [json.loads(line) for line in f]
    
    # Update entries with transcriptions
    updated_count = 0
    for entry in entries:
        audio_path = entry["audio_filepath"]
        if audio_path in all_transcriptions:
            entry["text"] = all_transcriptions[audio_path]
            updated_count += 1
    
    # Write updated manifest
    print(f"Writing updated manifest with {updated_count} transcriptions")
    with open(manifest_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Manifest updated: {manifest_path}")
    print(f"Transcriptions added: {updated_count}/{len(entries)}")
    
    # Clean up transcription files
    print(f"Removing transcription directory: {transcriptions_dir}")
    for task_file in task_files:
        task_file.unlink()
    transcriptions_dir.rmdir()
    
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Merge ASR task outputs into manifest"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to JSONL manifest"
    )
    
    args = parser.parse_args()
    merge_transcriptions(args.manifest)


if __name__ == "__main__":
    main()
