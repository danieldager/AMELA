#!/usr/bin/env python3
"""
Merge distributed ASR task outputs into final dataset.

Usage:
    python merge.py --type asr
"""

import argparse
import json
import shutil
from pathlib import Path


def merge_asr_transcriptions(transcriptions_dir: Path):
    """Merge per-task transcription files into final manifest."""
    
    # Extract dataset name: .expresso_transcriptions -> expresso
    dataset_name = transcriptions_dir.name[1:].replace('_transcriptions', '')
    
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{dataset_name}.jsonl"
    
    print(f"========================================")
    print(f"Merging ASR: {transcriptions_dir.name}")
    print(f"Output:      {output_path}")
    print(f"========================================")
    
    # Load all task files
    all_transcriptions = {}
    task_files = sorted(transcriptions_dir.glob("task_*.json"))
    
    if not task_files:
        print(f"ERROR: No task files found")
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
    with open(output_path, 'w') as f:
        for audio_path, text in all_transcriptions.items():
            f.write(json.dumps({"audio_filepath": audio_path, "text": text}) + '\n')
    
    print(f"✓ Written: {output_path} ({len(all_transcriptions)} entries)")
    
    # Clean up
    shutil.rmtree(transcriptions_dir)
    print(f"✓ Cleaned up: {transcriptions_dir}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Merge distributed ASR task outputs"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="asr",
        choices=["asr"],
        help="Type of merge operation (only ASR supported)",
    )
    
    args = parser.parse_args()
    
    metadata_dir = Path('metadata')
    
    # Find ASR transcription directories
    transcription_dirs = sorted([
        d for d in metadata_dir.iterdir()
        if d.is_dir() and d.name.startswith('.') and d.name.endswith('_transcriptions')
    ])
    
    if not transcription_dirs:
        print("No ASR transcription directories found in metadata/")
        print("Looking for: .{name}_transcriptions/")
        return
    
    print(f"Found {len(transcription_dirs)} ASR transcription directories\n")
    
    success_count = 0
    for trans_dir in transcription_dirs:
        if merge_asr_transcriptions(trans_dir):
            success_count += 1
        print()
    
    print(f"========================================")
    print(f"Successfully merged: {success_count}/{len(transcription_dirs)}")
    print(f"========================================")


if __name__ == "__main__":
    main()
