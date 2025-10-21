#!/usr/bin/env python3
"""
Split JSONL manifest into splits for parallel job array processing.

Usage:
    python split_manifest.py --input data.json --splits 30

Creates: splits/split_0000.json ... split_0029.json

Then submit with: sbatch --array=0-29 asr.slurm
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional


def split_manifest(input_path: str, num_splits: int, output_dir: Optional[str] = None):
    """
    Split a JSONL manifest into equal splits.

    Args:
        input_path: Path to input manifest
        num_splits: Number of splits to create
        output_dir: Directory to save splits (default: same as input with _splits suffix)
    """
    input_path_obj = Path(input_path)
    
    # Determine output directory
    if output_dir is None:
        output_dir_obj = input_path_obj.parent / f"{input_path_obj.stem}_splits"
    else:
        output_dir_obj = Path(output_dir)
    
    output_dir_obj.mkdir(parents=True, exist_ok=True)
    
    # Load all entries
    print(f"Reading manifest: {input_path_obj}")
    entries = []
    with open(input_path_obj, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    
    total_entries = len(entries)
    print(f"Total entries: {total_entries}")

    # Calculate split size
    split_size = (total_entries + num_splits - 1) // num_splits
    print(f"Creating {num_splits} splits of ~{split_size} entries each")

    # Write splits
    for i in range(num_splits):
        start_idx = i * split_size
        end_idx = min(start_idx + split_size, total_entries)
        split_entries = entries[start_idx:end_idx]

        if not split_entries:
            break

        split_path = output_dir_obj / f"split_{i:04d}.json"
        with open(split_path, 'w') as f:
            for entry in split_entries:
                f.write(json.dumps(entry) + '\n')

        print(f"  Split {i:04d}: {len(split_entries)} entries -> {split_path}")

    print(f"\nSplits saved to: {output_dir_obj}")
    print(f"To process with SLURM: sbatch --array=0-{num_splits-1} asr.slurm {output_dir_obj}")

    return str(output_dir_obj)


def main():
    parser = argparse.ArgumentParser(
        description="Split JSONL manifest into equal splits for parallel processing"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input manifest file"
    )
    parser.add_argument(
        "--splits",
        type=int,
        required=True,
        help="Number of splits to create"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save splits (default: <input>_splits)"
    )
    
    args = parser.parse_args()

    split_manifest(args.input, args.splits, args.output_dir)


if __name__ == "__main__":
    main()
