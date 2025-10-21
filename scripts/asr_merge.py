#!/usr/bin/env python3
"""
Merge JSONL results from parallel job arrays.

Usage:
    python merge_results.py \\
        --input-pattern "output/split_*.json" \\
        --output output/final.json \\
"""

import argparse
import glob
import json
from pathlib import Path
import sys


def merge_results(input_pattern: str, output_path: str):
    """
    Merge multiple JSONL files matching a pattern.
    
    Args:
        input_pattern: Glob pattern for input files (e.g., "output/split_*.json")
        output_path: Path to output merged file
        sort_by: Optional field to sort by (e.g., "audio_filepath")
    """
    input_files = sorted([Path(f) for f in glob.glob(input_pattern)])
    
    if not input_files:
        print(f"ERROR: No files found matching pattern: {input_pattern}")
        sys.exit(1)
    
    print(f"Found {len(input_files)} files to merge:")
    for f in input_files:
        print(f"  {f}")
    
    # Collect all entries
    all_entries = []
    total_lines = 0
    
    for input_file in input_files:
        with open(input_file, 'r') as f:
            file_entries = 0
            for line in f:
                line = line.strip()
                if line:
                    all_entries.append(json.loads(line))
                    file_entries += 1
            total_lines += file_entries
            print(f"  {input_file.name}: {file_entries} entries")

    print(f"\nTotal entries: {len(all_entries)}")
    
    # Write merged file
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path_obj, 'w') as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"\nMerged results saved to: {output_path_obj}")
    print(f"Output file size: {output_path_obj.stat().st_size / 1024 / 1024:.2f} MB")

    # Now remove input files
    for input_file in input_files:
        input_file.unlink()
    
    # Check that all input files have been removed
    remaining_files = glob.glob(input_pattern)
    if not remaining_files:
        print(f"All input files removed successfully.")
    else:
        print(f"WARNING: Some input files were not removed:")
        for f in remaining_files:
            print(f"  {f}")



def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple JSONL result files into a single file"
    )
    parser.add_argument(
        "--input_pattern",
        type=str,
        required=True,
        help='Glob pattern for input files (e.g., "output/split_*.json")'
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output merged file (e.g., output/final.json)"
    )
    
    args = parser.parse_args()

    merge_results(args.input_pattern, args.output_path)


if __name__ == "__main__":
    main()
