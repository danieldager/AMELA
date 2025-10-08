#!/usr/bin/env python3
"""Create symlinks for audio files based on chunks metadata."""

from pathlib import Path

import pandas as pd


def main():
    # Paths
    chunks_csv = Path("../metadata/chunks.csv")
    source_base = Path("/store/projects/InfTrain/dataset/wav/EN")
    target_base = Path("/store/projects/lexical-benchmark/audio/symlinks")

    # Read chunks
    chunks = pd.read_csv(chunks_csv)
    print(f"Processing {len(chunks)} chunks...")

    for _, row in chunks.iterrows():
        speaker = row["unique_speaker"]
        book_id = row["book_id"]
        chunk_id = row["chunk_id"]

        # Parse chunk_id: "50h_00" -> ("50h", "00")
        chunk_dir, chunk_subdir = chunk_id.split("_")

        # Build paths
        source_dir = source_base / str(speaker) / book_id
        target_dir = target_base / chunk_dir / chunk_subdir

        # Skip if source doesn't exist
        if not source_dir.exists():
            print(f"SKIP: {source_dir} doesn't exist")
            continue

        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)

        # Symlink all files
        for source_file in source_dir.glob("*"):
            target_file = target_dir / source_file.name

            if not target_file.exists():
                target_file.symlink_to(source_file)

    print("Done!")


if __name__ == "__main__":
    main()
