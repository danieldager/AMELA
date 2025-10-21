#!/bin/bash
#
# ASR Pipeline - Step 1: Split Manifest
# CPU-only job, no GPU needed
#
#SBATCH --job-name=asr_split
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=erc-dupoux
#SBATCH --output=logs/split_%j.out
#SBATCH --error=logs/split_%j.err
#SBATCH --export=ALL

set -e

# Get arguments
INPUT_MANIFEST="$1"
NUM_SPLITS="$2"

if [ -z "$INPUT_MANIFEST" ] || [ -z "$NUM_SPLITS" ]; then
    echo "ERROR: Usage: asr_split.slurm <manifest.json> <num_splits>"
    exit 1
fi

if [ ! -f "$INPUT_MANIFEST" ]; then
    echo "ERROR: Input manifest not found: $INPUT_MANIFEST"
    exit 1
fi

# Convert to absolute path
INPUT_MANIFEST=$(realpath "$INPUT_MANIFEST")
BASENAME=$(basename "$INPUT_MANIFEST" .json)

# Set paths
SCRIPT_DIR="/scratch2/ddager/amela/scripts"
SPLITS_DIR="/scratch2/ddager/amela/metadata/${BASENAME}_splits"

echo "==============================================="
echo "ASR Split Job"
echo "==============================================="
echo "Input: $INPUT_MANIFEST"
echo "Number of splits: $NUM_SPLITS"
echo "Output directory: $SPLITS_DIR"
echo "==============================================="

# Create splits directory
mkdir -p "$SPLITS_DIR"

# Split manifest
python "$SCRIPT_DIR/asr_split.py" \
    --input "$INPUT_MANIFEST" \
    --splits "$NUM_SPLITS" \
    --output_dir "$SPLITS_DIR"

echo "==============================================="
echo "Split complete!"
echo "Splits saved to: $SPLITS_DIR"
echo "==============================================="
