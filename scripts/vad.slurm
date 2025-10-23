#!/bin/bash
#SBATCH --job-name=vad
#SBATCH --output=logs/vad_%j.out
#SBATCH --error=logs/vad_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --partition=cpu

# Validate input
if [ -z "$1" ]; then
    echo "Usage: sbatch vad.slurm <data_dir> [output_base]"
    echo "Example: sbatch vad.slurm /path/to/audio/files"
    echo "  Creates: metadata/dirname_DD-MM-YY.csv and .json"
    echo "Example: sbatch vad.slurm /path/to/audio metadata/custom"
    echo "  Creates: metadata/custom.csv and metadata/custom.json"
    exit 1
fi

DATA_DIR="$1"
OUTPUT_BASE="${2:-}"

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi

mkdir -p logs

echo "=========================================="
echo "VAD Pipeline"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Data directory: $DATA_DIR"
echo "Output: ${OUTPUT_BASE:-auto-generated}.{csv,json}"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Started: $(date)"
echo "=========================================="

source ~/.bashrc
conda activate vad

# Build command
CMD="python -u scripts/vad.py \"$DATA_DIR\" --workers $SLURM_CPUS_PER_TASK"

if [ -n "$OUTPUT_BASE" ]; then
    CMD="$CMD --output \"$OUTPUT_BASE\""
fi

eval $CMD

echo "=========================================="
echo "VAD completed: $(date)"
echo "=========================================="
