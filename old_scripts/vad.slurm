#!/bin/bash
#SBATCH --job-name=vad
#SBATCH --output=logs/vad_%j.out
#SBATCH --error=logs/vad_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=erc-dupoux,erc-cristia

if [ -z "$1" ]; then
    echo "Usage: sbatch vad.slurm <DATASET>"
    echo "Output: metadata/<dataset>_<DDMMYY>/{metadata.csv, global.csv, flags/}"
    exit 1
fi

DATASET="$1"

if [ ! -d "$DATASET" ]; then
    echo "ERROR: Dataset not found: $DATASET"
    exit 1
fi

mkdir -p logs

echo "VAD Pipeline | Job: $SLURM_JOB_ID | CPUs: $SLURM_CPUS_PER_TASK"
echo "Dataset: $DATASET"
echo "Started: $(date)"

source ~/.bashrc
conda activate vad

python -u scripts/vad.py "$DATASET" -w $SLURM_CPUS_PER_TASK

echo "Completed: $(date)"
