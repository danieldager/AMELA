#!/bin/bash
#SBATCH --job-name=synthesize
#SBATCH --output=logs/synthesize_%j.out
#SBATCH --error=logs/synthesize_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu,erc-dupoux,erc-cristia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Usage: sbatch scripts/synthesize.slurm <input_dir> [--overwrite]

set -e

# Activate STS environment
source ~/.bashrc
conda activate textless

INPUT_DIR="${1:?Error: input_dir required}"
OVERWRITE_FLAG="${2:-}"

echo "=================================================="
echo "Audio Synthesis: $INPUT_DIR"
echo "=================================================="
echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "Started: $(date +%H:%M:%S)"
echo "=================================================="
echo

python -u scripts/synthesize.py \
    --input_dir "$INPUT_DIR" \
    ${OVERWRITE_FLAG:+--overwrite}

echo
echo "=================================================="
echo "Completed: $(date +%H:%M:%S)"
echo "=================================================="
