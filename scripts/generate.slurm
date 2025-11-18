#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu,erc-dupoux,erc-cristia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Usage: sbatch scripts/generate.slurm <model_name> <dataset_name> [checkpoint]

set -e

# Activate environment
source ~/.bashrc
conda activate amela_train

# Arguments
MODEL="${1:?Error: model_name required}"
DATASET="${2:?Error: dataset_name required}"
CHECKPOINT="${3:-}"

# Generation parameters
TEMPERATURES="0.8,1.0,1.2"
TOP_K="50,100,None"
TOP_P="0.9,0.95,None"
NUM_SAMPLES=5
MAX_LENGTH=500

echo "=================================================="
echo "LSTM Generation: $MODEL $DATASET"
echo "=================================================="
echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "Started: $(date +%H:%M:%S)"
echo "=================================================="
echo

python -u scripts/generate.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
    --num_samples $NUM_SAMPLES \
    --temperatures "$TEMPERATURES" \
    --top_k "$TOP_K" \
    --top_p "$TOP_P" \
    --max_length $MAX_LENGTH

echo
echo "=================================================="
echo "Completed: $(date +%H:%M:%S)"
echo "=================================================="
