#!/bin/bash
#SBATCH --job-name=sts_array
#SBATCH --output=logs/sts_%A_%a.out
#SBATCH --error=logs/sts_%A_%a.err
#SBATCH --array=0-6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=erc-dupoux,erc-cristia,gpu

# Validate inputs
if [ -z "$1" ]; then
    echo "Usage: sbatch sts.slurm <input_path> [output_name] [num_tasks]"
    echo "Example: sbatch sts.slurm metadata/expresso.csv"
    echo "Example: sbatch sts.slurm metadata/expresso.csv expresso_out"
    echo "Example (rerun subset): sbatch --array=0,1 sts.slurm metadata/expresso.csv expresso_out 16"
    exit 1
fi

INPUT_PATH="$1"
OUTPUT_NAME="${2:-}"         # Optional output folder name
NUM_TASKS_OVERRIDE="${3:-}"  # Optional third argument

# Validate input exists
if [ ! -f "$INPUT_PATH" ]; then
    echo "ERROR: Input not found: $INPUT_PATH"
    exit 1
fi

# Get total number of tasks
if [ -n "$NUM_TASKS_OVERRIDE" ]; then
    NUM_TASKS="$NUM_TASKS_OVERRIDE"
    echo "Using explicit num_tasks: $NUM_TASKS (overriding SLURM_ARRAY_TASK_COUNT)"
else
    NUM_TASKS=${SLURM_ARRAY_TASK_COUNT:-1}
fi
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

echo ""
echo "Speech-to-Speech Resynthesis"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES"
echo "Input: $INPUT_PATH"
echo "Started: $(date)"
echo ""

# Activate conda environment
source ~/.bashrc
conda activate textless

# Build command
CMD="python -u scripts/sts.py \"$INPUT_PATH\""
if [ -n "$OUTPUT_NAME" ]; then
    CMD="$CMD --output \"$OUTPUT_NAME\""
fi
CMD="$CMD --task-id $TASK_ID --num-tasks $NUM_TASKS"

# Run Python script
eval $CMD

echo ""
echo "Task $TASK_ID completed: $(date)"