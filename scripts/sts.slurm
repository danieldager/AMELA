#!/bin/bash
#SBATCH --job-name=sts_array
#SBATCH --output=logs/sts_%A_%a.out
#SBATCH --error=logs/sts_%A_%a.err
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu,erc-dupoux,erc-cristia

# Validate inputs
if [ -z "$1" ]; then
    echo "Usage: sbatch sts.slurm <manifest_path> [num_tasks]"
    echo "Example: sbatch sts.slurm metadata/expresso.csv"
    echo "Example (rerun subset): sbatch --array=0,1 sts.slurm metadata/expresso.csv 16"
    exit 1
fi

MANIFEST_PATH="$1"
NUM_TASKS_OVERRIDE="${2:-}"  # Optional second argument

# Validate manifest exists
if [ ! -f "$MANIFEST_PATH" ]; then
    echo "ERROR: Manifest not found: $MANIFEST_PATH"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Get total number of tasks
if [ -n "$NUM_TASKS_OVERRIDE" ]; then
    NUM_TASKS="$NUM_TASKS_OVERRIDE"
    echo "Using explicit num_tasks: $NUM_TASKS (overriding SLURM_ARRAY_TASK_COUNT)"
else
    NUM_TASKS=${SLURM_ARRAY_TASK_COUNT:-1}
fi
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "=========================================="
echo "SLURM Array Job: Speech-to-Speech Resynthesis"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Manifest: $MANIFEST_PATH"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Started: $(date)"
echo "=========================================="

# Activate conda environment
source ~/.bashrc
conda activate textless

# Run Python script
python -u scripts/sts.py \
    --manifest "$MANIFEST_PATH" \
    --task-id "$TASK_ID" \
    --num-tasks "$NUM_TASKS"

echo "=========================================="
echo "Task $TASK_ID completed: $(date)"
echo "=========================================="
