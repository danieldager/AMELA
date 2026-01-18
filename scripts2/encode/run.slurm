#!/bin/bash
#SBATCH --job-name=encode_array
#SBATCH --output=logs/encode/encode_%A_%a.out
#SBATCH --error=logs/encode/encode_%A_%a.err
#SBATCH --array=0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=erc-dupoux,gpu-p1,gpu-p2

# Validate inputs
if [ -z "$1" ]; then
    echo "Usage: sbatch encode.slurm <manifest_path>"
    echo "Example: sbatch encode.slurm metadata/expresso.csv"
    exit 1
fi

MANIFEST_PATH="$1"

if [ ! -f "$MANIFEST_PATH" ]; then
    echo "ERROR: Manifest not found: $MANIFEST_PATH"
    exit 1
fi

# Hardcoded model configuration
MODEL="mhubert-base-vp_mls_cv_8lang/kmeans/2000"
MODEL_NAME="mhubert"

NUM_TASKS=${SLURM_ARRAY_TASK_COUNT:-1}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "=========================================="
echo "Tokenization Array Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $TASK_ID / $NUM_TASKS"
echo "Manifest: $MANIFEST_PATH"
echo "Model: $MODEL"
echo "Output name: $MODEL_NAME"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo "=========================================="

# Activate conda environment
source ~/.bashrc
conda activate textless

# Run tokenization script
python -u scripts2/encode/encode.py \
    --manifest "$MANIFEST_PATH" \
    --model "$MODEL" \
    --model-name "$MODEL_NAME" \
    --task-id "$TASK_ID" \
    --num-tasks "$NUM_TASKS" \
    --device cuda

echo "=========================================="
echo "Task $TASK_ID completed: $(date)"
echo "=========================================="
