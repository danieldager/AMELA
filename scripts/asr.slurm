#!/bin/bash
#SBATCH --job-name=asr_array
#SBATCH --output=logs/asr_%A_%a.out
#SBATCH --error=logs/asr_%A_%a.err
#SBATCH --array=0-5%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu,erc-dupoux,erc-cristia

# Validate inputs
if [ -z "$1" ]; then
    echo "Usage: sbatch asr.slurm <manifest_path>"
    echo "Example: sbatch asr.slurm metadata/expresso.json"
    exit 1
fi

MANIFEST_PATH="$1"

if [ ! -f "$MANIFEST_PATH" ]; then
    echo "ERROR: Manifest not found: $MANIFEST_PATH"
    exit 1
fi

mkdir -p logs

NUM_TASKS=${SLURM_ARRAY_TASK_COUNT:-1}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "=========================================="
echo "ASR Array Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $TASK_ID / $NUM_TASKS"
echo "Manifest: $MANIFEST_PATH"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo "=========================================="

# Activate conda environment
source ~/.bashrc
conda activate canary

# Run ASR script
python -u scripts/asr.py \
    --manifest "$MANIFEST_PATH" \
    --task-id "$TASK_ID" \
    --num-tasks "$NUM_TASKS" \
    --batch-size 32 \
    --max-tokens 1000 \
    --device cuda

echo "=========================================="
echo "Task $TASK_ID completed: $(date)"
echo ""
echo "After ALL tasks finish, run merge:"
echo "  python -c \"from utils import merge_asr_task_outputs; merge_asr_task_outputs('$MANIFEST_PATH')\""
echo "=========================================="
