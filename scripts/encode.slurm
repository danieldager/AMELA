#!/bin/bash
#SBATCH --job-name=encode_array
#SBATCH --output=logs/encode_%A_%a.out
#SBATCH --error=logs/encode_%A_%a.err
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu,erc-dupoux,erc-cristia

# Validate inputs
if [ -z "$1" ]; then
    echo "Usage: sbatch encode.slurm <manifest_path> [dense_model] [quantizer] [vocab_size]"
    echo "Example: sbatch encode.slurm metadata/expresso.csv"
    echo "Example: sbatch encode.slurm metadata/expresso.csv mhubert-base-vp_mls_cv_8lang kmeans-expresso 2000"
    exit 1
fi

MANIFEST_PATH="$1"
DENSE_MODEL="${2:-mhubert-base-vp_mls_cv_8lang}"
QUANTIZER="${3:-kmeans-expresso}"
VOCAB_SIZE="${4:-2000}"

if [ ! -f "$MANIFEST_PATH" ]; then
    echo "ERROR: Manifest not found: $MANIFEST_PATH"
    exit 1
fi

mkdir -p logs

NUM_TASKS=${SLURM_ARRAY_TASK_COUNT:-1}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "=========================================="
echo "Tokenization Array Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $TASK_ID / $NUM_TASKS"
echo "Manifest: $MANIFEST_PATH"
echo "Model: $DENSE_MODEL + $QUANTIZER (vocab=$VOCAB_SIZE)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo "=========================================="

# Activate conda environment
source ~/.bashrc
conda activate textless

# Run tokenization script
python -u scripts/encode.py \
    --manifest "$MANIFEST_PATH" \
    --dense-model "$DENSE_MODEL" \
    --quantizer "$QUANTIZER" \
    --vocab-size "$VOCAB_SIZE" \
    --task-id "$TASK_ID" \
    --num-tasks "$NUM_TASKS" \
    --deduplicate \
    --device cuda

echo "=========================================="
echo "Task $TASK_ID completed: $(date)"
echo "=========================================="
