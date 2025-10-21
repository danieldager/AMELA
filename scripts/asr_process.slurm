#!/bin/bash
#
# ASR Pipeline - Step 2: Process Splits in Parallel
# GPU required - this is the main processing job
#
#SBATCH --job-name=asr_process
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=erc-dupoux
#SBATCH --output=logs/process_%A_%a.out
#SBATCH --error=logs/process_%A_%a.err
#SBATCH --export=ALL

set -e

# Get argument
INPUT_MANIFEST="$1"

if [ -z "$INPUT_MANIFEST" ]; then
    echo "ERROR: No input manifest provided"
    exit 1
fi

# Convert to absolute path
INPUT_MANIFEST=$(realpath "$INPUT_MANIFEST")
BASENAME=$(basename "$INPUT_MANIFEST" .json)

# Set paths
SCRIPT_DIR="/scratch2/ddager/amela/scripts"
OUTPUT_DIR="/scratch2/ddager/amela/output"
LOGS_DIR="/scratch2/ddager/amela/logs"
SPLITS_DIR="/scratch2/ddager/amela/metadata/${BASENAME}_splits"

# This task processes one split
SPLIT_ID=$SLURM_ARRAY_TASK_ID
SPLIT_FILE="${SPLITS_DIR}/split_$(printf '%04d' $SPLIT_ID).json"

echo "==============================================="
echo "ASR Process - Task $SPLIT_ID"
echo "==============================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Split file: $SPLIT_FILE"
echo "==============================================="

# Verify split file exists
if [ ! -f "$SPLIT_FILE" ]; then
    echo "ERROR: Split file not found: $SPLIT_FILE"
    exit 1
fi

# Auto-detect batch size based on GPU memory
BATCH_SIZE=8
if command -v nvidia-smi &> /dev/null; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    if [ "$GPU_MEM" -gt 70000 ]; then
        BATCH_SIZE=16
    elif [ "$GPU_MEM" -gt 35000 ]; then
        BATCH_SIZE=8
    else
        BATCH_SIZE=4
    fi
    echo "GPU memory: ${GPU_MEM}MB, using batch_size=$BATCH_SIZE"
fi

# Run ASR pipeline
echo "Starting ASR processing..."
python "$SCRIPT_DIR/asr_process.py" \
    --manifest "$SPLIT_FILE" \
    --output "${OUTPUT_DIR}/split_$(printf '%04d' $SPLIT_ID).json" \
    --model "nvidia/canary-qwen-2.5b" \
    --batch_size $BATCH_SIZE \
    --max_tokens 256 \
    --device cuda \
    --log_level INFO \

echo "==============================================="
echo "Task $SPLIT_ID completed at $(date)"
echo "==============================================="
