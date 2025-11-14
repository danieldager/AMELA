#!/bin/bash
#SBATCH --job-name=lstm_train
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
#SBATCH --array=0-0
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu,erc-dupoux,erc-cristia
#SBATCH --export=ALL

# Validate inputs
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch scripts/train.slurm <manifest_path> <tokens_dir>"
    echo "Example: sbatch scripts/train.slurm metadata/librivox_29-10-25.csv output/librivox_mhubert_expresso_2000"
    exit 1
fi

MANIFEST_PATH="$1"
TOKENS_DIR="$2"

if [ ! -f "$MANIFEST_PATH" ]; then
    echo "ERROR: Manifest not found: $MANIFEST_PATH"
    exit 1
fi

if [ ! -d "$TOKENS_DIR" ]; then
    echo "ERROR: Tokens directory not found: $TOKENS_DIR"
    exit 1
fi

mkdir -p logs

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

# Activate conda environment
source ~/.bashrc
conda activate amela_train

# ========================================
# GRID SEARCH CONFIGURATION
# ========================================
EMBEDDING_DIMS=(1024)
HIDDEN_SIZES=(2048)
NUM_LAYERS=(3)
DROPOUTS=(0.1)
BATCH_SIZES=(64)
LEARNING_RATES=(0.0003)
# ========================================

# Calculate total combinations
TOTAL_COMBINATIONS=$((
    ${#EMBEDDING_DIMS[@]} * \
    ${#HIDDEN_SIZES[@]} * \
    ${#NUM_LAYERS[@]} * \
    ${#DROPOUTS[@]} * \
    ${#BATCH_SIZES[@]} * \
    ${#LEARNING_RATES[@]}
))

# Validate array size matches total combinations
EXPECTED_ARRAY_MAX=$((TOTAL_COMBINATIONS - 1))
if [ -n "$SLURM_ARRAY_TASK_COUNT" ]; then
    if [ "$SLURM_ARRAY_TASK_COUNT" -ne "$TOTAL_COMBINATIONS" ]; then
        echo "ERROR: Array size mismatch!"
        echo "  Expected: --array=0-$EXPECTED_ARRAY_MAX (total: $TOTAL_COMBINATIONS)"
        echo "  Got:      Array count of $SLURM_ARRAY_TASK_COUNT"
        echo "  Please update the #SBATCH --array directive to match the grid search configuration"
        exit 1
    fi
fi

if [ $TASK_ID -ge $TOTAL_COMBINATIONS ]; then
    echo "ERROR: TASK_ID=$TASK_ID exceeds total combinations=$TOTAL_COMBINATIONS"
    exit 1
fi

# Map task ID to hyperparameter combination
# Use modulo arithmetic to iterate through all combinations
idx=$TASK_ID

lr_idx=$((idx % ${#LEARNING_RATES[@]}))
idx=$((idx / ${#LEARNING_RATES[@]}))

bs_idx=$((idx % ${#BATCH_SIZES[@]}))
idx=$((idx / ${#BATCH_SIZES[@]}))

dropout_idx=$((idx % ${#DROPOUTS[@]}))
idx=$((idx / ${#DROPOUTS[@]}))

layers_idx=$((idx % ${#NUM_LAYERS[@]}))
idx=$((idx / ${#NUM_LAYERS[@]}))

hidden_idx=$((idx % ${#HIDDEN_SIZES[@]}))
idx=$((idx / ${#HIDDEN_SIZES[@]}))

embed_idx=$idx

# Extract hyperparameters
EMBEDDING_DIM=${EMBEDDING_DIMS[$embed_idx]}
HIDDEN_SIZE=${HIDDEN_SIZES[$hidden_idx]}
NUM_LAYERS=${NUM_LAYERS[$layers_idx]}
DROPOUT=${DROPOUTS[$dropout_idx]}
BATCH_SIZE=${BATCH_SIZES[$bs_idx]}
LEARNING_RATE=${LEARNING_RATES[$lr_idx]}

echo "=========================================="
echo "LSTM Training (DDP)"
echo "=========================================="
echo "Job: $SLURM_JOB_ID | Task: $((TASK_ID + 1))/$TOTAL_COMBINATIONS"
echo "Model: lstm_h${HIDDEN_SIZE}_r${LEARNING_RATE}_e${EMBEDDING_DIM}l${NUM_LAYERS}_b${BATCH_SIZE}_d${DROPOUT}"
echo "Data: $(basename $MANIFEST_PATH)"
echo "Node: $SLURMD_NODENAME | GPUs: $NUM_GPUS"
echo "=========================================="

# Auto-detect precision based on GPU
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1)
if [[ "$GPU_NAME" == *"A100"* ]] || [[ "$GPU_NAME" == *"H100"* ]]; then
    PRECISION_FLAG="--use_bf16"
else
    PRECISION_FLAG="--use_fp16"
fi
echo "GPU: $GPU_NAME (${PRECISION_FLAG#--use_})"

# DDP environment variables
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONUNBUFFERED=1

echo "Launching training..."

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    scripts/train.py \
    --manifest "$MANIFEST_PATH" \
    --tokens_dir "$TOKENS_DIR" \
    --embedding_dim "$EMBEDDING_DIM" \
    --hidden_size "$HIDDEN_SIZE" \
    --num_layers "$NUM_LAYERS" \
    --dropout "$DROPOUT" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs 100 \
    --train_ratio 0.95 \
    --early_stopping 5 \
    --seed 42 \
    --gradient_accumulation_steps 4 \
    --dataloader_num_workers 4 \
    --group_by_length \
    $PRECISION_FLAG

echo "Completed: $(date) (exit code: $?)"
