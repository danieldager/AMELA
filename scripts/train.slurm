#!/bin/bash
#SBATCH --job-name=lstm_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:6
#SBATCH --ntasks-per-node=6
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

# Activate conda environment
source ~/.bashrc
conda activate amela_train

# ========================================
# HYPERPARAMETERS
# ========================================
EMBEDDING_DIM=2048
HIDDEN_SIZE=4096
NUM_LAYERS=3
DROPOUT=0.2
BATCH_SIZE=32
LEARNING_RATE=0.0003
NUM_EPOCHS=100
TRAIN_RATIO=0.95
EARLY_STOPPING=5
GRAD_ACCUM_STEPS=4
# ========================================

# Auto-detect number of GPUs and precision
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1)

# Model name for checkpoints/outputs
MODEL_NAME="lstm_r${LEARNING_RATE}_h${HIDDEN_SIZE}_e${EMBEDDING_DIM}_l${NUM_LAYERS}_b${BATCH_SIZE}_d${DROPOUT}"

echo "=========================================="
echo "LSTM Training - Job $SLURM_JOB_ID"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo ""
echo "Training: epochs=${NUM_EPOCHS}, early_stopping=${EARLY_STOPPING}"
echo "Batching: ${BATCH_SIZE}/gpu × ${GRAD_ACCUM_STEPS} accum × ${NUM_GPUS} gpus = $((BATCH_SIZE * GRAD_ACCUM_STEPS * NUM_GPUS)) effective"
echo "Hardware: ${NUM_GPUS}× $GPU_NAME @ $SLURMD_NODENAME"
echo "Data: $(basename $MANIFEST_PATH) (train_ratio=${TRAIN_RATIO})"
echo "=========================================="

# Use BF16 for A100/H100, FP16 for older GPUs
if [[ "$GPU_NAME" == *"A100"* ]] || [[ "$GPU_NAME" == *"H100"* ]]; then
    PRECISION_FLAG="--use_bf16"
    echo "Using BF16 mixed precision (A100/H100 optimized)"
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
    --num_epochs "$NUM_EPOCHS" \
    --train_ratio "$TRAIN_RATIO" \
    --early_stopping "$EARLY_STOPPING" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --dataloader_num_workers 4 \
    --group_by_length \
    --seed 42 \
    $PRECISION_FLAG

echo "Completed: $(date) (exit code: $?)"
