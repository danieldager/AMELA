#!/bin/bash
#SBATCH --job-name=gpt2_train
#SBATCH --output=logs/train/gpt2_%j.out
#SBATCH --error=logs/train/gpt2_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=erc-dupoux,gpu-p1,gpu-p2
#SBATCH --export=ALL

# Validate inputs
if [ "$1" = "--resume" ]; then
    RESUME_FLAG="--resume"
else
    RESUME_FLAG=""
fi

# Activate conda environment
source ~/.bashrc
conda activate amela_train

# ========================================
# DATASET PATHS (hardcoded)
# ========================================
# Training dataset
MANIFEST_PATH="metadata/chunks5.csv"
TOKENS_DIR="tokens/chunks5_mhubert"

# Validation dataset
VAL_MANIFEST="metadata/chunks-eval.csv"
VAL_TOKENS_DIR="tokens/chunks-eval_mhubert"
# ========================================

# ========================================
# HYPERPARAMETERS
# ========================================
# Use environment variables if set (for grid search), otherwise use defaults
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
NUM_LAYERS="${NUM_LAYERS:-12}"
DROPOUT="${DROPOUT:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
MAX_STEPS="${MAX_STEPS:-100000}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
EARLY_STOPPING="${EARLY_STOPPING:-3}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
# ========================================

# Auto-detect number of GPUs and precision
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1)

# Model name for checkpoints/outputs
MODEL_NAME="gpt2_r${LEARNING_RATE}_h${HIDDEN_SIZE}_l${NUM_LAYERS}_b${BATCH_SIZE}_d${DROPOUT}"

echo "=========================================="
echo "GPT2 Training - Job $SLURM_JOB_ID"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo ""
echo "Training: max_steps=${MAX_STEPS}, eval_steps=${EVAL_STEPS}, early_stopping=${EARLY_STOPPING}"
echo "Batching: ${BATCH_SIZE}/gpu × ${GRAD_ACCUM_STEPS} accum × ${NUM_GPUS} gpus = $((BATCH_SIZE * GRAD_ACCUM_STEPS * NUM_GPUS)) effective"
echo "Hardware: ${NUM_GPUS}× $GPU_NAME @ $SLURMD_NODENAME"
echo "Train data: $(basename $MANIFEST_PATH)"
echo "Val data: $(basename $VAL_MANIFEST)"
echo "=========================================="

echo "GPU: $GPU_NAME (BF16 precision)"

# Build resume argument if --resume flag provided
RESUME_ARG=""
if [ "$RESUME_FLAG" = "--resume" ]; then
    CHECKPOINT_DIR="checkpoints/gpt2_h${HIDDEN_SIZE}_r${LEARNING_RATE}_l${NUM_LAYERS}_b${BATCH_SIZE}_d${DROPOUT}"
    if [ -d "$CHECKPOINT_DIR" ]; then
        LATEST_CHECKPOINT=$(find "$CHECKPOINT_DIR" -type d -name "checkpoint-*" | sort -V | tail -n1)
        if [ -n "$LATEST_CHECKPOINT" ]; then
            RESUME_ARG="--resume_from_checkpoint $LATEST_CHECKPOINT"
            echo "Resuming from: $LATEST_CHECKPOINT"
        else
            echo "WARNING: --resume specified but no checkpoints found in $CHECKPOINT_DIR"
            echo "Starting training from scratch..."
        fi
    else
        echo "WARNING: --resume specified but checkpoint directory not found: $CHECKPOINT_DIR"
        echo "Starting training from scratch..."
    fi
fi

# Multi-GPU environment configuration
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "Launching training..."

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    scripts/train_gpt2.py \
    --manifest "$MANIFEST_PATH" \
    --tokens_dir "$TOKENS_DIR" \
    --val_manifest "$VAL_MANIFEST" \
    --val_tokens_dir "$VAL_TOKENS_DIR" \
    --hidden_size "$HIDDEN_SIZE" \
    --num_layers "$NUM_LAYERS" \
    --dropout "$DROPOUT" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --max_steps "$MAX_STEPS" \
    --eval_steps "$EVAL_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --warmup_steps "$WARMUP_STEPS" \
    --early_stopping "$EARLY_STOPPING" \
    --grad_accum_steps "$GRAD_ACCUM_STEPS" \
    --seed 42 \
    $RESUME_ARG

echo "Completed: $(date) (exit code: $?)"
