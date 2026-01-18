#!/bin/bash
#SBATCH --job-name=train_gpt2
#SBATCH --output=logs/train/gpt2_%j.out
#SBATCH --error=logs/train/gpt2_%j.err
#SBATCH --partition=erc-dupoux,gpu-p1,gpu-p2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Dataset paths
TRAIN_TOKENS="tokens/chunks5_mhubert"
VAL_TOKENS="tokens/chunks-eval_mhubert"

# Model hyperparameters
HIDDEN_SIZE=256
NUM_LAYERS=12
NUM_HEADS=4
LEARNING_RATE=0.0001
BATCH_SIZE=32
DROPOUT=0.1

# Training settings
MAX_STEPS=100000
WARMUP_STEPS=1000
EVAL_STEPS=1000
SAVE_STEPS=1000
GRAD_ACCUM_STEPS=4

# Handle optional resume flag
RESUME=""
if [[ "$1" == "--resume" ]]; then
    RESUME="--resume"
fi

# Setup environment
source ~/.bashrc
conda activate amela_train

# Extract number of GPUs from SLURM gres parameter
NUM_GPUS=$(grep "gres=gpu" "$0" | grep -oP 'gpu:\K\d+')

echo "========================================"
echo "Training started: $(date)"
echo "Job ID: $SLURM_JOB_ID | NUM_GPUS=$NUM_GPUS"
echo "========================================"
echo "Train tokens: $TRAIN_TOKENS"
echo "Val tokens: $VAL_TOKENS"
echo "Hidden size: $HIDDEN_SIZE, Layers: $NUM_LAYERS"
echo "Batch size: $BATCH_SIZE, LR: $LEARNING_RATE"
echo "========================================"
echo ""

# Set NCCL for multi-GPU
# export NCCL_DEBUG=INFO  # Uncomment to debug NCCL issues
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Run with torchrun (DDP)
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    scripts/train_simple_ddp.py \
    --train_tokens "$TRAIN_TOKENS" \
    --val_tokens "$VAL_TOKENS" \
    --hidden_size $HIDDEN_SIZE \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --dropout $DROPOUT \
    --max_steps $MAX_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --fast_debug

echo ""
echo "========================================"
echo "Training finished: $(date)"
echo "========================================"
