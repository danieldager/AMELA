#!/bin/bash
#SBATCH --job-name=lstm_train
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
#SBATCH --array=0-15  # UPDATE THIS: (total_combinations - 1)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu,erc-dupoux,erc-cristia

# Usage: sbatch train.slurm <manifest_path> <tokens_dir>
# Example: sbatch train.slurm metadata/librivox_29-10-25.csv output/librivox_mhubert_expresso_2000

# Validate inputs
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch train.slurm <manifest_path> <tokens_dir>"
    echo "Example: sbatch train.slurm metadata/librivox_29-10-25.csv output/librivox_mhubert_expresso_2000"
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

# ========================================
# GRID SEARCH CONFIGURATION
# ========================================
# Modify these arrays to customize your hyperparameter search.
# After editing, update --array=0-N above where N = (product of all array lengths) - 1
#
# Current configuration: 2*2*2*2*1*1 = 16 combinations
#
EMBEDDING_DIMS=(128 256)
HIDDEN_SIZES=(256 512)
NUM_LAYERS=(2 3)
DROPOUTS=(0.1 0.2)
BATCH_SIZES=(32)
LEARNING_RATES=(0.001)
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
echo "LSTM Training - Grid Search"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $TASK_ID / $TOTAL_COMBINATIONS"
echo "Manifest: $MANIFEST_PATH"
echo "Tokens: $TOKENS_DIR"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo "=========================================="
echo "Hyperparameters:"
echo "  embedding_dim:  $EMBEDDING_DIM"
echo "  hidden_size:    $HIDDEN_SIZE"
echo "  num_layers:     $NUM_LAYERS"
echo "  dropout:        $DROPOUT"
echo "  batch_size:     $BATCH_SIZE"
echo "  learning_rate:  $LEARNING_RATE"
echo "=========================================="

# Activate conda environment
source ~/.bashrc
conda activate textless

# Run training script
python -u scripts/train.py \
    --manifest "$MANIFEST_PATH" \
    --tokens_dir "$TOKENS_DIR" \
    --embedding_dim "$EMBEDDING_DIM" \
    --hidden_size "$HIDDEN_SIZE" \
    --num_layers "$NUM_LAYERS" \
    --dropout "$DROPOUT" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs 100 \
    --train_ratio 0.9 \
    --early_stopping 10 \
    --seed 42 \
    --device cuda

EXIT_CODE=$?

echo "=========================================="
echo "Task $TASK_ID completed: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
