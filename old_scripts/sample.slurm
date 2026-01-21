#!/bin/bash
#SBATCH --job-name=sample
#SBATCH --output=logs/sample_%j.out
#SBATCH --error=logs/sample_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu,erc-dupoux,erc-cristia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# End-to-end sampling pipeline: Generate → Synthesize → VAD → ASR
# Usage: sbatch scripts/sample.slurm <model_name> <dataset_name> [checkpoint]

set -e

# Validate inputs
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch scripts/sample.slurm <model_name> <dataset_name> [checkpoint]"
    echo "Example: sbatch scripts/sample.slurm lstm_h2048_r0.0003_e1024_l3_b64_d0.1 librivox"
    exit 1
fi

MODEL_NAME="$1"
DATASET_NAME="$2"
CHECKPOINT="${3:-}"

mkdir -p logs

# Generation parameters
TEMPERATURES="0.7,1.0,1.3"
TOP_K="50,100,None"
TOP_P="0.9,0.95,None"
NUM_SAMPLES=3
MAX_LENGTH=1000

# ASR parameters
ASR_BATCH_SIZE=32
ASR_MAX_TOKENS=1000

# Paths
OUTPUT_DIR="output/$MODEL_NAME/$DATASET_NAME"
TOKENS_DIR="$OUTPUT_DIR/tokens"
SPEECH_DIR="$OUTPUT_DIR/speech"
MANIFEST_CSV="$OUTPUT_DIR/manifest.csv"

echo "=========================================="
echo "End-to-End Sampling Pipeline"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME"
echo "Checkpoint: ${CHECKPOINT:-latest}"
echo "Output: $OUTPUT_DIR"
echo "Started: $(date)"
echo "=========================================="
echo

source ~/.bashrc

# Stage 1: Generate Tokens
echo "=========================================="
echo "STAGE 1: Generating Tokens"
echo "=========================================="
conda activate amela_train

python -u scripts/generate.py \
    --model "$MODEL_NAME" \
    --dataset "$DATASET_NAME" \
    ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
    --num_samples $NUM_SAMPLES \
    --temperatures "$TEMPERATURES" \
    --top_k "$TOP_K" \
    --top_p "$TOP_P" \
    --max_length $MAX_LENGTH

echo "Stage 1 complete: $(date)"
echo

# Stage 2: Synthesize Speech
echo "=========================================="
echo "STAGE 2: Synthesizing Speech"
echo "=========================================="
conda activate textless

python -u scripts/synthesize.py \
    --input_dir "$OUTPUT_DIR"

echo "Stage 2 complete: $(date)"
echo

# Stage 3: VAD Analysis
echo "=========================================="
echo "STAGE 3: VAD Analysis"
echo "=========================================="
conda activate vad

python -u scripts/vad.py \
    "$SPEECH_DIR" \
    --output "$OUTPUT_DIR/manifest" \
    --workers $SLURM_CPUS_PER_TASK

echo "Stage 3 complete: $(date)"
echo

# Stage 4: ASR Transcription
echo "=========================================="
echo "STAGE 4: ASR Transcription"
echo "=========================================="
conda activate canary

# Run ASR (updates manifest in-place)
python -u scripts/asr.py \
    --manifest "$OUTPUT_DIR/manifest.csv" \
    --batch_size $ASR_BATCH_SIZE \
    --max_tokens $ASR_MAX_TOKENS

echo "Stage 4 complete: $(date)"
echo

# Summary
echo "=========================================="
echo "Pipeline Complete"
echo "=========================================="
echo "Tokens: $TOKENS_DIR"
echo "Speech: $SPEECH_DIR"
echo "Manifest: $MANIFEST_CSV"
echo "Completed: $(date)"
echo "=========================================="
