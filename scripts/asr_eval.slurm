#!/bin/bash
#SBATCH --job-name=asr_eval
#SBATCH --output=logs/asr_eval_%j.out
#SBATCH --error=logs/asr_eval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --partition=erc-dupoux,erc-cristia

# ASR Evaluation (canary env)
#   - Transcribes original + resynthesized audio with Whisper Large V3 + Canary-Qwen
#   - Computes WER/CER with normalization
#   - Updates manifest with transcriptions and metrics
#   - Saves global metrics to output/{split}_global_metrics.csv

if [ -z "$1" ]; then
    echo "Usage: sbatch asr_eval.slurm <manifest_path>"
    exit 1
fi

MANIFEST_PATH="$1"

if [ ! -f "$MANIFEST_PATH" ]; then
    echo "ERROR: Manifest not found: $MANIFEST_PATH"
    exit 1
fi

mkdir -p logs

echo "=========================================="
echo "ASR Evaluation Pipeline"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Manifest: $MANIFEST_PATH"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo "=========================================="

# Activate conda environment
source ~/.bashrc
conda activate asr

# Run ASR evaluation
python -u scripts/asr_eval.py --manifest "$MANIFEST_PATH"

EVAL_EXIT_CODE=$?

if [ $EVAL_EXIT_CODE -ne 0 ]; then
    echo "ERROR: ASR evaluation failed with exit code $EVAL_EXIT_CODE"
    exit $EVAL_EXIT_CODE
fi

echo ""
echo "=========================================="
echo "ASR evaluation completed: $(date)"
echo "=========================================="

