#!/bin/bash
#SBATCH --job-name=asr_eval
#SBATCH --output=logs/asr_eval_%j.out
#SBATCH --error=logs/asr_eval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --partition=erc-dupoux

# ASR Evaluation
#   - Transcribes audio with Whisper Large V3 + Canary-Qwen
#   - Computes WER/CER with normalization
#   - Appends global metrics to output CSV

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: sbatch asr_eval.slurm <manifest> <output_csv> <dataset_name>"
    echo "Example: sbatch asr_eval.slurm metadata/ls-clean.csv output/asr_metrics.csv ls-clean-raw"
    exit 1
fi

MANIFEST="$1"
OUTPUT_CSV="$2"
DATASET_NAME="$3"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: Manifest not found: $MANIFEST"
    exit 1
fi

echo ""
echo "ASR Evaluation"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "Manifest: $MANIFEST"
echo "Output CSV: $OUTPUT_CSV"
echo "Dataset: $DATASET_NAME"
echo "Started: $(date)"
echo ""

# Activate conda environment
source ~/.bashrc
conda activate asr

module load ffmpeg

# Run ASR evaluation
python -u scripts/asr_eval.py "$MANIFEST" "$OUTPUT_CSV" "$DATASET_NAME"

EVAL_EXIT_CODE=$?

if [ $EVAL_EXIT_CODE -ne 0 ]; then
    echo "ERROR: ASR evaluation failed with exit code $EVAL_EXIT_CODE"
    exit $EVAL_EXIT_CODE
fi

echo ""
echo "ASR evaluation completed: $(date)"
echo ""