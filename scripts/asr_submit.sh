#!/bin/bash
#
# ASR Pipeline Submission Script
#
# Submits a complete ASR workflow with job dependencies:
#   1. Split manifest (CPU-only)
#   2. Process splits in parallel (GPU array job)
#   3. Merge results (CPU-only)
#
# Usage:
#   ./submit_asr.sh <manifest.json> <num_gpus>
#
# Example:
#   ./submit_asr.sh metadata/my_data.json 30
#

set -e

# Check arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <manifest.json> <num_gpus>"
    echo ""
    echo "Example:"
    echo "  $0 metadata/my_data.json 30"
    echo ""
    echo "This will:"
    echo "  - Split manifest into 30 splits"
    echo "  - Process all 30 splits in parallel on GPUs"
    echo "  - Merge results into output/<basename>_results.json"
    exit 1
fi

INPUT_MANIFEST="$1"
NUM_GPUS="$2"

# Validate inputs
if [ ! -f "$INPUT_MANIFEST" ]; then
    echo "ERROR: Input manifest not found: $INPUT_MANIFEST"
    exit 1
fi

if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [ "$NUM_GPUS" -lt 1 ]; then
    echo "ERROR: NUM_GPUS must be a positive integer"
    exit 1
fi

BASENAME=$(basename "$INPUT_MANIFEST" .json)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==============================================="
echo "ASR Pipeline Submission"
echo "==============================================="
echo "GPUs:      $NUM_GPUS"
echo "Input:     $INPUT_MANIFEST"
echo "Output:    output/${BASENAME}.json"
echo "==============================================="
echo ""

# Step 1: Submit split job (CPU-only, no GPU)
echo "Step 1: Submitting SPLIT job..."
SPLIT_JOB_ID=$(sbatch --parsable \
    "$SCRIPT_DIR/asr_split.slurm" \
    "$INPUT_MANIFEST" \
    "$NUM_GPUS")

echo "  → Split Job ID: $SPLIT_JOB_ID"
echo ""

# Step 2: Submit processing array job (GPU required)
# Wait for split job to complete successfully
echo "Step 2: Submitting PROCESS job array (${NUM_GPUS} tasks)..."
PROCESS_JOB_ID=$(sbatch --parsable \
    --array=0-$((NUM_GPUS - 1)) \
    --dependency=afterok:$SPLIT_JOB_ID \
    "$SCRIPT_DIR/asr_process.slurm" \
    "$INPUT_MANIFEST")

echo "  → Process Array Job ID: $PROCESS_JOB_ID"
echo "  → Tasks: 0-$((NUM_GPUS - 1))"
echo ""

# Step 3: Submit merge job (CPU-only, no GPU)
# Wait for all processing tasks to complete successfully
echo "Step 3: Submitting MERGE job..."
MERGE_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:$PROCESS_JOB_ID \
    "$SCRIPT_DIR/asr_merge.slurm" \
    "$INPUT_MANIFEST")

echo "  → Merge Job ID: $MERGE_JOB_ID"
echo ""

echo "==============================================="
echo "Workflow submitted successfully!"
echo "==============================================="
echo ""
echo "Job IDs:"
echo "  Split:   $SPLIT_JOB_ID"
echo "  Process: $PROCESS_JOB_ID"
echo "  Merge:   $MERGE_JOB_ID"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/split_${SPLIT_JOB_ID}.out"
echo "  tail -f logs/process_${PROCESS_JOB_ID}_0.out"
echo "  tail -f logs/merge_${MERGE_JOB_ID}.out"
echo ""
echo "Final results will be in:"
echo "  output/${BASENAME}.json"
echo "==============================================="
