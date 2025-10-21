#!/bin/bash
#
# ASR Pipeline - Step 3: Merge Results
# CPU-only job, no GPU needed
#
#SBATCH --job-name=asr_merge
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=erc-dupoux
#SBATCH --output=logs/merge_%j.out
#SBATCH --error=logs/merge_%j.err
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

echo "==============================================="
echo "ASR Merge Job"
echo "==============================================="
echo "Input manifest: $INPUT_MANIFEST"
echo "Output: ${OUTPUT_DIR}/${BASENAME}_results.json"
echo "==============================================="

# Merge all split results
echo "Merging results..."
python "$SCRIPT_DIR/asr_merge.py" \
    --input_pattern "${OUTPUT_DIR}/split_*.json" \
    --output_path "${OUTPUT_DIR}/${BASENAME}.json"

echo "==============================================="
echo "Merge complete!"
echo "==============================================="

# Verify output
INPUT_COUNT=$(wc -l < "$INPUT_MANIFEST")
OUTPUT_COUNT=$(wc -l < "${OUTPUT_DIR}/${BASENAME}.json")

echo "Input entries:  $INPUT_COUNT"
echo "Output entries: $OUTPUT_COUNT"

if [ "$INPUT_COUNT" -eq "$OUTPUT_COUNT" ]; then
    echo "SUCCESS: Counts match - all files processed!"
else
    echo "WARNING: Count mismatch! Some files may have failed."
    echo "Check process logs for errors: logs/process_*_*.err"
fi

echo "==============================================="
echo "Final results: ${OUTPUT_DIR}/${BASENAME}.json"
echo "==============================================="
