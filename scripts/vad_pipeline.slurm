#!/bin/bash
#SBATCH --job-name=almela_VAD
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=erc-dupoux
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --export=ALL

# Basic error handling
set -e

echo "==============================================="
echo "VAD Processing Job Started"
echo "==============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Python: $(which python) $(python -V)"
echo "Working directory: $(pwd)"

# # Test installation
echo "Testing TEN-VAD installation..."
python -c "from ten_vad import TenVad; print('TEN-VAD import successful')"

# Set data paths
SCRIPT_DIR="/scratch2/ddager/almela/scripts
echo "Scripts directory: $SCRIPT_DIR"

DATA_DIR="/store/projects/InfTrain/dataset/wav/EN_flat"
echo "Data directory: $DATA_DIR"

# Check data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi

# # Count files
# WAV_COUNT=$(find "$DATA_DIR" -name "*.wav" | wc -l)
# echo "Found $WAV_COUNT WAV files to process"

# Run the pipeline
echo "Starting VAD processing with $SLURM_CPUS_PER_TASK workers..."
python scripts/vad_pipeline.py \
    --data_dir "$DATA_DIR" \
    --workers $SLURM_CPUS_PER_TASK \
    --log_name "$SLURM_JOB_ID" \
    --log_level INFO

echo "==============================================="
echo "Job completed successfully at $(date)"
echo "==============================================="
