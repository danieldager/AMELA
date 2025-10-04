#!/bin/bash
#SBATCH --job-name=VAD_processing
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --partition=erc-dupoux
#SBATCH --cpus-per-task=32
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dager.verde@gmail.com
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Basic error handling
set -e

echo "==============================================="
echo "VAD Processing Job Started"
echo "==============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Working directory: $(pwd)"

# Load modules
echo "Loading Python module..."
module purge
module load python/3.11

# Go to project directory
cd "$SLURM_SUBMIT_DIR"
echo "Changed to project directory: $(pwd)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
else
    echo "Using existing virtual environment"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "Python: $(which python)"

# Fix C++ library compatibility for TEN-VAD
echo "Setting up C++ library compatibility..."
mkdir -p temp_lib
if [ -f "/usr/lib/x86_64-linux-gnu/libstdc++.so.6" ]; then
    ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 temp_lib/libc++.so.1
    export LD_LIBRARY_PATH="$(pwd)/temp_lib:$LD_LIBRARY_PATH"
    echo "Created libstdc++ -> libc++ compatibility link"
elif [ -f "/usr/lib64/libstdc++.so.6" ]; then
    ln -sf /usr/lib64/libstdc++.so.6 temp_lib/libc++.so.1
    export LD_LIBRARY_PATH="$(pwd)/temp_lib:$LD_LIBRARY_PATH"
    echo "Created libstdc++ -> libc++ compatibility link"
else
    echo "Warning: Could not find libstdc++.so.6"
fi

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install --quiet numpy pandas scipy
echo "Installing TEN-VAD..."
pip install --quiet git+https://github.com/TEN-framework/ten-vad.git

# Test installation
echo "Testing TEN-VAD installation..."
python -c "from ten_vad import TenVad; print('TEN-VAD import successful')"

# Set up CPU usage
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "Set CPU threads to: $OMP_NUM_THREADS"

# Set data paths
DATA_DIR="/store/projects/InfTrain/dataset/wav/EN_flat"
echo "Data directory: $DATA_DIR"

# Check data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi

# Count files
WAV_COUNT=$(find "$DATA_DIR" -name "*.wav" | wc -l)
echo "Found $WAV_COUNT WAV files to process"

# Run the pipeline
echo "Starting VAD processing with $SLURM_CPUS_PER_TASK workers..."
python scripts/vad_pipeline.py \
    --data_dir "$DATA_DIR" \
    --workers $SLURM_CPUS_PER_TASK \
    --log_level INFO

echo "==============================================="
echo "Job completed successfully at $(date)"
echo "==============================================="
