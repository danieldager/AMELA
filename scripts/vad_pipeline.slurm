#!/bin/bash
#SBATCH --job-name=VAD_processing
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=40
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dager.verde@gmail.com
#SBATCH --output=logs/vad_pipeline_%j.out
#SBATCH --error=logs/vad_pipeline_%j.err

# Set strict error handling
set -euo pipefail

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "==============================================="
echo "VAD Pipeline SLURM Job Information"
echo "==============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST" 
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory per node: $SLURM_MEM_PER_NODE MB"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Submit directory: $SLURM_SUBMIT_DIR"
echo "==============================================="

# Load required modules 
# module load python/3.11
# module load gcc/11.2.0
# module load cuda/11.8  # if GPU support needed

# Set environment variables for optimal CPU usage
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Change to the project directory
cd $SLURM_SUBMIT_DIR

# Function to check command success
check_command() {
    if [ $? -ne 0 ]; then
        echo "ERROR: $1 failed"
        exit 1
    fi
}

# Install uv if not available (most HPC systems won't have it pre-installed)
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"
    check_command "uv installation"
fi

# Use uv to create virtual environment and install all dependencies from lock file
# This will create a .venv directory and install everything from pyproject.toml + uv.lock
echo "Creating virtual environment and installing dependencies with uv..."
uv sync
check_command "uv sync (environment creation and dependency installation)"

# Activate the virtual environment created by uv
echo "Activating uv-managed virtual environment..."
source .venv/bin/activate
check_command "Virtual environment activation"

echo "Running job on $(hostname)"
echo "python: $(which python)"
echo "python-version $(python -V)"

# Verify python packages
echo "Verifying installed packages..."
python -c "import numpy, pandas, scipy, ten_vad; print('All packages imported successfully')"
check_command "Package verification"

# Set data directory paths for the cluster (script creates _flat, not _flatten)
EN_DIR="/store/projects/InfTrain/dataset/wav/EN"
EN_FLAT_DIR="/store/projects/InfTrain/dataset/wav/EN_flat"

# Check if flattened data already exists, otherwise create it
if [ -d "$EN_FLAT_DIR" ]; then
    echo "Found existing flattened data directory"
else
    echo "Flattened data not found. Creating flat structure..."
    
    # Verify original data exists
    if [ ! -d "$EN_DIR" ]; then
        echo "ERROR: Original data directory not found: $EN_DIR"
        exit 1
    fi
    
    # Check if flatten_dataset.sh exists
    if [ ! -f "flatten_dataset.sh" ]; then
        echo "ERROR: flatten_dataset.sh script not found!"
        exit 1
    fi
    
    # Make flatten_dataset.sh executable and run it
    chmod +x flatten_dataset.sh
    echo "Running: ./flatten_dataset.sh '$EN_DIR'"
    ./flatten_dataset.sh "$EN_DIR"
    check_command "Directory flattening"
    
    # Verify the flattened directory was created
    if [ ! -d "$EN_FLAT_DIR" ]; then
        echo "ERROR: Flattening failed - $EN_FLAT_DIR was not created"
        exit 1
    fi
    echo "Successfully created flattened directory"
fi

# Use the flattened directory for processing
DATA_DIR="$EN_FLAT_DIR"

# Count wav files and verify data
echo "Scanning for WAV files in: $DATA_DIR"
WAV_COUNT=$(find "$DATA_DIR" -name "*.wav" 2>/dev/null | wc -l)
echo "Found $WAV_COUNT WAV files to process"

if [ "$WAV_COUNT" -eq 0 ]; then
    echo "ERROR: No WAV files found in data directory!"
    echo "Contents of $DATA_DIR:"
    ls -la "$DATA_DIR" | head -20
    exit 1
fi

# Run the VAD pipeline
echo "==============================================="
echo "Starting VAD processing at $(date)"
echo "==============================================="

echo "Running VAD pipeline with logging and progress tracking..."
python vad_pipeline.py \
    --input-dir "$DATA_DIR" \
    --output "vad_results_$(date +%Y%m%d_%H%M%S).csv" \
    --workers $OPTIMAL_WORKERS \
    --hop-size 256 \
    --threshold 0.5 \
    --log-level INFO

check_command "VAD pipeline execution"

# Find the most recent results file
RESULTS_FILE=$(ls -t vad_results*.csv 2>/dev/null | head -1)
if [ -z "$RESULTS_FILE" ]; then
    RESULTS_FILE="vad_results.csv"
fi

# Check if output was created and analyze results
if [ -f "$RESULTS_FILE" ]; then
    RESULT_COUNT=$(tail -n +2 "$RESULTS_FILE" | wc -l)
    echo "==============================================="
    echo "Processing Complete!"
    echo "==============================================="
    echo "Generated results for $RESULT_COUNT files"
    echo "Output saved to: $RESULTS_FILE"
    echo "File size: $(du -h "$RESULTS_FILE" | cut -f1)"
    
else
    echo "ERROR: $RESULTS_FILE was not created!"
    echo "Check the error log for details."
    exit 1
fi

# Calculate total runtime
RUNTIME=$SECONDS
HOURS=$((RUNTIME / 3600))
MINUTES=$(((RUNTIME % 3600) / 60))
SECS=$((RUNTIME % 60))

echo ""
echo "==============================================="
echo "Job Summary"
echo "==============================================="
echo "Job completed at: $(date)"
echo "Total runtime: ${HOURS}h ${MINUTES}m ${SECS}s"
echo "Files processed: $RESULT_COUNT"
echo "Processing rate: $(python -c "print(f'{$RESULT_COUNT / ($RUNTIME + 1):.2f} files/second')")"
echo "Log files: logs/vad_pipeline_${SLURM_JOB_ID}.out, logs/vad_pipeline_${SLURM_JOB_ID}.err"
echo "Results: $RESULTS_FILE"

# Cleanup
echo "Deactivating virtual environment..."
deactivate

echo "Job completed successfully!"