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

# Load modules (including conda if available)
echo "Loading modules..."
module purge
module load python/3.11

# Try to load conda module if available
if module avail conda 2>&1 | grep -q conda; then
    module load conda
    echo "Loaded conda module"
elif command -v conda &> /dev/null; then
    echo "Conda already available"
else
    echo "Conda not available, will use pip approach"
fi

# Go to project directory
cd "$SLURM_SUBMIT_DIR"
echo "Changed to project directory: $(pwd)"

# Try conda approach first if available
if command -v conda &> /dev/null; then
    echo "Using conda environment approach..."
    
    # Create conda environment with C++ runtime libraries
    ENV_NAME="vad_env_$$"  # Unique name using process ID
    echo "Creating conda environment: $ENV_NAME"
    conda create -n "$ENV_NAME" python=3.11 libcxx libcxxabi libstdcxx-ng -y
    
    # Activate conda environment
    echo "Activating conda environment..."
    source activate "$ENV_NAME"
    echo "Python: $(which python)"
    
    # Install Python packages
    echo "Installing Python packages with conda..."
    conda install numpy pandas scipy -y
    
    # Install TEN-VAD with pip in conda env
    echo "Installing TEN-VAD..."
    pip install git+https://github.com/TEN-framework/ten-vad.git
    
else
    echo "Falling back to venv approach..."
    
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
    
    # Install dependencies
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install --quiet numpy pandas scipy
    echo "Installing TEN-VAD..."
    pip install --quiet git+https://github.com/TEN-framework/ten-vad.git
fi

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

# Cleanup conda environment if we created one
if command -v conda &> /dev/null && [ ! -z "$ENV_NAME" ]; then
    echo "Cleaning up conda environment: $ENV_NAME"
    conda deactivate
    conda env remove -n "$ENV_NAME" -y
fi

echo "==============================================="
echo "Job completed successfully at $(date)"
echo "==============================================="
