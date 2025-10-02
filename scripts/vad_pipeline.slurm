#!/bin/bash
#SBATCH --job-name=vad_pipeline_enhanced
#SBATCH --output=logs/vad_pipeline_%j.out
#SBATCH --error=logs/vad_pipeline_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@domain.com

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

# Load required modules (adjust based on your cluster)
# Uncomment and modify these lines based on your HPC system:
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

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    check_command "Virtual environment creation"
fi

echo "Activating virtual environment..."
source venv/bin/activate
check_command "Virtual environment activation"

# Upgrade pip and install dependencies
echo "Upgrading pip..."
pip install --upgrade pip
check_command "Pip upgrade"

# Install dependencies using uv if available, otherwise pip
if command -v uv &> /dev/null; then
    echo "Installing dependencies with uv..."
    uv pip install numpy>=2.3.3 pandas>=2.3.3 scipy>=1.16.2
    uv pip install git+https://github.com/TEN-framework/ten-vad.git
else
    echo "Installing dependencies with pip..."
    pip install numpy>=2.3.3 pandas>=2.3.3 scipy>=1.16.2
    pip install git+https://github.com/TEN-framework/ten-vad.git
fi
check_command "Dependency installation"

# Verify python packages
echo "Verifying installed packages..."
python -c "import numpy, pandas, scipy, ten_vad; print('All packages imported successfully')"
check_command "Package verification"

# Check if data needs to be flattened
if [ ! -d "EN_flat" ] && [ -d "EN" ]; then
    echo "EN_flat directory not found, but EN directory exists."
    echo "Running flatten.sh to create flat structure..."
    
    # Make flatten.sh executable if it isn't
    chmod +x flatten.sh
    ./flatten.sh EN
    check_command "Directory flattening"
elif [ ! -d "EN_flat" ]; then
    echo "ERROR: Neither EN_flat nor EN directory found!"
    echo "Available directories:"
    ls -la
    exit 1
fi

# Count wav files and verify data
WAV_COUNT=$(find EN_flat -name "*.wav" 2>/dev/null | wc -l)
echo "Found $WAV_COUNT WAV files to process"

if [ "$WAV_COUNT" -eq 0 ]; then
    echo "ERROR: No WAV files found in EN_flat directory!"
    echo "Contents of EN_flat:"
    ls -la EN_flat/ | head -20
    exit 1
fi

# Estimate processing time (rough estimate: 100-1000 files per minute depending on file size)
ESTIMATED_MINUTES=$((WAV_COUNT / 500))
echo "Estimated processing time: ~$ESTIMATED_MINUTES minutes"

# Determine optimal number of workers
OPTIMAL_WORKERS=$SLURM_CPUS_PER_TASK
if [ "$WAV_COUNT" -lt "$SLURM_CPUS_PER_TASK" ]; then
    OPTIMAL_WORKERS=$WAV_COUNT
fi
echo "Using $OPTIMAL_WORKERS workers for optimal performance"

# Run the enhanced VAD pipeline with appropriate parameters
echo "==============================================="
echo "Starting VAD processing at $(date)"
echo "==============================================="

# Choose which pipeline to run
if [ -f "pipeline_enhanced.py" ]; then
    echo "Running enhanced pipeline with logging and progress tracking..."
    python pipeline_enhanced.py \
        --input-dir EN_flat \
        --output "vad_results_$(date +%Y%m%d_%H%M%S).csv" \
        --workers $OPTIMAL_WORKERS \
        --hop-size 256 \
        --threshold 0.5 \
        --log-level INFO
else
    echo "Running original pipeline..."
    python pipeline.py
fi

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
    
    # Show sample results
    echo ""
    echo "Sample of results (first 5 rows):"
    head -n 6 "$RESULTS_FILE"
    
    # Basic statistics if the enhanced pipeline was used
    if command -v python &> /dev/null; then
        echo ""
        echo "Basic statistics:"
        python -c "
import pandas as pd
df = pd.read_csv('$RESULTS_FILE')
if 'duration' in df.columns and not df['duration'].isna().all():
    print(f'Total files: {len(df)}')
    print(f'Average duration: {df[\"duration\"].mean():.2f}s')
    print(f'Total audio duration: {df[\"duration\"].sum():.2f}s ({df[\"duration\"].sum()/3600:.2f} hours)')
    if 'spch-ratio' in df.columns:
        print(f'Average speech ratio: {df[\"spch-ratio\"].mean():.3f}')
    if 'flagged_1m' in df.columns:
        print(f'Files >1min: {df[\"flagged_1m\"].sum()}')
    if 'flagged_ns' in df.columns:
        print(f'Files flagged (no speech): {df[\"flagged_ns\"].sum()}')
else:
    print('Results file created but may contain errors. Check the log file.')
"
    fi
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