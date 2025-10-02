# VAD Pipeline for SLURM

This repository contains a Voice Activity Detection (VAD) pipeline optimized for high-performance computing environments using SLURM job scheduling.

## Overview

The pipeline processes audio files to detect speech vs. non-speech segments using the TEN-VAD library. It's designed to handle large datasets efficiently using multiprocessing and provides detailed logging and progress tracking.

## Files Description

### Core Pipeline Files

- `vad_pipeline.py` - VAD processing script, outputs detailed CSV results and logs
- `vad_pipeline.slurm` - SLURM script with environment setup, data validation, and result analysis
- `flatten.sh` - Script to flatten nested directory structure into a single level

### Configuration

- `pyproject.toml` - Python dependencies and project configuration
- `uv.lock` - Lock file for reproducible dependency installation

## Prerequisites

1. **Python 3.11+** with virtual environment support
2. **SLURM workload manager** configured on your HPC system
3. **Audio data** in WAV format
4. **Dependencies**: numpy, pandas, scipy, ten-vad

## Data Preparation

Your audio data should be organized in one of these structures:

### Option 1: Pre-flattened (recommended for large datasets)

```
EN_flat/
├── file1.wav
├── file2.wav
└── ...
```

### Option 2: Nested structure (will be automatically flattened)

```
EN/
├── subfolder1/
│   ├── audio1.wav
│   └── audio2.wav
├── subfolder2/
│   └── audio3.wav
└── ...
```

If using nested structure, the script will automatically run `flatten.sh` to create a flat structure with encoded filenames.

## Usage

### Quick Start

1. **Prepare your data** (ensure audio files are in `EN_flat/` or `EN/`)

2. **Submit the job:**
   ```bash
   sbatch vad_pipeline.slurm
   ```

3. **Monitor progress:**
   ```bash
   squeue -u $USER
   tail -f logs/vad_pipeline_<JOB_ID>.out
   ```

### Custom Resource Allocation

```bash
# Use 32 CPUs and 64GB RAM
sbatch --cpus-per-task=32 --mem=64G vad_pipeline.slurm

# Use different partition with 12-hour time limit
sbatch --partition=gpu --time=12:00:00 vad_pipeline.slurm
```

### Manual Pipeline Execution (for testing)

```bash
# Activate environment
source .venv/bin/activate

# Run pipeline with custom parameters
python vad_pipeline.py --input-dir EN_flat --workers 8 --log-level DEBUG
```

## SLURM Configuration

### Resource Requirements

The default configuration is optimized for CPU-intensive processing:

- **CPUs**: 16 cores (adjustable based on dataset size)
- **Memory**: 32GB (increase for very large datasets)
- **Time**: 24 hours (adjust based on dataset size)
- **Partition**: `cpu` (change to `gpu` if GPU acceleration is available)

### Scaling Guidelines

| Dataset Size       | Recommended CPUs | Memory   | Estimated Time |
| ------------------ | ---------------- | -------- | -------------- |
| < 1,000 files      | 8-16             | 16-32GB  | 1-4 hours      |
| 1,000-10,000 files | 16-32            | 32-64GB  | 4-12 hours     |
| > 10,000 files     | 32-64            | 64-128GB | 12-24 hours    |

## Output

The pipeline generates:

1. **CSV Results File**: `vad_results_YYYYMMDD_HHMMSS.csv` containing:

   - File metadata (filename, top-file, mid-file, sequence)
   - Audio metrics (duration, speech ratio)
   - Speech segment statistics (max, min, average duration)
   - Quality flags (flagged_1m, flagged_ns)

2. **Log Files**:
   - `logs/vad_pipeline_<JOB_ID>.out` - Standard output and job progress
   - `logs/vad_pipeline_<JOB_ID>.err` - Error messages
   - `vad_pipeline.log` - Detailed processing log (enhanced version only)

## Monitoring and Troubleshooting

### Check Job Status

```bash
# View job queue
squeue -u $USER

# View detailed job info
scontrol show job <JOB_ID>

# View job efficiency
seff <JOB_ID>
```

### Monitor Progress

```bash
# Follow output log
tail -f logs/vad_pipeline_<JOB_ID>.out

# Check for errors
tail -f logs/vad_pipeline_<JOB_ID>.err

# Monitor resource usage (if available)
sstat -j <JOB_ID> --format=JobID,MaxRSS,MaxVMSize,AveCPU
```

### Common Issues

1. **Out of Memory**: Reduce number of workers or increase memory allocation
2. **Time Limit**: Increase time limit or reduce dataset size
3. **No WAV files found**: Check data directory structure and permissions
4. **Module not found**: Ensure dependencies are properly installed

### Performance Optimization

1. **Optimal Worker Count**: Usually equals number of CPUs, but may need tuning
2. **Memory Usage**: Monitor with `sstat` and adjust allocation
3. **I/O Performance**: Use local scratch space for large datasets if available

## Customization

### Modify VAD Parameters

Edit the pipeline script or use command-line arguments:

- `--hop-size`: Frame size for processing (default: 256)
- `--threshold`: VAD sensitivity threshold (default: 0.5)
- `--workers`: Number of parallel processes

### Adapt for Different HPC Systems

Edit the SLURM script headers:

- Module loading commands
- Partition names
- Resource limits
- Email notifications

### Environment Setup

The scripts support both `uv` and `pip` for dependency management. Customize the installation section for your environment.

## Performance Metrics

Typical performance on modern HPC systems:

- **Processing Rate**: 100-1000 files per minute (depending on file size and CPU count)
- **Memory Usage**: ~2-4GB base + ~100MB per worker
- **CPU Utilization**: Near 100% with optimal worker count

## Support

For issues related to:

- **SLURM configuration**: Consult your HPC documentation
- **TEN-VAD library**: Check the [TEN-VAD repository](https://github.com/TEN-framework/ten-vad)
- **Pipeline bugs**: Review error logs and adjust parameters

## License

This pipeline is provided as-is for research and educational purposes. Please respect the licenses of the underlying dependencies (TEN-VAD, etc.).
