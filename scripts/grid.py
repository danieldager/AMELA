#!/usr/bin/env python3
"""
Launch grid search for LSTM training.

Define parameter lists of equal length - each index is one training run.
Launches train.slurm with appropriate parameters.

Usage:
    python grid.py --manifest metadata/librivox.csv --tokens_dir output/tokens
"""

import argparse
import os
import subprocess
from pathlib import Path


# ========================================
# GRID SEARCH PARAMETERS
# ========================================
# Each list must have the same length
# Index i across all lists = configuration for run i

# HIDDEN_SIZES = [512, 1024, 2048]
# EMBEDDING_DIMS = [256, 512, 1024]
# BATCH_SIZES = [512, 256, 128]
# NUM_LAYERS = [2, 2, 3]
# NUM_GPUS = [1, 2, 4]
# LEARNING_RATES = [0.0003] * 3
# DROPOUTS = [0.0] * 3
# GRAD_ACCUM_STEPS = [4] * 3

HIDDEN_SIZES = [512]
EMBEDDING_DIMS = [256]
BATCH_SIZES = [512]
NUM_LAYERS = [2]
NUM_GPUS = [1]
LEARNING_RATES = [0.0003]
DROPOUTS = [0.0]
GRAD_ACCUM_STEPS = [4]


# Fixed parameters (same for all runs)
NUM_EPOCHS = 100
TRAIN_RATIO = 0.95
EARLY_STOPPING = 5


def validate_grid():
    """Check that all parameter lists have the same length."""
    lengths = [
        len(HIDDEN_SIZES),
        len(EMBEDDING_DIMS),
        len(NUM_LAYERS),
        len(BATCH_SIZES),
        len(DROPOUTS),
        len(LEARNING_RATES),
        len(NUM_GPUS),
        len(GRAD_ACCUM_STEPS),
    ]
    if len(set(lengths)) != 1:
        raise ValueError(f"All parameter lists must have same length. Got: {lengths}")
    return lengths[0]


def launch_job(manifest_path: str, tokens_dir: str, config: dict):
    """
    Launch train.slurm with parameters set via environment variables.

    Args:
        manifest_path: Path to CSV manifest
        tokens_dir: Path to tokens directory
        config: Dict with hyperparameters including 'num_gpus'

    Returns:
        Job ID if launched, None on failure
    """
    # Build environment variables for the SLURM script
    env = {
        "EMBEDDING_DIM": str(config["embedding_dim"]),
        "HIDDEN_SIZE": str(config["hidden_size"]),
        "NUM_LAYERS": str(config["num_layers"]),
        "DROPOUT": str(config["dropout"]),
        "BATCH_SIZE": str(config["batch_size"]),
        "LEARNING_RATE": str(config["learning_rate"]),
        "NUM_EPOCHS": str(config["num_epochs"]),
        "TRAIN_RATIO": str(config["train_ratio"]),
        "EARLY_STOPPING": str(config["early_stopping"]),
        "GRAD_ACCUM_STEPS": str(config["grad_accum_steps"]),
    }

    # Build sbatch command with GPU count
    cmd = [
        "sbatch",
        f"--gres=gpu:{config['num_gpus']}",
        f"--ntasks-per-node={config['num_gpus']}",
        "scripts/train.slurm",
        manifest_path,
        tokens_dir,
    ]

    try:
        # Submit job with environment variables
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, env={**os.environ, **env}
        )

        # Extract job ID
        job_id = result.stdout.strip().split()[-1]
        model_name = (
            f"lstm_h{config['hidden_size']}_"
            f"e{config['embedding_dim']}_"
            f"l{config['num_layers']}"
        )
        print(f"✓ Run {config['run_id']}: {model_name} → Job {job_id}")

        return job_id

    except subprocess.CalledProcessError as e:
        print(f"✗ Run {config['run_id']} failed: {e.stderr}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Launch grid search for LSTM training")
    parser.add_argument(
        "--manifest", type=str, required=True, help="Path to CSV manifest"
    )
    parser.add_argument(
        "--tokens_dir", type=str, required=True, help="Path to tokens directory"
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.manifest).exists():
        print(f"ERROR: Manifest not found: {args.manifest}")
        return 1

    if not Path(args.tokens_dir).exists():
        print(f"ERROR: Tokens directory not found: {args.tokens_dir}")
        return 1

    # Validate grid
    try:
        num_runs = validate_grid()
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    print("=" * 60)
    print("LSTM Grid Search")
    print("=" * 60)
    print(f"Manifest: {args.manifest}")
    print(f"Tokens: {args.tokens_dir}")
    print(f"Runs: {num_runs}")
    print("=" * 60)
    print()

    # Launch all jobs
    job_ids = []
    for i in range(num_runs):
        config = {
            "run_id": i + 1,
            "hidden_size": HIDDEN_SIZES[i],
            "embedding_dim": EMBEDDING_DIMS[i],
            "num_layers": NUM_LAYERS[i],
            "batch_size": BATCH_SIZES[i],
            "dropout": DROPOUTS[i],
            "learning_rate": LEARNING_RATES[i],
            "num_gpus": NUM_GPUS[i],
            "grad_accum_steps": GRAD_ACCUM_STEPS[i],
            "num_epochs": NUM_EPOCHS,
            "train_ratio": TRAIN_RATIO,
            "early_stopping": EARLY_STOPPING,
        }

        job_id = launch_job(args.manifest, args.tokens_dir, config)

        if job_id:
            job_ids.append(job_id)

    print()
    print("=" * 60)
    print(f"Launched: {len(job_ids)}/{num_runs} jobs")
    print("=" * 60)
    print("Job IDs:", ", ".join(job_ids))

    return 0


if __name__ == "__main__":
    exit(main())
