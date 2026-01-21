#!/usr/bin/env python3
"""Shared utilities for LSTM and GPT2 training."""

import os
import random
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import IterableDataset
from transformers import TrainerCallback

# Suppress warnings
warnings.filterwarnings("ignore", message="Could not estimate the number of tokens")
warnings.filterwarnings("ignore", message=".*barrier.*device_id.*")
warnings.filterwarnings("ignore", message=".*WANDB_DISABLED.*deprecated.*")
warnings.filterwarnings("ignore", message=".*PYTORCH_CUDA_ALLOC_CONF.*deprecated.*")


# Constants
NUM_AUDIO_TOKENS = 2000  # mHuBERT/EnCodec codebook size
SOS_TOKEN_ID = NUM_AUDIO_TOKENS
PAD_TOKEN_ID = NUM_AUDIO_TOKENS + 1
VOCAB_SIZE = NUM_AUDIO_TOKENS + 2
MAX_SEQ_LEN = 2048


def is_task_zero():
    """Check if main process (rank 0 or no DDP)."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    return local_rank == -1 or local_rank == 0


def set_seed(seed: int = 42):
    """Set global seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class TokenDataset(IterableDataset):
    """Stream audio token sequences from webdataset tar files.

    With DDP: Each rank processes subset of shards via split_by_node().
    For validation (rank 0 only): Disable split_by_node to get full dataset.
    """

    def __init__(
        self,
        manifest_path,
        tokens_dir,
        split="train",
        train_ratio=0.9,
        seed=42,
        max_seq_len=MAX_SEQ_LEN,
        shuffle_buffer=10000,
        use_split_by_node=True,
    ):
        self.tokens_dir = Path(tokens_dir)
        self.max_seq_len = max_seq_len
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.use_split_by_node = use_split_by_node

        # Get DDP rank and world size from environment
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        if not self.tokens_dir.exists():
            raise ValueError(f"tokens_dir not found: {self.tokens_dir}")

        # Find all webdataset shard files
        shard_files = sorted(self.tokens_dir.glob("shard-*.tar"))

        if not shard_files:
            raise ValueError(f"No webdataset shards found in {self.tokens_dir}")

        # Split shards into train/val based on shard filenames (deterministic)
        rng = random.Random(seed)
        shard_files_shuffled = list(shard_files)
        rng.shuffle(shard_files_shuffled)

        split_idx = int(len(shard_files_shuffled) * train_ratio)
        if split == "train":
            selected_shards = shard_files_shuffled[:split_idx]
        else:
            selected_shards = shard_files_shuffled[split_idx:]

        # Store all shards for this split (webdataset + split_by_node() handles rank distribution)
        self.shard_urls = [str(f) for f in selected_shards]

        if is_task_zero():
            print(
                f"{split.capitalize()} dataset: {len(self.shard_urls)} shards"
            )
            if self.use_split_by_node:
                print(f"Distributed across {self.world_size} ranks via split_by_node()")
            else:
                print(f"Full dataset on rank 0 (split_by_node disabled)")

    def _process_sample(self, sample):
        """Process a webdataset sample: validate, truncate, add SOS token."""
        try:
            # Extract token array from sample
            tokens = sample["token.npy"]
            tokens = torch.from_numpy(tokens).squeeze().long()

            # Validate
            if tokens.ndim == 0 or len(tokens) == 0:
                return None

            if tokens.min() < 0 or tokens.max() >= NUM_AUDIO_TOKENS:
                return None

            # Truncate if needed
            if len(tokens) > self.max_seq_len:
                tokens = tokens[: self.max_seq_len]

            # Prepend SOS token
            sequence = torch.cat(
                [torch.tensor([SOS_TOKEN_ID], dtype=torch.long), tokens]
            )

            return {"input_ids": sequence}
        except Exception as e:
            # Skip corrupted samples
            return None

    def __iter__(self):
        """Create iterator for streaming from webdataset.
        
        Uses split_by_node() for training (distributed across ranks).
        Skips split_by_node() for validation (rank 0 gets full dataset).
        """
        dataset = wds.WebDataset(self.shard_urls, shardshuffle=False)  # type: ignore
        
        # Apply DDP distribution only for training
        if self.use_split_by_node:
            dataset = dataset.split_by_node()  # Distribute shards across ranks
        
        dataset = dataset.decode()  # Decode numpy arrays automatically

        # Process and filter samples
        for sample in dataset:
            processed = self._process_sample(sample)
            if processed is not None:
                yield processed


class FormattedLoggingCallback(TrainerCallback):
    """Clean step logging."""

    def __init__(self):
        self.header_printed = False
        self.train_logs = {}
        self.previous_time = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not is_task_zero():
            return

        if "loss" in logs and "eval_loss" not in logs:
            self.train_logs = logs.copy()
            return

        if "eval_loss" not in logs:
            return

        if not self.header_printed:
            print("\n step    loss    v_loss    ppl     v_ppl      lr       time")
            print("------  -------  -------  ------  ------  ---------  --------")
            self.header_printed = True
            self.previous_time = time.time()

        current_time = time.time()
        elapsed = current_time - self.previous_time if self.previous_time else 0
        self.previous_time = current_time

        step = state.global_step
        loss = self.train_logs.get("loss", 0)
        v_loss = logs.get("eval_loss", 0)
        ppl = np.exp(loss) if loss else 0
        v_ppl = np.exp(v_loss) if v_loss else 0
        lr = self.train_logs.get("learning_rate", 0)

        print(
            f"{int(step):6d}  {loss:7.4f}  {v_loss:7.4f}  {ppl:6.1f}  {v_ppl:6.1f}  {lr:9.6f}  {elapsed:8.1f}"
        )


def create_output_directory(checkpoint_name, manifest_path):
    """Create checkpoint output directory."""
    dataset_name = Path(manifest_path).stem.split("_")[0]
    timestamp = datetime.now().strftime("%d-%m-%y")
    output_dir = Path("checkpoints") / checkpoint_name / dataset_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def create_datasets(train_manifest, train_tokens_dir, val_manifest, val_tokens_dir, seed):
    """Create train and validation datasets from separate manifests.

    In DDP mode:
    - Training: distributed via split_by_node() across all ranks
    - Validation: distributed via split_by_node() across all ranks
    - Trainer automatically aggregates eval metrics from all ranks
    """
    train_dataset = TokenDataset(
        train_manifest, train_tokens_dir, "train", train_ratio=1.0, seed=seed, use_split_by_node=True
    )
    # All ranks evaluate (distributed), Trainer aggregates metrics automatically
    val_dataset = TokenDataset(
        val_manifest, val_tokens_dir, "val", train_ratio=0.0, seed=seed, use_split_by_node=True
    )
    return train_dataset, val_dataset


def print_training_summary(trainer, total_duration):
    if not is_task_zero():
        return

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)

    eval_result = trainer.evaluate()
    print(f"Final eval loss: {eval_result['eval_loss']:.4f}")
    print(f"Final eval perplexity: {np.exp(eval_result['eval_loss']):.2f}")

    if hasattr(trainer.state, "best_metric") and trainer.state.best_metric is not None:
        print(f"Best eval loss: {trainer.state.best_metric:.4f}")
        print(f"Best eval perplexity: {np.exp(trainer.state.best_metric):.2f}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {total_duration / 60:.1f} min")
    print("=" * 60)
