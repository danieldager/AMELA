#!/usr/bin/env python3
"""
Train LSTM language model on audio tokens.

Usage:
    python train.py --manifest metadata/chunk0.csv --tokens_dir output/librivox_mhubert_expresso_2000 \
                    --embedding_dim 256 --hidden_size 512 --num_layers 2 --dropout 0.1 \
                    --batch_size 64 --learning_rate 0.001 --num_epochs 20
"""

import argparse
import os
import random
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers.trainer_callback
import wandb
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from models import LSTM, LSTMConfig

# Suppress warnings
warnings.filterwarnings("ignore", message="Could not estimate the number of tokens")
warnings.filterwarnings("ignore", message=".*barrier.*device_id.*")
warnings.filterwarnings("ignore", message=".*WANDB_DISABLED.*deprecated.*")
warnings.filterwarnings("ignore", message=".*PYTORCH_CUDA_ALLOC_CONF.*deprecated.*")


# ==========================================
# Constants
# ==========================================

NUM_AUDIO_TOKENS = 2000  # mHuBERT/EnCodec codebook size
SOS_TOKEN_ID = NUM_AUDIO_TOKENS
PAD_TOKEN_ID = NUM_AUDIO_TOKENS + 1
VOCAB_SIZE = NUM_AUDIO_TOKENS + 2


# ==========================================
# Distributed Training Helpers
# ==========================================


def is_main_process():
    """Return True if this is rank 0 or not using DDP."""
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


# ==========================================
# Dataset & Data Loading
# ==========================================


class TokenDataset(Dataset):
    """Dataset for loading audio tokens from .pt files."""

    def __init__(
        self,
        manifest_path,
        tokens_dir,
        split="train",
        train_ratio=0.9,
        seed=42,
        max_seq_len=2000,
    ):
        self.tokens_dir = Path(tokens_dir)
        self.max_seq_len = max_seq_len

        # Load and split manifest
        df = pd.read_csv(manifest_path)
        if "file_id" not in df.columns:
            raise ValueError(f"Manifest must have 'file_id' column")

        file_ids = df["file_id"].tolist()
        rng = random.Random(seed)
        rng.shuffle(file_ids)

        split_idx = int(len(file_ids) * train_ratio)
        file_ids = file_ids[:split_idx] if split == "train" else file_ids[split_idx:]

        if is_main_process():
            print(f"{split.capitalize()} dataset: {len(file_ids)} files")
            print(f"Loading sequences into memory...")

        # Load all sequences into memory
        self.sequences = []
        truncated = 0
        skipped_invalid = []

        for file_id in file_ids:
            token_path = self.tokens_dir / f"{file_id}.pt"
            if not token_path.exists():
                continue

            tokens = torch.load(token_path).squeeze().long()

            # Skip invalid tokens (0-d tensors or empty sequences)
            if tokens.ndim == 0 or len(tokens) == 0:
                skipped_invalid.append(
                    (file_id, tokens.ndim, len(tokens) if tokens.ndim > 0 else 0)
                )
                continue

            if len(tokens) > self.max_seq_len:
                tokens = tokens[: self.max_seq_len]
                truncated += 1

            # Prepend SOS token
            sequence = torch.cat(
                [torch.tensor([SOS_TOKEN_ID], dtype=torch.long), tokens]
            )
            self.sequences.append(sequence)

        if is_main_process():
            print(f"Loaded {len(self.sequences)} sequences")
            if truncated > 0:
                print(
                    f"Truncated {truncated} sequences to max length {self.max_seq_len}"
                )
            if skipped_invalid:
                print(f"WARNING: Skipped {len(skipped_invalid)} invalid files")
                for file_id, ndim, length in skipped_invalid[:5]:
                    print(f"  {file_id}: ndim={ndim}, len={length}")
                if len(skipped_invalid) > 5:
                    print(f"  ... and {len(skipped_invalid) - 5} more")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {"input_ids": self.sequences[idx]}


def collate_fn(batch):
    """Collate variable-length sequences with padding."""
    sequences = [item["input_ids"] for item in batch]
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded_batch = pad_sequence(sequences, batch_first=True, padding_value=PAD_TOKEN_ID)

    return {
        "inputs": padded_batch,
        "labels": padded_batch,
        "lengths": lengths,
    }


# ==========================================
# Logging Callback
# ==========================================


class FormattedLoggingCallback(TrainerCallback):
    """Minimal callback for clean epoch logging."""

    def __init__(self):
        self.header_printed = False
        self.train_logs = {}
        self.t_start = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.t_start = time.time()
        self.train_logs = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not is_main_process():
            return

        # Buffer training metrics
        if "loss" in logs and "eval_loss" not in logs:
            self.train_logs = logs.copy()
            return

        # Print when we have eval metrics
        if "eval_loss" not in logs:
            return

        # Print header once
        if not self.header_printed:
            print("\nepoch     loss       ppl  val_loss   val_ppl  epoch_s         lr")
            print("-----  -------  --------  --------  --------  -------  ---------")
            self.header_printed = True

        # Calculate and print metrics
        epoch_s = int(time.time() - self.t_start) if self.t_start else 0
        epoch = logs.get("epoch", 0)
        loss = self.train_logs.get("loss", 0)
        ppl = np.exp(loss) if loss else 0
        val_loss = logs.get("eval_loss", 0)
        val_ppl = np.exp(val_loss) if val_loss else 0
        lr = self.train_logs.get("learning_rate", 0)

        print(
            f"{epoch:5.2f}  {loss:7.4f}  {ppl:8.1f}  {val_loss:8.4f}  {val_ppl:8.1f}  "
            f"{epoch_s:7}  {lr:9.6f}",
            flush=True,
        )


# ==========================================
# Utilities
# ==========================================


def create_checkpoint_name(
    learning_rate, hidden_size, embedding_dim, num_layers, batch_size, dropout
):
    """Create checkpoint directory name from hyperparameters."""
    return f"lstm_h{hidden_size}_r{learning_rate}_e{embedding_dim}_l{num_layers}_b{batch_size}_d{dropout}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--tokens_dir", type=str, required=True)
    parser.add_argument("--embedding_dim", type=int, required=True)
    parser.add_argument("--hidden_size", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--early_stopping", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--group_by_length", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def setup_wandb(checkpoint_name):
    """Initialize Weights & Biases logging (main process only)."""
    if is_main_process():
        wandb.init(
            project="amela-lstm",
            name=checkpoint_name,
            id=os.environ["SLURM_JOB_ID"],
            resume="allow",
        )


def create_output_directory(checkpoint_name, manifest_path):
    """Create and return output directory for checkpoints."""
    dataset_name = Path(manifest_path).stem.split("_")[0]
    timestamp = datetime.now().strftime("%d-%m-%y")
    output_dir = Path("checkpoints") / checkpoint_name / dataset_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def create_datasets(manifest_path, tokens_dir, train_ratio, seed):
    """Create training and validation datasets."""
    train_dataset = TokenDataset(manifest_path, tokens_dir, "train", train_ratio, seed)
    val_dataset = TokenDataset(manifest_path, tokens_dir, "val", train_ratio, seed)
    return train_dataset, val_dataset


def create_model(args):
    """Create and initialize LSTM model."""
    config = LSTMConfig(
        vocab_size=VOCAB_SIZE,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        sos_token_id=SOS_TOKEN_ID,
    )
    model = LSTM(config)

    # Print parameter counts
    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}\n")

    return model


def create_training_args(args, output_dir, checkpoint_name):
    """Create Hugging Face TrainingArguments."""
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        gradient_accumulation_steps=args.grad_accum_steps,
        max_grad_norm=args.max_grad_norm,
        fp16=args.use_fp16,
        bf16=args.use_bf16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        report_to="wandb" if is_main_process() else "none",
        run_name=checkpoint_name,
        disable_tqdm=True,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        group_by_length=args.group_by_length,
        seed=args.seed,
        remove_unused_columns=False,
    )


def create_trainer(model, training_args, train_dataset, val_dataset, args):
    """Create Hugging Face Trainer with callbacks."""
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping),
            FormattedLoggingCallback(),
        ],
    )

    # Remove default logging callbacks
    trainer.remove_callback(transformers.trainer_callback.PrinterCallback)
    trainer.remove_callback(transformers.trainer_callback.ProgressCallback)

    return trainer


def print_training_summary(trainer, total_duration):
    """Print final training results."""
    if not is_main_process():
        return

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)

    # Final evaluation metrics
    eval_result = trainer.evaluate()
    print(f"Final eval loss: {eval_result['eval_loss']:.4f}")
    print(f"Final eval perplexity: {np.exp(eval_result['eval_loss']):.2f}")

    # Best model metrics
    if hasattr(trainer.state, "best_metric") and trainer.state.best_metric is not None:
        print(f"Best eval loss: {trainer.state.best_metric:.4f}")
        print(f"Best eval perplexity: {np.exp(trainer.state.best_metric):.2f}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {total_duration / 60:.1f} min")
    print("=" * 60)


# ==========================================
# Main Training Loop
# ==========================================


def main():
    args = parse_args()
    set_seed(args.seed)
    script_start_time = time.time()

    # Print startup info
    if is_main_process():
        print("=" * 60)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Manifest: {args.manifest}")
        print(f"Tokens: {args.tokens_dir}")
        print(f"Seed: {args.seed}")
        print("=" * 60)
        print()

    # Setup
    checkpoint_name = create_checkpoint_name(
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        dropout=args.dropout,
    )

    setup_wandb(checkpoint_name)
    output_dir = create_output_directory(checkpoint_name, args.manifest)

    if is_main_process():
        print(f"Checkpoints: {output_dir}\n")

    # Prepare data and model
    train_dataset, val_dataset = create_datasets(
        args.manifest, args.tokens_dir, args.train_ratio, args.seed
    )
    model = create_model(args)

    # Setup training
    training_args = create_training_args(args, output_dir, checkpoint_name)
    trainer = create_trainer(model, training_args, train_dataset, val_dataset, args)

    # Print precision info
    if is_main_process():
        precision = (
            "BF16" if training_args.bf16 else "FP16" if training_args.fp16 else "FP32"
        )
        print(f"Training precision: {precision}\n")
        print("Starting training...")

    # Train
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save and summarize
    if is_main_process():
        print("\nSaving final model...")
        trainer.save_model(str(output_dir / "final_model"))

    print_training_summary(trainer, time.time() - script_start_time)


if __name__ == "__main__":
    main()
