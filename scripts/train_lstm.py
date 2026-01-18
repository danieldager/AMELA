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
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import transformers.trainer_callback
import wandb
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from models import LSTM, LSTMConfig
from train_utils import (
    VOCAB_SIZE,
    PAD_TOKEN_ID,
    SOS_TOKEN_ID,
    MAX_SEQ_LEN,
    TokenDataset,
    FormattedLoggingCallback,
    is_task_zero,
    set_seed,
    create_output_directory,
    print_training_summary,
)

# ==========================================
# Data Loading
# ==========================================


def collate_fn(batch):
    """Collate variable-length sequences with padding."""
    sequences = [item["input_ids"] for item in batch]
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded_batch = pad_sequence(sequences, batch_first=True, padding_value=PAD_TOKEN_ID)

    return {
        "input_ids": padded_batch,
        "labels": padded_batch,
        "lengths": lengths,
    }


def create_datasets(manifest_path, tokens_dir, train_ratio, seed):
    """Create training and validation datasets."""
    train_dataset = TokenDataset(manifest_path, tokens_dir, "train", train_ratio, seed)
    val_dataset = TokenDataset(manifest_path, tokens_dir, "val", train_ratio, seed)
    return train_dataset, val_dataset


# ==========================================
# Utilities
# ==========================================


def parse_lstm_args():
    """Parse args specific to LSTM (adds embedding_dim)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--tokens_dir", type=str, required=True)
    parser.add_argument("--embedding_dim", type=int, required=True)
    parser.add_argument("--hidden_size", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--early_stopping", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--group_by_length", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def setup_wandb(checkpoint_name):
    """Initialize Weights & Biases logging (main process only)."""
    if is_task_zero():
        wandb.init(
            project="amela-lstm",
            name=checkpoint_name,
            id=os.environ["SLURM_JOB_ID"],
            resume="allow",
        )


def create_checkpoint_name(
    learning_rate, hidden_size, embedding_dim, num_layers, batch_size, dropout
):
    """Create checkpoint directory name from hyperparameters."""
    return f"lstm_h{hidden_size}_r{learning_rate}_e{embedding_dim}_l{num_layers}_b{batch_size}_d{dropout}"


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
    if is_task_zero():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}\n")

    return model


def create_training_args(args, output_dir, checkpoint_name):
    """Create Hugging Face TrainingArguments."""
    return TrainingArguments(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type="inverse_sqrt",
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.grad_accum_steps,
        max_grad_norm=0.0,
        bf16=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_strategy="steps",
        logging_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=20,
        report_to="wandb" if is_task_zero() else "none",
        run_name=checkpoint_name,
        disable_tqdm=True,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        adam_beta1=0.9,
        adam_beta2=0.98,
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
    if not is_task_zero():
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
    args = parse_lstm_args()
    set_seed(args.seed)
    script_start_time = time.time()

    # Print startup info
    if is_task_zero():
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

    if is_task_zero():
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
    if is_task_zero():
        print(f"Training precision: BF16\n")
        print("Starting training...")

    # Train
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save and summarize
    if is_task_zero():
        print("\nSaving final model...")
        trainer.save_model(str(output_dir / "final_model"))

    print_training_summary(trainer, time.time() - script_start_time)


if __name__ == "__main__":
    main()
