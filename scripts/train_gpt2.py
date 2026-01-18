#!/usr/bin/env python3
"""
Train GPT2 language model on audio tokens.

Usage:
    python train_gpt2.py --manifest metadata/chunk0.csv --tokens_dir output/librivox_mhubert_expresso_2000 \
                         --hidden_size 256 --num_layers 2 --dropout 0.1 \
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
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)

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
    create_datasets,
    print_training_summary,
)

# ==========================================
# Data Loading
# ==========================================


def collate_fn(batch):
    """Collate variable-length sequences with padding and attention mask."""
    sequences = [item["input_ids"] for item in batch]
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded_batch = pad_sequence(sequences, batch_first=True, padding_value=PAD_TOKEN_ID)

    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = (padded_batch != PAD_TOKEN_ID).long()

    return {
        "input_ids": padded_batch,
        "labels": padded_batch,
        "attention_mask": attention_mask,
        "lengths": lengths,
    }


# ==========================================
# Utilities
# ==========================================


def create_checkpoint_name(learning_rate, hidden_size, num_layers, batch_size, dropout):
    """Create checkpoint directory name from hyperparameters."""
    return (
        f"gpt2_h{hidden_size}_r{learning_rate}_l{num_layers}_b{batch_size}_d{dropout}"
    )


def parse_gpt2_args():
    """Parse args specific to GPT2."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=str, required=True, help="Training manifest path"
    )
    parser.add_argument(
        "--tokens_dir", type=str, required=True, help="Training tokens directory"
    )
    parser.add_argument(
        "--val_manifest", type=str, required=True, help="Validation manifest path"
    )
    parser.add_argument(
        "--val_tokens_dir", type=str, required=True, help="Validation tokens directory"
    )
    parser.add_argument("--hidden_size", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--early_stopping", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def setup_wandb(checkpoint_name):
    """Initialize Weights & Biases logging (main process only)."""
    if is_task_zero():
        wandb.init(
            project="amela-gpt2",
            name=checkpoint_name,
            id=os.environ["SLURM_JOB_ID"],
            resume="allow",
        )


def create_model(args):
    """Create and initialize GPT2 model."""
    # GPT2Config with optimal number of attention heads
    # Hidden size should be divisible by num_attention_heads
    num_attention_heads = min(16, max(1, args.hidden_size // 64))
    if args.hidden_size % num_attention_heads != 0:
        num_attention_heads = 1

    config = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_positions=MAX_SEQ_LEN,
        n_embd=args.hidden_size,
        n_layer=args.num_layers,
        n_head=num_attention_heads,
        resid_pdrop=args.dropout,
        embd_pdrop=args.dropout,
        attn_pdrop=args.dropout,
        initializer_range=0.02,
        summary_first_dropout=args.dropout,
        bos_token_id=SOS_TOKEN_ID,
        eos_token_id=PAD_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
        use_cache=False,  # Disable KV cache for training
    )
    model = GPT2LMHeadModel(config)

    # Disable SDPA to avoid masking kernel issues with gather operations
    # This ensures attention uses the standard PyTorch implementation
    model.config.attn_implementation = "eager"

    # Print parameter counts
    if is_task_zero():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}\n")

    return model


def create_training_args(args, output_dir, checkpoint_name):
    """Create Hugging Face TrainingArguments.

    Note: In DDP mode, all ranks evaluate their portion of data.
    Trainer automatically aggregates metrics across ranks.
    """
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
        # All ranks participate in distributed evaluation
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_strategy="steps",
        logging_steps=args.eval_steps,
        # Rank 0 loads best model (other ranks follow automatically in DDP)
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=20,
        report_to="wandb" if is_task_zero() else "none",
        run_name=checkpoint_name,
        disable_tqdm=True,
        ddp_find_unused_parameters=False,
        # IterableDataset + webdataset: each worker gets independent shuffled copy.
        # Must be 0 to avoid duplicate samples across GPUs.
        dataloader_num_workers=0,
        dataloader_pin_memory=True,  # Efficient GPU transfer from CPU
        adam_beta1=0.9,
        adam_beta2=0.98,
        seed=args.seed,
        remove_unused_columns=False,
        # IterableDataset doesn't support length-based batching.
        # Sequences are shuffled and padded in collate_fn instead.
        group_by_length=False,
    )


def create_trainer(model, training_args, train_dataset, val_dataset, args):
    """Create Hugging Face Trainer with callbacks.

    All ranks participate in distributed evaluation.
    """
    callbacks: list = [
        FormattedLoggingCallback(),
        EarlyStoppingCallback(early_stopping_patience=args.early_stopping),
    ]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=callbacks,
    )

    # Remove default logging callbacks
    trainer.remove_callback(transformers.trainer_callback.PrinterCallback)
    trainer.remove_callback(transformers.trainer_callback.ProgressCallback)

    return trainer


# ==========================================
# Main Training Loop
# ==========================================


def main():
    args = parse_gpt2_args()
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
        train_manifest=args.manifest,
        train_tokens_dir=args.tokens_dir,
        val_manifest=args.val_manifest,
        val_tokens_dir=args.val_tokens_dir,
        seed=args.seed,
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
