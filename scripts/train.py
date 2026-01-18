#!/usr/bin/env python3
"""Simple GPT-2 training script."""

import argparse
import sys
import time
import os
from pathlib import Path
from datetime import datetime

import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from train_utils2 import (
    TokenDataset,
    LoggingCallback,
    print_final_results,
    VOCAB_SIZE,
    PAD_TOKEN_ID,
)


def collate_fn(batch):
    """Pad sequences and create labels for causal LM."""
    # Get max length in batch
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids = []
    labels = []
    attention_mask = []

    for item in batch:
        seq = item["input_ids"]
        pad_len = max_len - len(seq)

        # Pad input_ids with PAD_TOKEN_ID (same as in dataset)
        padded_input = torch.cat(
            [seq, torch.full((pad_len,), PAD_TOKEN_ID, dtype=torch.long)]
        )
        input_ids.append(padded_input)

        # Labels: same as input, but -100 for padding (ignored in loss)
        padded_labels = torch.cat([seq, torch.full((pad_len,), -100, dtype=torch.long)])
        labels.append(padded_labels)

        # Attention mask: 1 for real tokens, 0 for padding
        mask = torch.cat(
            [
                torch.ones(len(seq), dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long),
            ]
        )
        attention_mask.append(mask)

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


def parse_args():
    parser = argparse.ArgumentParser()

    # Model architecture
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=4)

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)

    # Dataset paths (passed from SLURM)
    parser.add_argument("--train_tokens", type=str, required=True)
    parser.add_argument("--val_tokens", type=str, required=True)

    # Optional
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--fast_debug", action="store_true", help="Run a short, no-eval debug loop"
    )

    return parser.parse_args()


def create_model(args):
    """Create GPT-2 model."""
    config = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_positions=2048,
        n_embd=args.hidden_size,
        n_layer=args.num_layers,
        n_head=args.num_heads,
        resid_pdrop=args.dropout,
        embd_pdrop=args.dropout,
        attn_pdrop=args.dropout,
        bos_token_id=VOCAB_SIZE - 2,  # SOS token
        eos_token_id=VOCAB_SIZE - 2,
        pad_token_id=PAD_TOKEN_ID,
        # use_cache=False,
    )

    model = GPT2LMHeadModel(config)
    # model.config.attn_implementation = "eager"

    return model


def main():
    start_time = time.time()
    args = parse_args()

    # Create output directory
    if args.output_dir is None:
        model_name = f"gpt2_h{args.hidden_size}_r{args.learning_rate}_l{args.num_layers}_b{args.batch_size}_d{args.dropout}"
        timestamp = datetime.now().strftime("%d-%m-%y")
        args.output_dir = f"checkpoints/{model_name}/{timestamp}"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Create datasets
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    print(f"\n{'='*60}")
    print(
        f"[Rank {rank}/{world_size}] TRAINING SCRIPT START at {datetime.now().strftime('%H:%M:%S')}"
    )
    print(f"[Rank {rank}] Train tokens: {args.train_tokens}")
    print(f"[Rank {rank}] Val tokens: {args.val_tokens}")
    print(f"{'='*60}\n")

    print(f"[Rank {rank}] Creating train dataset...")
    train_dataset = TokenDataset(args.train_tokens)
    print(f"[Rank {rank}] Train dataset created")

    # Quick smoke test: fetch one sample to ensure dataset iteration works per rank
    try:
        sample = next(iter(train_dataset))
        seq_len = len(sample["input_ids"]) if sample and "input_ids" in sample else -1
        print(f"[Rank {rank}] Sampled one train sequence (len={seq_len})")
    except Exception as e:
        print(f"[Rank {rank}] Error sampling train dataset: {e}")

    print(f"[Rank {rank}] Creating val dataset...")
    val_dataset = TokenDataset(args.val_tokens)
    print(f"[Rank {rank}] Val dataset created")

    # Create model
    print(f"[Rank {rank}] Creating model at {datetime.now().strftime('%H:%M:%S')}...")
    model = create_model(args)

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating trainer...")

    # Fast debug overrides (short run, no eval/checkpoints)
    if args.fast_debug:
        os.environ.setdefault("WANDB_MODE", "offline")
        max_steps = 20
        eval_strategy = "no"
        save_strategy = "no"
        logging_steps = 1
        run_name = "fast_debug"
    else:
        max_steps = args.max_steps
        eval_strategy = "steps"
        save_strategy = "steps"
        logging_steps = args.eval_steps
        run_name = Path(args.output_dir).parent.name

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        # Training schedule
        max_steps=max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="inverse_sqrt",
        # Batch sizes
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        # Evaluation
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps,
        eval_do_concat_batches=False,  # Don't try to concatenate variable-length batches
        # Checkpointing
        save_strategy=save_strategy,
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=not args.fast_debug,
        metric_for_best_model="eval_loss",
        # Logging
        logging_steps=logging_steps,
        report_to=[] if args.fast_debug else "wandb",
        run_name=run_name,
        disable_tqdm=True,
        # DDP settings
        ddp_find_unused_parameters=False,
        # IterableDataset requirements
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        dataloader_drop_last=False,
        # Optimization
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=0.0,
        adam_beta1=0.9,
        adam_beta2=0.98,
        bf16=True,
        # Other
        remove_unused_columns=False,
    )

    # Create trainer
    print(
        f"[Rank {rank}] Creating Trainer object at {datetime.now().strftime('%H:%M:%S')}..."
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=[
            LoggingCallback(),
            EarlyStoppingCallback(early_stopping_patience=5),
        ],
    )
    print(f"[Rank {rank}] Trainer created at {datetime.now().strftime('%H:%M:%S')}")

    # Train
    print(
        f"\n[Rank {rank}] ===== CALLING trainer.train() at {datetime.now().strftime('%H:%M:%S')} =====\n"
    )

    trainer.train(resume_from_checkpoint=args.resume if args.resume else None)

    print(
        f"\n[Rank {rank}] ===== TRAINING COMPLETED at {datetime.now().strftime('%H:%M:%S')} =====\n"
    )

    # Final evaluation
    print_final_results(trainer)

    # Save final model
    if rank == 0:
        elapsed = time.time() - start_time
        trainer.save_model(args.output_dir)
        print(f"Model saved to {args.output_dir}")
        print(f"Total (h/m) : {elapsed/3600:.1f} / {elapsed/60:.1f}")


if __name__ == "__main__":
    main()
