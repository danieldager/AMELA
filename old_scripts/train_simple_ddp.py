#!/usr/bin/env python3
"""Simple DDP training loop - no Trainer, no Accelerate complications."""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from transformers import GPT2Config, GPT2LMHeadModel

from train_utils2 import (
    TokenDataset,
    VOCAB_SIZE,
    PAD_TOKEN_ID,
    SOS_TOKEN_ID,
)


def collate_fn(batch):
    """Pad sequences to max length in batch."""
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    labels = []
    attention_mask = []

    for item in batch:
        seq = item["input_ids"]
        pad_len = max_len - len(seq)

        padded_input = torch.cat(
            [seq, torch.full((pad_len,), PAD_TOKEN_ID, dtype=torch.long)]
        )
        input_ids.append(padded_input)

        padded_labels = torch.cat([seq, torch.full((pad_len,), -100, dtype=torch.long)])
        labels.append(padded_labels)

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


def get_inverse_sqrt_schedule(optimizer, warmup_steps):
    """Inverse sqrt learning rate schedule."""

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0 / np.sqrt(max(step, 1))

    return LambdaLR(optimizer, lr_lambda)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--train_tokens", type=str, required=True)
    parser.add_argument("--val_tokens", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--fast_debug", action="store_true", help="Quick debug run")
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
        bos_token_id=SOS_TOKEN_ID,
        eos_token_id=SOS_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
    )
    return GPT2LMHeadModel(config)


def main():
    args = parse_args()

    # DDP setup
    dist.init_process_group(
        backend="nccl",
        init_method='env://'
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    
    # CRITICAL: Tell NCCL which GPU this rank uses
    torch.cuda.set_device(device)

    print(f"\n[Rank {rank}/{world_size}] DDP initialized, device={device}")

    # Create output directory
    if args.output_dir is None:
        model_name = f"gpt2_h{args.hidden_size}_r{args.learning_rate}_l{args.num_layers}_b{args.batch_size}_d{args.dropout}"
        timestamp = datetime.now().strftime("%d-%m-%y")
        args.output_dir = f"checkpoints/{model_name}/{timestamp}"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Override for fast debug
    if args.fast_debug:
        args.max_steps = 20
        args.eval_steps = 10
        args.save_steps = 1000  # Don't save in debug mode

    # Create datasets
    print(f"[Rank {rank}] Loading train dataset from {args.train_tokens}...")
    train_dataset = TokenDataset(args.train_tokens)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, collate_fn=collate_fn
    )

    print(f"[Rank {rank}] Loading val dataset from {args.val_tokens}...")
    val_dataset = TokenDataset(args.val_tokens)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, collate_fn=collate_fn
    )

    # Create model, move to device
    print(f"[Rank {rank}] Creating model...")
    model = create_model(args)
    model.to(device)
    
    # Synchronize all ranks before DDP wrapping
    dist.barrier()
    
    model = DDP(model, device_ids=[rank])

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        weight_decay=0.0,
    )
    scheduler = get_inverse_sqrt_schedule(optimizer, args.warmup_steps)

    # Training setup
    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {n_params:,} parameters")
        print(
            f"Training: max_steps={args.max_steps}, batch_size={args.batch_size}, grad_accum={args.gradient_accumulation_steps}"
        )

    model.train()
    global_step = 0
    accumulation_step = 0
    start_time = time.time()

    print(f"\n[Rank {rank}] Starting training loop...\n")

    # Main training loop
    train_iterator = iter(train_loader)
    while global_step < args.max_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            # Restart iterator when exhausted
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss / args.gradient_accumulation_steps

        # Backward pass
        loss.backward()
        accumulation_step += 1

        # Optimizer step
        if accumulation_step >= args.gradient_accumulation_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            accumulation_step = 0

            # Logging
            if rank == 0 and global_step % 100 == 0:
                elapsed = time.time() - start_time
                throughput = global_step / elapsed
                eta_hours = (
                    (args.max_steps - global_step) / throughput / 3600
                    if throughput > 0
                    else 0
                )
                print(
                    f"[{time.strftime('%H:%M:%S')}] Step {global_step}/{args.max_steps} | {throughput:.2f} steps/s | ETA: {eta_hours:.1f}h"
                )

            # Evaluation
            if global_step % args.eval_steps == 0 and global_step > 0:
                model.eval()
                eval_loss = 0.0
                eval_count = 0

                with torch.no_grad():
                    val_iterator = iter(val_loader)
                    for _ in range(
                        min(100, len(val_dataset))
                    ):  # Limit eval to 100 batches
                        try:
                            batch = next(val_iterator)
                        except StopIteration:
                            break

                        batch = {k: v.to(device) for k, v in batch.items()}
                        outputs = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                        )
                        eval_loss += outputs.loss.item()
                        eval_count += 1

                if eval_count > 0:
                    eval_loss /= eval_count
                    if rank == 0:
                        print(
                            f"[Rank {rank}] Step {global_step}: eval_loss={eval_loss:.4f}, ppl={np.exp(eval_loss):.2f}"
                        )

                model.train()

            # Save checkpoint
            if global_step % args.save_steps == 0 and global_step > 0 and rank == 0:
                checkpoint_dir = f"{args.output_dir}/checkpoint-{global_step}"
                Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                model.module.save_pretrained(checkpoint_dir)
                print(f"Saved checkpoint to {checkpoint_dir}")

    # Final save
    if rank == 0:
        model.module.save_pretrained(args.output_dir)
        elapsed = time.time() - start_time
        print(f"\nTraining complete!")
        print(f"Total time: {elapsed/3600:.1f}h / {elapsed/60:.1f}m")
        print(f"Model saved to {args.output_dir}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
