#!/usr/bin/env python3
"""Minimal training utilities - simple and functional."""

import os
import time
import numpy as np
import torch
import webdataset as wds
from torch.utils.data import IterableDataset
from transformers import TrainerCallback
from pathlib import Path


# Token vocabulary
NUM_AUDIO_TOKENS = 2000
SOS_TOKEN_ID = NUM_AUDIO_TOKENS
PAD_TOKEN_ID = NUM_AUDIO_TOKENS + 1
VOCAB_SIZE = NUM_AUDIO_TOKENS + 2
MAX_SEQ_LEN = 2048


def is_rank_zero():
    """Check if main process."""
    return int(os.environ.get("RANK", 0)) == 0


class TokenDataset(IterableDataset):
    """Load token sequences from webdataset shards. That's it."""

    def __init__(self, tokens_dir, max_seq_len=MAX_SEQ_LEN):
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        print(f"[Rank {rank}/{world_size}] TokenDataset.__init__() called for {tokens_dir}")
        
        self.tokens_dir = Path(tokens_dir)
        self.max_seq_len = max_seq_len

        # Find all shards
        self.shards = sorted(self.tokens_dir.glob("shard-*.tar"))
        if not self.shards:
            raise ValueError(f"No shards found in {self.tokens_dir}")

        # Convert to strings for webdataset
        self.shard_paths = [str(s) for s in self.shards]

        print(f"[Rank {rank}] Found {len(self.shards)} shards: {[Path(s).name for s in self.shard_paths]}")
        print(f"[Rank {rank}] TokenDataset.__init__() completed")

    def __iter__(self):
        """Stream samples from shards with proper DDP distribution."""
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"[Rank {rank}/{world_size}] TokenDataset.__iter__() called at {time.strftime('%H:%M:%S')}")
        print(f"[Rank {rank}] Total shards available: {len(self.shard_paths)}")
        print(f"[Rank {rank}] Shard names: {[Path(s).name for s in self.shard_paths]}")
        
        # Pass ALL shards to webdataset, let it handle DDP distribution
        print(f"[Rank {rank}] Creating WebDataset...")
        dataset = wds.WebDataset(self.shard_paths, shardshuffle=False, nodesplitter=wds.split_by_node)
        print(f"[Rank {rank}] WebDataset created ({time.time()-start_time:.2f}s)")
        
        print(f"[Rank {rank}] Adding decode()...")
        dataset = dataset.decode()
        print(f"[Rank {rank}] decode() added ({time.time()-start_time:.2f}s)")
        
        print(f"[Rank {rank}] Adding repeat()...")
        dataset = dataset.repeat()
        print(f"[Rank {rank}] repeat() added ({time.time()-start_time:.2f}s)")
        
        print(f"[Rank {rank}] Pipeline setup complete, starting iteration...")
        print(f"{'='*60}\n")
        
        sample_count = 0
        yield_count = 0
        for sample in dataset:
            sample_count += 1
            if sample_count <= 5:
                print(f"[Rank {rank}] Processing sample {sample_count}...")
            try:
                # Extract tokens - key has full suffix, so find the one ending with .token.npy
                token_key = [k for k in sample.keys() if k.endswith('.token.npy')][0]
                tokens = sample[token_key]
                tokens = torch.from_numpy(tokens).squeeze().long()

                # Skip empty
                if tokens.ndim == 0 or len(tokens) == 0:
                    continue

                # Skip invalid token IDs
                if tokens.min() < 0 or tokens.max() >= NUM_AUDIO_TOKENS:
                    continue

                # Truncate if needed
                if len(tokens) > self.max_seq_len:
                    tokens = tokens[: self.max_seq_len]

                # Add SOS token
                sequence = torch.cat(
                    [torch.tensor([SOS_TOKEN_ID], dtype=torch.long), tokens]
                )

                yield_count += 1
                if yield_count <= 5:
                    print(f"[Rank {rank}] ✓ Yielded sample {yield_count} (seq_len={len(sequence)})")
                
                yield {"input_ids": sequence}

            except Exception as e:
                # Skip corrupted samples
                if sample_count <= 10:  # Log early errors
                    print(f"[Rank {rank}] ✗ Error processing sample {sample_count}: {e}")
                continue


class LoggingCallback(TrainerCallback):
    """Log training progress with timing info."""

    def __init__(self):
        self.header_printed = False
        self.train_logs = {}
        self.last_time = None
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        """Log training speed every N steps."""
        rank = int(os.environ.get("RANK", 0))
        if rank != 0:
            return

        if state.global_step % 100 == 0 and state.global_step > 0:
            elapsed = time.time() - self.start_time
            throughput = state.global_step / elapsed
            eta_sec = (
                (args.max_steps - state.global_step) / throughput
                if throughput > 0
                else 0
            )
            print(
                f"[{time.strftime('%H:%M:%S')}] Step {state.global_step}/{args.max_steps} | {throughput:.2f} steps/s | ETA: {eta_sec/3600:.1f}h"
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or int(os.environ.get("RANK", 0)) != 0:
            return

        # Store training logs
        if "loss" in logs and "eval_loss" not in logs:
            self.train_logs = logs.copy()
            return

        # Only print when we have eval results
        if "eval_loss" not in logs:
            return

        # Print header once
        if not self.header_printed:
            print("\n step    loss    v_loss    ppl     v_ppl      lr       time")
            print("------  -------  -------  ------  ------  ---------  --------")
            self.header_printed = True
            self.last_time = time.time()

        # Calculate elapsed time
        current_time = time.time()
        elapsed = current_time - self.last_time if self.last_time else 0
        self.last_time = current_time

        # Extract metrics
        step = state.global_step
        loss = self.train_logs.get("loss", 0)
        v_loss = logs.get("eval_loss", 0)
        ppl = np.exp(loss) if loss else 0
        v_ppl = np.exp(v_loss) if v_loss else 0
        lr = self.train_logs.get("learning_rate", 0)

        print(
            f"{int(step):6d}  {loss:7.4f}  {v_loss:7.4f}  {ppl:6.1f}  {v_ppl:6.1f}  {lr:9.6f}  {elapsed:8.1f}"
        )


def print_final_results(trainer):
    """Print training summary."""
    if not is_rank_zero():
        return

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)

    results = trainer.evaluate()
    print(f"Final eval loss: {results['eval_loss']:.4f}")
    print(f"Final eval perplexity: {np.exp(results['eval_loss']):.2f}")

    if hasattr(trainer.state, "best_metric") and trainer.state.best_metric:
        print(f"Best eval loss: {trainer.state.best_metric:.4f}")
        print(f"Best eval perplexity: {np.exp(trainer.state.best_metric):.2f}")

    print("=" * 60)
