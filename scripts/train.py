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
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback
import transformers.trainer_callback

from models import LSTM, LSTMConfig

# Suppress warnings
warnings.filterwarnings("ignore", message="Could not estimate the number of tokens")
warnings.filterwarnings("ignore", message=".*barrier.*device_id.*")


# Constants
NUM_AUDIO_TOKENS = 2000  # mHuBERT/EnCodec codebook size
SOS_TOKEN_ID = NUM_AUDIO_TOKENS
PAD_TOKEN_ID = NUM_AUDIO_TOKENS + 1
VOCAB_SIZE = NUM_AUDIO_TOKENS + 2


# DDP helper: Check if we're in distributed mode and get rank
def is_main_process():
    """Return True if this is rank 0 or not using DDP."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    return local_rank == -1 or local_rank == 0


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
        if not logs:
            return
        
        # Only print on main process (rank 0)
        if not is_main_process():
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
        
        # Calculate total epoch time
        epoch_s = int(time.time() - self.t_start) if self.t_start else 0
        
        # Extract values
        epoch = logs.get("epoch", 0)
        loss = self.train_logs.get("loss", 0)
        ppl = np.exp(loss) if loss else 0
        val_loss = logs.get("eval_loss", 0)
        val_ppl = np.exp(val_loss) if val_loss else 0
        lr = self.train_logs.get("learning_rate", 0)
        
        # Print formatted row
        print(f"{epoch:5.2f}  {loss:7.4f}  {ppl:8.1f}  {val_loss:8.4f}  {val_ppl:8.1f}  "
              f"{epoch_s:7}  {lr:9.6f}", flush=True)


def set_seed(seed: int = 42):
    """Set global seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class TokenDataset(Dataset):
    """Dataset for loading audio tokens from .pt files."""
    
    def __init__(self, manifest_path, tokens_dir, split="train", train_ratio=0.9, seed=42, max_seq_len=3000):
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
        skipped_invalid = [] # To track skipped files
        
        for file_id in file_ids:
            token_path = self.tokens_dir / f"{file_id}.pt"
            if not token_path.exists():
                continue
            
            tokens = torch.load(token_path).squeeze().long()
            
            # Skip invalid tokens (0-d tensors or empty sequences)
            if tokens.ndim == 0 or len(tokens) == 0:
                skipped_invalid.append((file_id, tokens.ndim, len(tokens) if tokens.ndim > 0 else 0))
                continue
            
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
                truncated += 1
            
            # Prepend SOS
            sequence = torch.cat([torch.tensor([SOS_TOKEN_ID], dtype=torch.long), tokens])
            self.sequences.append(sequence)
        
        if is_main_process():
            print(f"Loaded {len(self.sequences)} sequences")
            if truncated > 0:
                print(f"Truncated {truncated} sequences to max length {self.max_seq_len}")
            if skipped_invalid: # Warn about skipped invalid files
                print(f"WARNING: Skipped {len(skipped_invalid)} invalid files (0-d tensor or empty)")
                for file_id, ndim, length in skipped_invalid[:5]:  # Show first 5
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


def create_checkpoint_name(learning_rate, hidden_size, embedding_dim, num_layers, batch_size, dropout):
    """Create checkpoint directory name from hyperparameters."""
    return f"lstm_h{hidden_size}_r{learning_rate}_e{embedding_dim}_l{num_layers}_b{batch_size}_d{dropout}"


def main():
    parser = argparse.ArgumentParser(
        description="Train LSTM language model on audio tokens"
    )
    
    # Data arguments
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to CSV manifest with file_id column",
    )
    parser.add_argument(
        "--tokens_dir",
        type=str,
        required=True,
        help="Directory containing .pt token files",
    )
    
    # Model hyperparameters
    parser.add_argument(
        "--embedding_dim",
        type=int,
        required=True,
        help="Dimension of token embeddings",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        required=True,
        help="LSTM hidden state size",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        required=True,
        help="Number of LSTM layers",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        required=True,
        help="Dropout probability",
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=True,
        help="Learning rate",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        required=True,
        help="Number of training epochs",
    )
    
    # Optional arguments
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Fraction of data for training (default: 0.9)",
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=10,
        help="Early stopping patience in epochs (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps (default: 1)",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers for parallel data loading (default: 4)",
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Use mixed precision (FP16) training for speedup (default: False)",
    )
    parser.add_argument(
        "--use_bf16",
        action="store_true",
        help="Use mixed precision (BF16) training for speedup (default: False)",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=5.0,
        help="Maximum gradient norm for clipping (default: 5.0)",
    )
    parser.add_argument(
        "--group_by_length",
        action="store_true",
        help="Group sequences by similar lengths to reduce padding waste (default: False)",
    )
    
    args = parser.parse_args()
    
    # Set global seed
    set_seed(args.seed)
    
    script_start_time = time.time()
    
    if is_main_process():
        print("=" * 60)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Manifest: {args.manifest}")
        print(f"Tokens: {args.tokens_dir}")
        print(f"Gradient accumulation: {args.gradient_accumulation_steps} steps")
        print(f"DataLoader workers: {args.dataloader_num_workers}")
        print(f"Seed: {args.seed}")
        print("=" * 60)
        print()
    
    # Create checkpoint directory
    checkpoint_name = create_checkpoint_name(
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    
    # Extract dataset name from manifest path (e.g., "librivox_29-10-25.csv" -> "librivox")
    manifest_path = Path(args.manifest)
    dataset_name = manifest_path.stem.split('_')[0]  # Split on '_' and take first part
    
    checkpoints_root = Path("checkpoints")
    checkpoints_root.mkdir(exist_ok=True)
    
    # Structure: checkpoints/model_name/dataset_name/DD-MM-YY/
    timestamp = datetime.now().strftime("%d-%m-%y")
    output_dir = checkpoints_root / checkpoint_name / dataset_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if is_main_process():
        print(f"Checkpoints: {output_dir}\n")
    
    # Create datasets
    if is_main_process():
        print("Loading datasets...")
    train_dataset = TokenDataset(
        manifest_path=args.manifest,
        tokens_dir=args.tokens_dir,
        split="train",
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    
    val_dataset = TokenDataset(
        manifest_path=args.manifest,
        tokens_dir=args.tokens_dir,
        split="val",
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    # Create model config
    config = LSTMConfig(
        vocab_size=VOCAB_SIZE,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        sos_token_id=SOS_TOKEN_ID,
    )
    
    # Initialize model
    if is_main_process():
        print("Initializing model...")
    model = LSTM(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process():
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}\n")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",  # No LR decay (default is "linear")
        
        # Gradient accumulation for larger effective batch size
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        
        # Mixed precision training (2-3x speedup on modern GPUs)
        fp16=args.use_fp16,
        bf16=args.use_bf16,
        
        # Evaluate, save, and log every epoch
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        
        # Save best model
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Keep best 3 checkpoints
        save_total_limit=3,
        
        # Logging
        report_to="none",  # Disable wandb/tensorboard
        disable_tqdm=True,  # Disable progress bars (we handle logging in callback)
        
        # DDP optimization
        ddp_find_unused_parameters=False,  # Our LSTM uses all parameters
        
        # DataLoader optimization
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,  # Faster CPU-GPU transfer
        group_by_length=args.group_by_length,  # Group sequences by length to reduce padding
        
        # Misc
        seed=args.seed,
        remove_unused_columns=False,  # Keep our custom columns
    )
    
    # Create trainer
    if is_main_process():
        print("Creating trainer...")
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
    
    # Remove default logging callbacks that print dicts
    # Trainer adds PrinterCallback or ProgressCallback by default
    trainer.remove_callback(transformers.trainer_callback.PrinterCallback)
    trainer.remove_callback(transformers.trainer_callback.ProgressCallback)
    
    # Print precision info
    if is_main_process():
        print(f"Training precision: ", end="")
        if training_args.fp16:
            print("FP16 (half precision)")
        elif training_args.bf16:
            print("BF16 (bfloat16)")
        else:
            print("FP32 (full precision)")
        print()
    
    # Train
    if is_main_process():
        print("Starting training...")
    train_result = trainer.train()
    
    # Save final model
    if is_main_process():
        print("\nSaving final model...")
    trainer.save_model(str(output_dir / "final_model"))
    
    # Calculate total script time
    total_duration = time.time() - script_start_time
    
    # Print results (only on main process)
    if is_main_process():
        print("\n" + "=" * 60)
        print("Training Complete")
        print("=" * 60)
        
        # Get final evaluation metrics
        eval_result = trainer.evaluate()
        print(f"Final eval loss: {eval_result['eval_loss']:.4f}")
        print(f"Final eval perplexity: {np.exp(eval_result['eval_loss']):.2f}")
        
        # Best model metrics (from early stopping)
        if hasattr(trainer.state, 'best_metric') and trainer.state.best_metric is not None:
            print(f"Best eval loss: {trainer.state.best_metric:.4f}")
            print(f"Best eval perplexity: {np.exp(trainer.state.best_metric):.2f}")
        
        print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total runtime: {total_duration / 60:.1f} min")
        print("=" * 60)


if __name__ == "__main__":
    main()
