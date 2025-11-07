#!/usr/bin/env python3
"""
Train LSTM language model on audio tokens.

Usage:
    python train.py --manifest metadata/chunk0.csv --tokens_dir output/librivox_mhubert_expresso_2000 \
                    --embedding_dim 256 --hidden_size 512 --num_layers 2 --dropout 0.1 \
                    --batch_size 64 --learning_rate 0.001 --num_epochs 20
"""

import argparse
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

from lstm import LSTM, LSTMConfig


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
    
    def __init__(
        self,
        manifest_path: str,
        tokens_dir: str,
        split: str = "train",
        train_ratio: float = 0.9,
        sos_token_id: int = 2000,
        seed: int = 42,
        load_on_the_fly: bool = False,
    ):
        """Initialize dataset.
        
        Args:
            manifest_path: Path to CSV manifest with file_id column
            tokens_dir: Directory containing .pt token files
            split: 'train' or 'val'
            train_ratio: Fraction of data for training (rest for validation)
            sos_token_id: ID of start-of-sequence token to prepend
            seed: Random seed for reproducible split
            load_on_the_fly: Load tokens from disk on-the-fly instead of caching in memory (default: False)
        """
        self.tokens_dir = Path(tokens_dir)
        self.sos_token_id = sos_token_id
        self.load_on_the_fly = load_on_the_fly
        
        # Load manifest
        df = pd.read_csv(manifest_path)
        
        if "file_id" not in df.columns:
            raise ValueError(f"Manifest must have 'file_id' column. Found: {df.columns.tolist()}")
        
        # Shuffle and split
        file_ids = df["file_id"].tolist()
        rng = random.Random(seed)
        rng.shuffle(file_ids)
        
        split_idx = int(len(file_ids) * train_ratio)
        
        if split == "train":
            file_ids = file_ids[:split_idx]
        elif split == "val":
            file_ids = file_ids[split_idx:]
        else:
            raise ValueError(f"split must be 'train' or 'val', got {split}")
        
        print(f"{split.capitalize()} dataset: {len(file_ids)} files")
        
        if self.load_on_the_fly:
            # Store file IDs only, load from disk in __getitem__
            self.file_ids = file_ids
            print(f"Will load sequences on-the-fly from disk")
        else:
            # Load all sequences into memory with SOS prepended (default, faster)
            print(f"Loading {len(file_ids)} sequences into memory...")
            self.sequences = []
            
            for file_id in file_ids:
                token_path = self.tokens_dir / f"{file_id}.pt"
                tokens = torch.load(token_path)
                
                # Ensure 1D tensor
                if tokens.ndim > 1:
                    tokens = tokens.squeeze()
                
                # Prepend SOS token
                sos_tensor = torch.tensor([self.sos_token_id], dtype=tokens.dtype)
                complete_sequence = torch.cat([sos_tensor, tokens])
                
                self.sequences.append(complete_sequence)
            
            print(f"Loaded {len(self.sequences)} sequences into memory")
    
    def __len__(self):
        return len(self.sequences) if not self.load_on_the_fly else len(self.file_ids)
    
    def __getitem__(self, idx):
        """Return sequence (pre-loaded from memory or loaded on-the-fly)."""
        if self.load_on_the_fly:
            # Load from disk
            file_id = self.file_ids[idx]
            token_path = self.tokens_dir / f"{file_id}.pt"
            tokens = torch.load(token_path)
            
            # Ensure 1D tensor
            if tokens.ndim > 1:
                tokens = tokens.squeeze()
            
            # Prepend SOS token
            sos_tensor = torch.tensor([self.sos_token_id], dtype=tokens.dtype)
            return torch.cat([sos_tensor, tokens])
        else:
            # Return from pre-loaded memory
            return self.sequences[idx]


def collate_fn(batch):
    """Collate variable-length sequences into nested tensor.
    
    Args:
        batch: List of 1D token tensors with different lengths
        
    Returns:
        Dict with 'inputs' and 'labels' as nested tensors
    """
    # Create nested tensor from variable-length sequences
    nested_tensor = torch.nested.nested_tensor(batch)
    
    return {
        "inputs": nested_tensor,
        "labels": nested_tensor,  # Same tensor; shifting happens in model
    }


def create_checkpoint_name(
    learning_rate: float,
    hidden_size: int,
    embedding_dim: int,
    num_layers: int,
    batch_size: int,
    dropout: float,
) -> str:
    """Create descriptive checkpoint directory name from hyperparameters."""
    return f"lstm_r{learning_rate}_h{hidden_size}_e{embedding_dim}_l{num_layers}_b{batch_size}_d{dropout}"


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
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--load_on_the_fly",
        action="store_true",
        help="Load tokens on-the-fly instead of caching in memory (default: False)",
    )
    
    args = parser.parse_args()
    
    # Set global seed
    set_seed(args.seed)
    
    print("=" * 60)
    print("LSTM Training")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Manifest: {args.manifest}")
    print(f"Tokens: {args.tokens_dir}")
    print(f"Model: e{args.embedding_dim}_h{args.hidden_size}_l{args.num_layers}_d{args.dropout}")
    print(f"Training: b{args.batch_size}_r{args.learning_rate}_epochs{args.num_epochs}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"Load mode: {'on-the-fly' if args.load_on_the_fly else 'cached in memory'}")
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
    
    checkpoints_root = Path("checkpoints")
    checkpoints_root.mkdir(exist_ok=True)
    
    output_dir = checkpoints_root / checkpoint_name
    output_dir.mkdir(exist_ok=True)
    
    print(f"Checkpoints: {output_dir}\n")
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = TokenDataset(
        manifest_path=args.manifest,
        tokens_dir=args.tokens_dir,
        split="train",
        train_ratio=args.train_ratio,
        seed=args.seed,
        load_on_the_fly=args.load_on_the_fly,
    )
    
    val_dataset = TokenDataset(
        manifest_path=args.manifest,
        tokens_dir=args.tokens_dir,
        split="val",
        train_ratio=args.train_ratio,
        seed=args.seed,
        load_on_the_fly=args.load_on_the_fly,
    )
    print()

    # Create model config
    config = LSTMConfig(
        vocab_size=2001,  # 2000 audio tokens + SOS
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        sos_token_id=2000,
    )
    
    # Initialize model
    print("Initializing model...")
    model = LSTM(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        
        # Evaluate, save, and log every epoch
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        
        # Save best model
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Keep only best checkpoint
        save_total_limit=1,
        
        # Logging
        report_to="none",  # Disable wandb/tensorboard
        logging_first_step=True,
        
        # Misc
        seed=args.seed,
        dataloader_num_workers=0,  # Nested tensors may not work with multiprocessing
        remove_unused_columns=False,  # Keep our custom columns
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping)],
    )
    
    # Train
    print("Starting training...\n")
    train_result = trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(str(output_dir / "final_model"))
    
    # Print results
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Final train loss: {train_result.training_loss:.4f}")
    
    # Get best validation loss
    eval_result = trainer.evaluate()
    print(f"Final eval loss: {eval_result['eval_loss']:.4f}")
    
    # Find best epoch from logs
    if hasattr(trainer.state, 'best_metric'):
        print(f"Best eval loss: {trainer.state.best_metric:.4f}")
    
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
