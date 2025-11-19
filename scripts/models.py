#!/usr/bin/env python3
"""
Language models for audio token sequences.

Implements LSTM-based language model with pack_padded_sequence for efficient
variable-length sequence handling on GPU.
"""

import re
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import PretrainedConfig, PreTrainedModel


# Constants
PAD_TOKEN_ID = 2001  # Padding token (beyond vocab: 0-1999 audio tokens, 2000 SOS)


class LSTMConfig(PretrainedConfig):
    """Configuration for LSTM language model."""
    
    model_type = "lstm"
    
    def __init__(
        self,
        vocab_size: int = 2001,  # 2000 audio tokens + 1 SOS token
        embedding_dim: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        sos_token_id: int = 2000,
        **kwargs,
    ):
        """Initialize LSTM configuration.
        
        Args:
            vocab_size: Size of vocabulary (2000 audio tokens + SOS)
            embedding_dim: Dimension of token embeddings
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout probability (applied if num_layers > 1)
            sos_token_id: ID of the start-of-sequence token
        """
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.sos_token_id = sos_token_id


class LSTM(PreTrainedModel):
    """LSTM language model with pack_padded_sequence for efficient variable-length sequences."""
    
    config_class = LSTMConfig
    
    def __init__(self, config: LSTMConfig):
        """Initialize LSTM model."""
        super().__init__(config)
        
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True,
        )
        self.output = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(
        self,
        inputs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """Forward pass with pack_padded_sequence support.
        
        Args:
            inputs: Padded token IDs [batch_size, max_seq_len]
            labels: Target token IDs for loss calculation (same format as inputs)
            lengths: Actual sequence lengths before padding [batch_size]
            return_dict: Whether to return dict or tuple
            
        Returns:
            Dict with 'loss' and 'logits' if return_dict=True, else tuple
        """
        embeddings = self.embedding(inputs)
        
        # Use pack_padded_sequence if lengths provided
        if lengths is not None:
            # Move lengths to CPU only once if not already there
            if lengths.device.type != 'cpu':
                lengths_cpu = lengths.cpu()
            else:
                lengths_cpu = lengths
            
            # Pack the padded embeddings with their actual lengths
            packed_embeddings = pack_padded_sequence(
                embeddings, 
                lengths_cpu,
                batch_first=True, 
                enforce_sorted=False  # Allow unsorted lengths
            )
            
            # LSTM processes packed sequence (skips padding!)
            packed_output, _ = self.lstm(packed_embeddings)
            
            # Unpack back to padded format
            lstm_output, _ = pad_packed_sequence(
                packed_output, 
                batch_first=True,
                total_length=inputs.size(1)  # Ensure same length as input
            )
        else:
            # Fallback: standard padded processing (processes padding too)
            lstm_output, _ = self.lstm(embeddings)
        
        logits = self.output(lstm_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction: input[:-1] predicts target[1:]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Ensure labels are long (int64) for CrossEntropyLoss on CUDA
            shift_labels = shift_labels.long()
            
            # Use ignore_index to skip padding in loss calculation
            # Padding token is beyond vocab (0-1999 audio tokens, 2000 SOS)
            loss_fct = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
        
        if return_dict:
            return {"loss": loss, "logits": logits}
        return (loss, logits)
    
    def generate(
        self,
        inputs: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Generate audio token sequences autoregressively.
        
        Args:
            inputs: Starting tokens [batch_size, seq_len] (typically just SOS token)
            max_length: Maximum length to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k filtering (0 = disabled)
            top_p: Nucleus sampling threshold (1.0 = disabled)
            device: Device to generate on
            
        Returns:
            Generated token sequences [batch_size, max_length]
        """
        self.eval()
        batch_size = inputs.shape[0]
        
        # Initialize hidden state
        h0 = torch.zeros(self.config.num_layers, batch_size, self.config.hidden_size, device=device)
        c0 = torch.zeros(self.config.num_layers, batch_size, self.config.hidden_size, device=device)
        hidden_state = (h0, c0)
        
        generated = inputs.clone()
        
        with torch.no_grad():
            # Process initial sequence if provided
            if inputs.shape[1] > 0:
                embeddings = self.embedding(inputs)
                _, hidden_state = self.lstm(embeddings, hidden_state)
            
            # Generate tokens one by one
            for _ in range(max_length - inputs.shape[1]):
                # Process last token with maintained hidden state
                last_token = generated[:, -1:]
                embeddings = self.embedding(last_token)
                lstm_output, hidden_state = self.lstm(embeddings, hidden_state)
                logits = self.output(lstm_output)
                
                # Sample next token
                next_logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    mask = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[mask] = float('-inf')
                
                # Nucleus (top-p) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    mask = cum_probs > top_p
                    mask[..., 1:] = mask[..., :-1].clone()
                    mask[..., 0] = 0
                    
                    # Map back to original indices
                    mask = mask.scatter(1, sorted_indices, mask)
                    next_logits[mask] = float('-inf')

                # Sample from distribution
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated


def load_model_from_name(
    model_name: str,
    dataset_name: str,
    checkpoint: Optional[str] = None,
    device: str = "cuda",
) -> LSTM:
    """Load LSTM model from checkpoint directory name.
    
    Parses model architecture from the directory name format:
    lstm_h{hidden}_r{lr}_e{emb}_l{layers}_b{batch}_d{dropout}
    
    Args:
        model_name: Checkpoint directory name (e.g., "lstm_h512_r0.0003_e256_l2_b256_d0.1")
        dataset_name: Dataset name (e.g., "chunk0", "chunk0-23")
        checkpoint: Optional checkpoint to load. Can be:
            - None: Load best model from "final_model/" (default)
            - Integer (e.g., "5"): Load from "checkpoint-5/"
            - String path: Load from custom path
        device: Device to load model on
        
    Returns:
        Loaded LSTM model in eval mode
        
    Raises:
        ValueError: If model_name is not properly formatted
        FileNotFoundError: If checkpoint directory doesn't exist
    """
    # Parse model config from name
    pattern = r"lstm_h(\d+)_r([0-9.]+)_e(\d+)_l(\d+)_b(\d+)_d([0-9.]+)"
    match = re.match(pattern, model_name)
    
    if not match:
        raise ValueError(
            f"Model name '{model_name}' doesn't match expected format: "
            "lstm_h{{hidden}}_r{{lr}}_e{{emb}}_l{{layers}}_b{{batch}}_d{{dropout}}"
        )

    hidden_size, _, embedding_dim, num_layers, _, dropout = match.groups()

    # Create config
    config = LSTMConfig(
        vocab_size=2002,  # 2000 tokens + SOS + PAD
        embedding_dim=int(embedding_dim),
        hidden_size=int(hidden_size),
        num_layers=int(num_layers),
        dropout=float(dropout),
        sos_token_id=2000,
    )
    
    # Determine checkpoint path: checkpoints/model_name/dataset_name/timestamp/
    checkpoints_root = Path("checkpoints")
    model_dir = checkpoints_root / model_name / dataset_name
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Find the latest timestamp directory (e.g., 13-11-25)
    timestamp_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir()])
    if not timestamp_dirs:
        raise FileNotFoundError(f"No timestamp directories found in {model_dir}")
    
    latest_dir = timestamp_dirs[-1]  # Last one alphabetically is most recent
    
    if checkpoint is None:
        # Default: load best model
        checkpoint_path = latest_dir / "final_model"
    elif checkpoint.isdigit():
        # Load specific epoch checkpoint
        checkpoint_path = latest_dir / f"checkpoint-{checkpoint}"
    else:
        # Custom path
        checkpoint_path = Path(checkpoint)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading model from: {checkpoint_path}")
    
    # Load model
    model = LSTM.from_pretrained(checkpoint_path, config=config)
    model.to(device) # type: ignore
    model.eval()
    
    return model
