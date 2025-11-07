#!/usr/bin/env python3
"""
LSTM language model for audio token sequences.

Uses torch.nested_tensor for efficient variable-length sequence handling.
"""

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel


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
    """LSTM language model with nested tensor support for variable-length sequences."""
    
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
        inputs: torch.Tensor | torch.nested.nested_tensor,
        labels: torch.Tensor | torch.nested.nested_tensor | None = None,
        return_dict: bool = True,
    ):
        """Forward pass with nested tensor support.
        
        Args:
            inputs: Token IDs (nested tensor for variable lengths or regular tensor)
            labels: Target token IDs for loss calculation (same format as inputs)
            return_dict: Whether to return dict or tuple
            
        Returns:
            Dict with 'loss' and 'logits' if return_dict=True, else tuple
        """
        # Check if we're using nested tensors
        is_nested = isinstance(inputs, torch.nested.nested_tensor)
    
        embeddings = self.embedding(inputs)
        lstm_output, _ = self.lstm(embeddings)
        logits = self.output(lstm_output)
    
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            if is_nested:
                # For nested tensors, we need to handle loss calculation carefully
                # Shift logits and labels for next-token prediction
                # This is tricky with nested tensors, so we'll convert to packed format
                loss = self._compute_nested_loss(logits, labels)
            else:
                # Standard loss calculation with shifted sequences
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1)
                )
        
        if return_dict:
            return {"loss": loss, "logits": logits}
        return (loss, logits)
    
    def _compute_nested_loss(
        self,
        logits: torch.nested.nested_tensor,
        labels: torch.nested.nested_tensor,
    ) -> torch.Tensor:
        """Compute loss for nested tensors by converting to packed format.
        
        Args:
            logits: Nested tensor of logits [batch_size, var_seq_len, vocab_size]
            labels: Nested tensor of labels [batch_size, var_seq_len]
            
        Returns:
            Scalar loss tensor
        """
        # Convert nested tensors to packed format for loss calculation
        # This unbinds the nested tensor into a list of tensors
        logits_list = logits.unbind()
        labels_list = labels.unbind()
    
        total_loss = 0.0
        total_tokens = 0
        
        for seq_logits, seq_labels in zip(logits_list, labels_list):
            # Shift for next-token prediction
            shift_logits = seq_logits[:-1, :]  # [seq_len-1, vocab_size]
            shift_labels = seq_labels[1:]      # [seq_len-1]
            
            # Compute loss for this sequence
            loss_fct = nn.CrossEntropyLoss()
            seq_loss = loss_fct(shift_logits, shift_labels)
            seq_len = shift_labels.size(0)
            
            # Accumulate weighted by sequence length
            total_loss += seq_loss * seq_len
            total_tokens += seq_len
        
        # Return average loss per token
        return total_loss / total_tokens if total_tokens > 0 else torch.tensor(0.0)
    
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
            temperature: Sampling temperature
            top_k: Top-k filtering (0 = disabled)
            top_p: Nucleus sampling threshold
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
            # Process initial sequence
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
