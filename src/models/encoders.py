"""
Sequence encoders for protein functionality prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceEncoder(nn.Module):
    """
    Lightweight sequence encoder for protein sequences.

    Args:
        vocab_size (int): Size of amino acid vocabulary
        embed_dim (int): Embedding dimension
        max_length (int): Maximum sequence length
        pooling (str): Pooling strategy ('mean', 'max', 'attention')
    """

    def __init__(self, vocab_size=22, embed_dim=128, max_length=1024, pooling='mean'):
        super().__init__()
        self.pooling = pooling

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        if pooling == 'attention':
            self.attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x_ids):
        """
        Forward pass through sequence encoder.

        Args:
            x_ids (torch.Tensor): Token IDs [batch, seq_len]

        Returns:
            torch.Tensor: Sequence embeddings [batch, embed_dim]
        """
        batch_size, seq_len = x_ids.shape

        # Get embeddings
        emb = self.embedding(x_ids)  # [batch, seq_len, embed_dim]

        # Apply pooling
        if self.pooling == 'mean':
            # Create mask for non-padding tokens
            mask = (x_ids != 0).float().unsqueeze(-1)
            pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        elif self.pooling == 'max':
            pooled = F.adaptive_max_pool1d(emb.transpose(1, 2), 1).squeeze(-1)
        elif self.pooling == 'attention':
            # Add CLS token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            emb_with_cls = torch.cat([cls_tokens, emb], dim=1)

            # Self-attention
            attended, _ = self.attention(emb_with_cls, emb_with_cls, emb_with_cls)
            pooled = attended[:, 0]  # Use CLS token representation
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        return pooled


class ESMEncoder(nn.Module):
    """
    Wrapper for ESM protein language model.

    Args:
        model_name (str): ESM model name
        freeze_layers (int): Number of layers to freeze (0 = no freezing)
        pooling (str): Pooling strategy for sequence-level representation
    """

    def __init__(self, model_name='facebook/esm2_t6_8M_UR50D', freeze_layers=0, pooling='mean'):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pooling = pooling

        # Freeze specified layers
        if freeze_layers > 0:
            for param in list(self.model.parameters())[:freeze_layers]:
                param.requires_grad = False

    def tokenize(self, sequences, max_length=1024):
        """
        Tokenize protein sequences.

        Args:
            sequences (list): List of amino acid sequences
            max_length (int): Maximum sequence length

        Returns:
            dict: Tokenized sequences ready for model input
        """
        return self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )

    def forward(self, sequences=None, tokens=None):
        """
        Forward pass through ESM encoder.

        Args:
            sequences (list, optional): List of amino acid sequences
            tokens (dict, optional): Pre-tokenized sequences

        Returns:
            torch.Tensor: Sequence embeddings [batch, hidden_dim]
        """
        if tokens is None:
            if sequences is None:
                raise ValueError("Either sequences or tokens must be provided")
            tokens = self.tokenize(sequences)

        # Move tokens to same device as model
        device = next(self.model.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}

        # Forward pass
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.model(**tokens)
            hidden_states = outputs.hidden_states[-1]  # Last layer

        # Apply pooling
        if self.pooling == 'mean':
            # Mean pooling over sequence length
            embeddings = hidden_states.mean(dim=1)
        elif self.pooling == 'cls':
            # Use CLS token (first token)
            embeddings = hidden_states[:, 0]
        elif self.pooling == 'max':
            # Max pooling over sequence length
            embeddings = hidden_states.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        return embeddings