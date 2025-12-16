"""
Multi-modal fusion architecture for combining sequence and MSA features.

Implements Stage 2 architecture that fuses:
- ESM2 sequence embeddings
- MSA evolutionary features
- Cross-attention fusion mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class MSAEncoder(nn.Module):
    """
    Encode MSA features into dense representations.

    Takes PSSM-like MSA features and projects them into same
    dimension as ESM2 embeddings for fusion.

    Args:
        msa_dim: Input MSA feature dimension (default: 20 for PSSM)
        hidden_dim: Output hidden dimension (match ESM2)
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        msa_dim=20,
        hidden_dim=320,
        num_layers=2,
        num_heads=4,
        dropout=0.1
    ):
        super().__init__()
        self.msa_dim = msa_dim
        self.hidden_dim = hidden_dim

        # Project MSA features to hidden dim
        self.input_projection = nn.Linear(msa_dim, hidden_dim)

        # Transformer encoder for MSA
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, msa_features):
        """
        Encode MSA features.

        Args:
            msa_features: MSA features of shape (batch, seq_len, msa_dim)

        Returns:
            Encoded MSA features of shape (batch, seq_len, hidden_dim)
        """
        # Project to hidden dimension
        x = self.input_projection(msa_features)  # (batch, seq_len, hidden_dim)

        # Apply transformer
        x = self.transformer(x)  # (batch, seq_len, hidden_dim)

        # Layer norm
        x = self.layer_norm(x)

        return x


class CrossAttentionFusion(nn.Module):
    """
    Fuse sequence and MSA features using cross-attention.

    Args:
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(self, hidden_dim=320, num_heads=4, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Cross-attention: sequence attends to MSA
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, seq_features, msa_features):
        """
        Fuse sequence and MSA features.

        Args:
            seq_features: Sequence features (batch, seq_len, hidden_dim)
            msa_features: MSA features (batch, seq_len, hidden_dim)

        Returns:
            Fused features (batch, seq_len, hidden_dim)
        """
        # Cross-attention: sequence as query, MSA as key/value
        attn_output, _ = self.cross_attention(
            query=seq_features,
            key=msa_features,
            value=msa_features
        )

        # Residual connection + norm
        x = self.norm1(seq_features + attn_output)

        # Feed-forward + residual + norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x


class MultiModalMoEModel(nn.Module):
    """
    Multi-modal model combining sequence (ESM2) and MSA features with MoE.

    Stage 2 architecture:
    1. ESM2 sequence encoder
    2. MSA feature encoder
    3. Cross-attention fusion
    4. Mixture of Experts prediction

    Args:
        esm2_model_name: HuggingFace ESM2 model name
        num_experts: Number of expert networks
        expert_hidden_dim: Hidden dimension for experts
        msa_dim: MSA feature dimension
        freeze_esm2: Whether to freeze ESM2 weights
        dropout: Dropout rate
        use_msa: Whether to use MSA features (Stage 2) or not (Stage 1)
    """

    def __init__(
        self,
        esm2_model_name='facebook/esm2_t33_650M_UR50D',
        num_experts=8,
        expert_hidden_dim=512,
        msa_dim=20,
        freeze_esm2=False,
        dropout=0.15,
        use_msa=True
    ):
        super().__init__()

        self.use_msa = use_msa
        self.num_experts = num_experts
        self.expert_hidden_dim = expert_hidden_dim

        # Load ESM2 encoder
        self.esm2_encoder = AutoModel.from_pretrained(esm2_model_name)
        self.hidden_dim = self.esm2_encoder.config.hidden_size

        if freeze_esm2:
            for param in self.esm2_encoder.parameters():
                param.requires_grad = False

        # MSA encoder (only if using MSA)
        if use_msa:
            self.msa_encoder = MSAEncoder(
                msa_dim=msa_dim,
                hidden_dim=self.hidden_dim,
                num_layers=2,
                num_heads=4,
                dropout=dropout
            )

            # Cross-attention fusion
            self.fusion = CrossAttentionFusion(
                hidden_dim=self.hidden_dim,
                num_heads=8,
                dropout=dropout
            )

        # Pooling
        self.pooling = nn.AdaptiveAvgPool1d(1)

        # Mixture of Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, expert_hidden_dim),
                nn.LayerNorm(expert_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden_dim, expert_hidden_dim // 2),
                nn.LayerNorm(expert_hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden_dim // 2, 1)
            )
            for _ in range(num_experts)
        ])

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, input_ids, attention_mask=None, msa_features=None):
        """
        Forward pass.

        Args:
            input_ids: Tokenized sequences (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            msa_features: MSA features (batch, seq_len, msa_dim), optional

        Returns:
            Predicted fitness scores (batch,)
        """
        # Encode sequence with ESM2
        esm_output = self.esm2_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        seq_features = esm_output.last_hidden_state  # (batch, seq_len, hidden_dim)

        # If using MSA, fuse features
        if self.use_msa and msa_features is not None:
            # Encode MSA
            msa_encoded = self.msa_encoder(msa_features)  # (batch, seq_len, hidden_dim)

            # Fuse via cross-attention
            fused_features = self.fusion(seq_features, msa_encoded)
        else:
            fused_features = seq_features

        # Pool sequence features
        # (batch, seq_len, hidden_dim) -> (batch, hidden_dim)
        pooled = fused_features.mean(dim=1)

        # Compute gating weights
        gate_weights = self.gate(pooled)  # (batch, num_experts)

        # Compute expert predictions
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(pooled)  # (batch, 1)
            expert_outputs.append(expert_out)

        expert_outputs = torch.cat(expert_outputs, dim=1)  # (batch, num_experts)

        # Weighted combination
        predictions = (gate_weights * expert_outputs).sum(dim=1)  # (batch,)

        return predictions


# Convenience function
def create_multimodal_model(
    config,
    use_msa=True
):
    """
    Create multi-modal model from config.

    Args:
        config: Configuration dict
        use_msa: Whether to use MSA features

    Returns:
        MultiModalMoEModel instance
    """
    model_config = config.get('model', {})

    return MultiModalMoEModel(
        esm2_model_name=model_config.get('esm2_model', 'facebook/esm2_t33_650M_UR50D'),
        num_experts=model_config.get('num_experts', 8),
        expert_hidden_dim=model_config.get('expert_hidden_dim', 512),
        msa_dim=model_config.get('msa_dim', 20),
        freeze_esm2=model_config.get('freeze_esm2', False),
        dropout=model_config.get('dropout', 0.15),
        use_msa=use_msa
    )
