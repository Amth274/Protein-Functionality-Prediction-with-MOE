"""
Expert network implementations for Mixture-of-Experts architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """
    Individual expert network for MoE architecture.

    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden layer dimension
        output_dim (int): Output dimension (defaults to input_dim for residual connections)
        dropout (float): Dropout probability
    """

    def __init__(self, input_dim, hidden_dim, output_dim=None, dropout=0.1):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Forward pass through expert network.

        Args:
            x (torch.Tensor): Input tensor [batch, input_dim]

        Returns:
            torch.Tensor: Expert output [batch, output_dim]
        """
        return self.fc(x)


class ProteinExpert(nn.Module):
    """
    Specialized expert for protein-specific transformations.

    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden layer dimension
        expert_type (str): Type of protein expert ('viral', 'human', 'structural', etc.)
        dropout (float): Dropout probability
    """

    def __init__(self, input_dim, hidden_dim, expert_type='general', dropout=0.1):
        super().__init__()
        self.expert_type = expert_type

        # Type-specific architecture choices
        if expert_type == 'viral':
            # Smaller, more focused networks for viral proteins
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, input_dim)
            )
        elif expert_type == 'structural':
            # Deeper networks for structural features
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, input_dim)
            )
        else:  # general
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, input_dim)
            )

    def forward(self, x):
        """
        Forward pass through protein expert.

        Args:
            x (torch.Tensor): Input protein embeddings [batch, input_dim]

        Returns:
            torch.Tensor: Expert-transformed embeddings [batch, input_dim]
        """
        return self.network(x)