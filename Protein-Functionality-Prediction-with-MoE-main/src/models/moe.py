"""
Mixture-of-Experts implementation for protein functionality prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .experts import Expert, ProteinExpert


class MoELayer(nn.Module):
    """
    Mixture-of-Experts layer with gating mechanism.

    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension for expert networks
        num_experts (int): Number of expert networks
        k (int): Number of top experts to use per input
        expert_type (str): Type of expert networks ('standard' or 'protein')
        gate_type (str): Gating mechanism type ('linear' or 'attention')
        load_balancing (bool): Whether to use load balancing loss
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_experts=4,
        k=2,
        expert_type='standard',
        gate_type='linear',
        load_balancing=True
    ):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.load_balancing = load_balancing

        # Gating network
        if gate_type == 'linear':
            self.gate = nn.Linear(input_dim, num_experts)
        elif gate_type == 'attention':
            self.gate = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_experts)
            )
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}")

        # Expert networks
        if expert_type == 'standard':
            self.experts = nn.ModuleList([
                Expert(input_dim, hidden_dim)
                for _ in range(num_experts)
            ])
        elif expert_type == 'protein':
            expert_types = ['viral', 'structural', 'general', 'general']  # customize as needed
            self.experts = nn.ModuleList([
                ProteinExpert(input_dim, hidden_dim, expert_types[i % len(expert_types)])
                for i in range(num_experts)
            ])
        else:
            raise ValueError(f"Unknown expert_type: {expert_type}")

    def forward(self, x):
        """
        Forward pass through MoE layer.

        Args:
            x (torch.Tensor): Input tensor [batch, input_dim]

        Returns:
            tuple: (output, auxiliary_loss) where output is [batch, input_dim]
                   and auxiliary_loss is load balancing loss
        """
        batch_size = x.size(0)

        # Compute gating weights
        gate_logits = self.gate(x)  # [batch, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Select top-k experts
        topk_vals, topk_idx = torch.topk(gate_probs, self.k, dim=-1)
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)  # renormalize

        # Compute expert outputs
        expert_outputs = []
        for i in range(self.num_experts):
            expert_outputs.append(self.experts[i](x))
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, input_dim]

        # Combine outputs using top-k gating
        output = torch.zeros_like(x)
        for i in range(self.k):
            expert_idx = topk_idx[:, i]  # [batch]
            weight = topk_vals[:, i].unsqueeze(-1)  # [batch, 1]

            # Gather expert outputs for selected experts
            selected_outputs = expert_outputs[torch.arange(batch_size), expert_idx]
            output += weight * selected_outputs

        # Compute load balancing loss
        auxiliary_loss = 0.0
        if self.load_balancing and self.training:
            # Encourage uniform expert usage
            expert_usage = gate_probs.mean(dim=0)  # [num_experts]
            balance_loss = (expert_usage * torch.log(expert_usage + 1e-8)).sum()
            auxiliary_loss = 0.01 * balance_loss

        return output, auxiliary_loss


class MoEModel(nn.Module):
    """
    Complete MoE model for protein functionality prediction.

    Args:
        input_dim (int): Input embedding dimension
        hidden_dim (int): Hidden dimension for MoE layers
        num_experts (int): Number of expert networks
        k (int): Number of top experts to use
        num_layers (int): Number of MoE layers
        output_dim (int): Final output dimension (1 for regression)
        dropout (float): Dropout probability
    """

    def __init__(
        self,
        input_dim=320,
        hidden_dim=512,
        num_experts=4,
        k=2,
        num_layers=2,
        output_dim=1,
        dropout=0.1
    ):
        super().__init__()
        self.num_layers = num_layers

        # MoE layers
        self.moe_layers = nn.ModuleList([
            MoELayer(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                k=k,
                expert_type='protein'
            )
            for _ in range(num_layers)
        ])

        # Final regression head
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        """
        Forward pass through MoE model.

        Args:
            x (torch.Tensor): Input embeddings [batch, input_dim]

        Returns:
            tuple: (predictions, auxiliary_loss) where predictions is [batch, output_dim]
        """
        total_aux_loss = 0.0

        # Pass through MoE layers
        for moe_layer in self.moe_layers:
            x, aux_loss = moe_layer(x)
            total_aux_loss += aux_loss

        # Final prediction
        predictions = self.regressor(x)

        return predictions, total_aux_loss