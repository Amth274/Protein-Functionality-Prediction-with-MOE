"""
Loss functions for protein functionality prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class RankingLoss(nn.Module):
    """
    Ranking loss for protein fitness prediction.

    Args:
        margin (float): Margin for ranking loss
        reduction (str): Reduction strategy ('mean', 'sum', 'none')
    """

    def __init__(self, margin: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute ranking loss.

        Args:
            predictions (torch.Tensor): Model predictions [batch]
            targets (torch.Tensor): Target scores [batch]

        Returns:
            torch.Tensor: Ranking loss
        """
        batch_size = predictions.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # Create all pairwise comparisons
        pred_diff = predictions.unsqueeze(1) - predictions.unsqueeze(0)  # [batch, batch]
        target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)  # [batch, batch]

        # Mask for valid comparisons (different targets)
        mask = (target_diff != 0).float()

        # Sign of target differences
        target_sign = torch.sign(target_diff)

        # Ranking loss: max(0, margin - target_sign * pred_diff)
        loss = F.relu(self.margin - target_sign * pred_diff) * mask

        # Apply reduction
        if self.reduction == 'mean':
            return loss.sum() / (mask.sum() + 1e-8)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class PairwiseRankingLoss(nn.Module):
    """
    Memory-efficient pairwise ranking loss with sampling.

    Args:
        margin (float): Margin for ranking loss
        n_pairs (int): Number of pairs to sample
        reduction (str): Reduction strategy
    """

    def __init__(self, margin: float = 0.1, n_pairs: int = 1000, reduction: str = 'mean'):
        super().__init__()
        self.margin = margin
        self.n_pairs = n_pairs
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise ranking loss with sampling.

        Args:
            predictions (torch.Tensor): Model predictions [batch]
            targets (torch.Tensor): Target scores [batch]

        Returns:
            torch.Tensor: Ranking loss
        """
        batch_size = predictions.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # Sample pairs
        n_pairs = min(self.n_pairs, batch_size * (batch_size - 1) // 2)
        idx_i = torch.randint(0, batch_size, (n_pairs,), device=predictions.device)
        idx_j = torch.randint(0, batch_size, (n_pairs,), device=predictions.device)

        # Ensure different indices
        mask = (idx_i != idx_j) & (targets[idx_i] != targets[idx_j])
        if mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        idx_i = idx_i[mask]
        idx_j = idx_j[mask]

        # Compute differences
        pred_diff = predictions[idx_i] - predictions[idx_j]
        target_diff = targets[idx_i] - targets[idx_j]
        target_sign = torch.sign(target_diff)

        # Ranking loss
        loss = F.relu(self.margin - target_sign * pred_diff)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MarginRankingLoss(nn.Module):
    """
    PyTorch-style margin ranking loss.

    Args:
        margin (float): Margin for ranking loss
        reduction (str): Reduction strategy
    """

    def __init__(self, margin: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.margin_loss = nn.MarginRankingLoss(margin=margin, reduction=reduction)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute margin ranking loss using PyTorch implementation.

        Args:
            predictions (torch.Tensor): Model predictions [batch]
            targets (torch.Tensor): Target scores [batch]

        Returns:
            torch.Tensor: Ranking loss
        """
        batch_size = predictions.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # Sample pairs for memory efficiency
        n_pairs = min(1000, batch_size * (batch_size - 1) // 2)
        idx_i = torch.randint(0, batch_size, (n_pairs,), device=predictions.device)
        idx_j = torch.randint(0, batch_size, (n_pairs,), device=predictions.device)

        # Filter valid pairs
        mask = (idx_i != idx_j) & (targets[idx_i] != targets[idx_j])
        if mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        idx_i = idx_i[mask]
        idx_j = idx_j[mask]

        # Create ranking targets
        y = torch.sign(targets[idx_i] - targets[idx_j])

        return self.margin_loss(predictions[idx_i], predictions[idx_j], y)


class CombinedLoss(nn.Module):
    """
    Combines multiple loss functions.

    Args:
        losses (dict): Dictionary of loss functions and their weights
        aux_weight (float): Weight for auxiliary losses
    """

    def __init__(self, losses: dict, aux_weight: float = 0.01):
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.aux_weight = aux_weight

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        aux_loss: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Target scores
            aux_loss (torch.Tensor, optional): Auxiliary loss (e.g., from MoE)

        Returns:
            Tuple[torch.Tensor, dict]: Total loss and individual loss components
        """
        total_loss = 0.0
        loss_dict = {}

        # Compute main losses
        for name, (loss_fn, weight) in self.losses.items():
            loss_value = loss_fn(predictions, targets)
            total_loss += weight * loss_value
            loss_dict[name] = loss_value.item()

        # Add auxiliary loss
        if aux_loss is not None:
            aux_loss_value = self.aux_weight * aux_loss
            total_loss += aux_loss_value
            loss_dict['aux_loss'] = aux_loss.item()

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict