"""
Ranking loss for direct Spearman correlation optimization.

Implements differentiable ranking loss that encourages correct ordering
of predictions, optimizing for Spearman correlation metric.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RankingLoss(nn.Module):
    """
    Pairwise ranking loss optimized for Spearman correlation.

    Uses differentiable soft ranking to approximate Spearman correlation
    and minimize negative correlation (= maximize correlation).

    Args:
        temperature: Temperature for soft ranking (lower = sharper rankings)
        margin: Margin for pairwise ranking loss
        loss_type: 'spearman' or 'pairwise'
    """

    def __init__(self, temperature=0.1, margin=0.1, loss_type='spearman'):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.loss_type = loss_type

    def soft_rank(self, x):
        """
        Compute differentiable soft ranks using sigmoid approximation.

        Args:
            x: Tensor of shape (N,) containing values to rank

        Returns:
            Soft ranks of shape (N,)
        """
        n = x.shape[0]
        # Pairwise comparisons: x[i] - x[j]
        pairwise = x.unsqueeze(1) - x.unsqueeze(0)
        # Soft counting: how many values are less than x[i]
        ranks = torch.sigmoid(pairwise / self.temperature).sum(dim=1)
        return ranks

    def spearman_loss(self, predictions, targets):
        """
        Compute negative Spearman correlation as loss.

        Uses soft ranks to approximate Spearman correlation and returns
        negative value for minimization.

        Args:
            predictions: Predicted values (N,)
            targets: True values (N,)

        Returns:
            Negative Spearman correlation
        """
        # Get soft ranks
        pred_ranks = self.soft_rank(predictions)
        target_ranks = self.soft_rank(targets)

        # Center the ranks
        pred_centered = pred_ranks - pred_ranks.mean()
        target_centered = target_ranks - target_ranks.mean()

        # Compute Pearson correlation on ranks (= Spearman on values)
        numerator = (pred_centered * target_centered).sum()
        denominator = torch.sqrt(
            (pred_centered ** 2).sum() * (target_centered ** 2).sum() + 1e-8
        )

        correlation = numerator / denominator

        # Return negative for minimization
        return -correlation

    def pairwise_ranking_loss(self, predictions, targets):
        """
        Compute pairwise ranking loss.

        For all pairs where target[i] > target[j], encourage pred[i] > pred[j].

        Args:
            predictions: Predicted values (N,)
            targets: True values (N,)

        Returns:
            Average pairwise hinge loss
        """
        n = predictions.shape[0]

        # Create all pairwise differences
        pred_diff = predictions.unsqueeze(1) - predictions.unsqueeze(0)  # (N, N)
        target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)  # (N, N)

        # Only consider pairs where targets are different
        # Sign: +1 if target[i] > target[j], -1 if target[i] < target[j]
        sign = torch.sign(target_diff)

        # Hinge loss: max(0, margin - sign * pred_diff)
        loss = F.relu(self.margin - sign * pred_diff)

        # Mask diagonal (self-comparisons)
        mask = 1 - torch.eye(n, device=predictions.device)
        loss = loss * mask

        # Average over all pairs
        return loss.sum() / (n * (n - 1) + 1e-8)

    def forward(self, predictions, targets):
        """
        Compute ranking loss.

        Args:
            predictions: Predicted fitness scores (N,)
            targets: True fitness scores (N,)

        Returns:
            Ranking loss value
        """
        # Need at least 2 samples for ranking
        if len(predictions) < 2:
            return torch.tensor(0.0, device=predictions.device)

        # Flatten if needed
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Compute selected loss type
        if self.loss_type == 'spearman':
            return self.spearman_loss(predictions, targets)
        elif self.loss_type == 'pairwise':
            return self.pairwise_ranking_loss(predictions, targets)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")


class CombinedLoss(nn.Module):
    """
    Combine MSE loss with ranking loss.

    Args:
        mse_weight: Weight for MSE loss
        ranking_weight: Weight for ranking loss
        ranking_temperature: Temperature for soft ranking
        ranking_type: Type of ranking loss ('spearman' or 'pairwise')
    """

    def __init__(
        self,
        mse_weight=0.5,
        ranking_weight=0.5,
        ranking_temperature=0.1,
        ranking_type='spearman'
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.ranking_weight = ranking_weight

        self.mse_loss = nn.MSELoss()
        self.ranking_loss = RankingLoss(
            temperature=ranking_temperature,
            loss_type=ranking_type
        )

    def forward(self, predictions, targets):
        """
        Compute combined loss.

        Args:
            predictions: Predicted values (N,)
            targets: True values (N,)

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Compute individual losses
        mse = self.mse_loss(predictions, targets)
        ranking = self.ranking_loss(predictions, targets)

        # Combine
        total_loss = self.mse_weight * mse + self.ranking_weight * ranking

        loss_dict = {
            'mse': mse.item(),
            'ranking': ranking.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict


# Convenience function
def compute_combined_loss(
    predictions,
    targets,
    mse_weight=0.5,
    ranking_weight=0.5,
    ranking_temperature=0.1
):
    """
    Compute combined MSE + ranking loss.

    Args:
        predictions: Predicted values
        targets: True values
        mse_weight: Weight for MSE (default: 0.5)
        ranking_weight: Weight for ranking (default: 0.5)
        ranking_temperature: Temperature for soft ranking

    Returns:
        total_loss: Combined loss value
        loss_dict: Dictionary with individual components
    """
    criterion = CombinedLoss(
        mse_weight=mse_weight,
        ranking_weight=ranking_weight,
        ranking_temperature=ranking_temperature
    )
    return criterion(predictions, targets)
