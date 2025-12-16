"""
Metrics utilities for protein functionality prediction.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Union
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error


def compute_ranking_metrics(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray]
) -> Dict[str, float]:
    """
    Compute ranking metrics for protein functionality prediction.

    Args:
        predictions (Union[torch.Tensor, np.ndarray]): Model predictions
        targets (Union[torch.Tensor, np.ndarray]): Ground truth targets

    Returns:
        Dict[str, float]: Dictionary of computed metrics
    """
    # Convert to numpy arrays
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()

    # Ensure 1D arrays
    predictions = predictions.flatten()
    targets = targets.flatten()

    metrics = {}

    # Spearman correlation (primary metric for ranking)
    spearman_corr, spearman_p = spearmanr(predictions, targets)
    metrics['spearman'] = spearman_corr if not np.isnan(spearman_corr) else 0.0
    metrics['spearman_pvalue'] = spearman_p if not np.isnan(spearman_p) else 1.0

    # Pearson correlation
    pearson_corr, pearson_p = pearsonr(predictions, targets)
    metrics['pearson'] = pearson_corr if not np.isnan(pearson_corr) else 0.0
    metrics['pearson_pvalue'] = pearson_p if not np.isnan(pearson_p) else 1.0

    # Regression metrics
    metrics['mse'] = mean_squared_error(targets, predictions)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(targets, predictions)

    # R-squared
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    # Ranking-specific metrics
    metrics['kendall_tau'] = compute_kendall_tau(predictions, targets)
    metrics['ndcg_at_10'] = compute_ndcg(predictions, targets, k=10)
    metrics['top_k_accuracy_10'] = compute_top_k_overlap(predictions, targets, k=10)

    return metrics


def compute_spearman_correlation(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Compute Spearman correlation coefficient.

    Args:
        predictions (Union[torch.Tensor, np.ndarray]): Model predictions
        targets (Union[torch.Tensor, np.ndarray]): Ground truth targets

    Returns:
        float: Spearman correlation coefficient
    """
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()

    corr, _ = spearmanr(predictions.flatten(), targets.flatten())
    return corr if not np.isnan(corr) else 0.0


def compute_kendall_tau(
    predictions: np.ndarray,
    targets: np.ndarray
) -> float:
    """
    Compute Kendall's tau correlation coefficient.

    Args:
        predictions (np.ndarray): Model predictions
        targets (np.ndarray): Ground truth targets

    Returns:
        float: Kendall's tau coefficient
    """
    from scipy.stats import kendalltau

    tau, _ = kendalltau(predictions, targets)
    return tau if not np.isnan(tau) else 0.0


def compute_ndcg(
    predictions: np.ndarray,
    targets: np.ndarray,
    k: int = 10
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) at k.

    Args:
        predictions (np.ndarray): Model predictions
        targets (np.ndarray): Ground truth targets
        k (int): Number of top items to consider

    Returns:
        float: NDCG@k score
    """
    if len(predictions) < k:
        k = len(predictions)

    # Sort by predictions (descending)
    pred_indices = np.argsort(predictions)[::-1][:k]
    pred_relevance = targets[pred_indices]

    # Sort by targets (descending) for ideal ranking
    ideal_indices = np.argsort(targets)[::-1][:k]
    ideal_relevance = targets[ideal_indices]

    # Compute DCG
    dcg = np.sum(pred_relevance / np.log2(np.arange(2, k + 2)))

    # Compute IDCG
    idcg = np.sum(ideal_relevance / np.log2(np.arange(2, k + 2)))

    # Compute NDCG
    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_top_k_overlap(
    predictions: np.ndarray,
    targets: np.ndarray,
    k: int = 10
) -> float:
    """
    Compute overlap between top-k predictions and top-k targets.

    Args:
        predictions (np.ndarray): Model predictions
        targets (np.ndarray): Ground truth targets
        k (int): Number of top items to consider

    Returns:
        float: Overlap ratio (0.0 to 1.0)
    """
    if len(predictions) < k:
        k = len(predictions)

    # Get top-k indices
    pred_top_k = set(np.argsort(predictions)[::-1][:k])
    target_top_k = set(np.argsort(targets)[::-1][:k])

    # Compute overlap
    overlap = len(pred_top_k.intersection(target_top_k))
    return overlap / k


def compute_ranking_loss_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """
    Compute metrics specifically for ranking loss evaluation.

    Args:
        predictions (torch.Tensor): Model predictions
        targets (torch.Tensor): Ground truth targets

    Returns:
        Dict[str, float]: Ranking loss metrics
    """
    predictions_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    metrics = {}

    # Pairwise ranking accuracy
    metrics['pairwise_accuracy'] = compute_pairwise_accuracy(predictions_np, targets_np)

    # Concordance index (C-index)
    metrics['concordance_index'] = compute_concordance_index(predictions_np, targets_np)

    # Ranking precision at different k values
    for k in [5, 10, 20]:
        if len(predictions) >= k:
            metrics[f'precision_at_{k}'] = compute_precision_at_k(predictions_np, targets_np, k)

    return metrics


def compute_pairwise_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray
) -> float:
    """
    Compute pairwise ranking accuracy.

    Args:
        predictions (np.ndarray): Model predictions
        targets (np.ndarray): Ground truth targets

    Returns:
        float: Pairwise accuracy
    """
    n = len(predictions)
    if n < 2:
        return 0.0

    correct = 0
    total = 0

    for i in range(n):
        for j in range(i + 1, n):
            if targets[i] != targets[j]:  # Only consider pairs with different targets
                pred_order = predictions[i] > predictions[j]
                true_order = targets[i] > targets[j]
                if pred_order == true_order:
                    correct += 1
                total += 1

    return correct / total if total > 0 else 0.0


def compute_concordance_index(
    predictions: np.ndarray,
    targets: np.ndarray
) -> float:
    """
    Compute concordance index (C-index).

    Args:
        predictions (np.ndarray): Model predictions
        targets (np.ndarray): Ground truth targets

    Returns:
        float: Concordance index
    """
    return compute_pairwise_accuracy(predictions, targets)  # Same as pairwise accuracy


def compute_precision_at_k(
    predictions: np.ndarray,
    targets: np.ndarray,
    k: int
) -> float:
    """
    Compute precision at k for ranking evaluation.

    Args:
        predictions (np.ndarray): Model predictions
        targets (np.ndarray): Ground truth targets
        k (int): Number of top items to consider

    Returns:
        float: Precision at k
    """
    if len(predictions) < k:
        k = len(predictions)

    # Get top-k predictions
    top_k_indices = np.argsort(predictions)[::-1][:k]

    # Determine relevance threshold (top 20% of targets)
    relevance_threshold = np.percentile(targets, 80)
    relevant_items = targets >= relevance_threshold

    # Count relevant items in top-k predictions
    relevant_in_top_k = np.sum(relevant_items[top_k_indices])

    return relevant_in_top_k / k


def aggregate_metrics_across_tasks(
    task_metrics: list
) -> Dict[str, float]:
    """
    Aggregate metrics across multiple tasks.

    Args:
        task_metrics (list): List of metric dictionaries for each task

    Returns:
        Dict[str, float]: Aggregated metrics
    """
    if not task_metrics:
        return {}

    # Get all metric names
    all_metrics = set()
    for metrics in task_metrics:
        all_metrics.update(metrics.keys())

    aggregated = {}

    for metric in all_metrics:
        values = []
        for task_metric in task_metrics:
            if metric in task_metric and not np.isnan(task_metric[metric]):
                values.append(task_metric[metric])

        if values:
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_median'] = np.median(values)
        else:
            aggregated[f'{metric}_mean'] = 0.0
            aggregated[f'{metric}_std'] = 0.0
            aggregated[f'{metric}_median'] = 0.0

    return aggregated