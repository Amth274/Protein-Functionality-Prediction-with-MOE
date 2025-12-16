"""
Training utilities for protein functionality prediction.
"""

import os
import torch
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np

from ..utils.metrics import compute_ranking_metrics


def save_checkpoint(
    checkpoint: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: Union[str, Path],
    filename: str = 'checkpoint.pth'
):
    """
    Save model checkpoint.

    Args:
        checkpoint (Dict[str, Any]): Checkpoint dictionary
        is_best (bool): Whether this is the best checkpoint
        checkpoint_dir (Union[str, Path]): Directory to save checkpoint
        filename (str): Checkpoint filename
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / filename
    torch.save(checkpoint, checkpoint_path)

    if is_best:
        best_path = checkpoint_dir / 'best_model.pth'
        shutil.copy2(checkpoint_path, best_path)


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        checkpoint_path (Union[str, Path]): Path to checkpoint file
        model (torch.nn.Module): Model to load state into
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler to load state into
        device (torch.device, optional): Device to load checkpoint on

    Returns:
        Dict[str, Any]: Checkpoint dictionary
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    if hasattr(model, 'module'):
        # Handle DataParallel/DistributedDataParallel
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


class EarlyStopping:
    """
    Early stopping utility for training.

    Args:
        patience (int): Number of epochs to wait before stopping
        min_delta (float): Minimum change to qualify as improvement
        mode (str): 'min' for decreasing metrics, 'max' for increasing metrics
        restore_best_weights (bool): Whether to restore best weights when stopping
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_score = None
        self.counter = 0
        self.best_weights = None

        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater

    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """
        Check if training should stop.

        Args:
            score (float): Current metric score
            model (torch.nn.Module): Model to save best weights

        Returns:
            bool: Whether to stop training
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self.monitor_op(score, self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True

        return False

    def save_checkpoint(self, model: torch.nn.Module):
        """Save the best model weights."""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()


class MetricTracker:
    """
    Track and store training metrics.

    Args:
        metrics (list): List of metric names to track
    """

    def __init__(self, metrics: list):
        self.metrics = metrics
        self.reset()

    def reset(self):
        """Reset all tracked metrics."""
        self.values = {metric: [] for metric in self.metrics}
        self.averages = {metric: 0.0 for metric in self.metrics}

    def update(self, **kwargs):
        """Update metrics with new values."""
        for metric, value in kwargs.items():
            if metric in self.values:
                self.values[metric].append(value)

    def compute_averages(self):
        """Compute average values for all metrics."""
        for metric in self.metrics:
            if self.values[metric]:
                self.averages[metric] = np.mean(self.values[metric])

    def get_summary(self) -> Dict[str, float]:
        """Get summary of tracked metrics."""
        self.compute_averages()
        return self.averages.copy()

    def get_last_values(self) -> Dict[str, float]:
        """Get last values for all metrics."""
        return {metric: values[-1] if values else 0.0
                for metric, values in self.values.items()}


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.

    Args:
        model (torch.nn.Module): Model to count parameters for

    Returns:
        Dict[str, int]: Parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def get_model_size(model: torch.nn.Module) -> float:
    """
    Get model size in MB.

    Args:
        model (torch.nn.Module): Model to get size for

    Returns:
        float: Model size in MB
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def set_random_seeds(seed: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): Random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device_info() -> Dict[str, Any]:
    """
    Get device information.

    Returns:
        Dict[str, Any]: Device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
        'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None
    }

    if torch.cuda.is_available():
        info['memory_allocated'] = torch.cuda.memory_allocated()
        info['memory_reserved'] = torch.cuda.memory_reserved()

    return info


def clear_cuda_cache():
    """Clear CUDA cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class GradientClipper:
    """
    Gradient clipping utility.

    Args:
        max_norm (float): Maximum norm for gradient clipping
        norm_type (float): Type of norm to use
    """

    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type

    def __call__(self, model: torch.nn.Module) -> float:
        """
        Clip gradients and return the norm.

        Args:
            model (torch.nn.Module): Model with gradients to clip

        Returns:
            float: Gradient norm before clipping
        """
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.max_norm,
            norm_type=self.norm_type
        )
        return float(total_norm)


def warmup_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_lr: float
):
    """
    Create a learning rate scheduler with warmup.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        warmup_steps (int): Number of warmup steps
        max_lr (float): Maximum learning rate

    Returns:
        torch.optim.lr_scheduler: Learning rate scheduler
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_ranking_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """
    Compute ranking metrics (wrapper for utils.metrics function).

    Args:
        predictions (torch.Tensor): Model predictions
        targets (torch.Tensor): Target values

    Returns:
        Dict[str, float]: Ranking metrics
    """
    from ..utils.metrics import compute_ranking_metrics as _compute_ranking_metrics
    return _compute_ranking_metrics(predictions, targets)