#!/usr/bin/env python3
"""
HPC Training Script for Protein Functionality Prediction with MoE

This script is designed for training on High Performance Computing (HPC) clusters
with SLURM job scheduler. It includes proper logging, checkpointing, and distributed
training capabilities.

Usage:
    python scripts/train_hpc.py --config configs/moe_config.yaml

SLURM Usage:
    sbatch scripts/submit_job.sh

Author: Generated for Protein Functionality Research
Date: 2024
"""

import os
import sys
import argparse
import logging
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import MoEModel
from data import ProteinDataModule, EmbeddingDataset
from training.losses import PairwiseRankingLoss, CombinedLoss
from training.utils import save_checkpoint, load_checkpoint, compute_ranking_metrics
from utils.logging import setup_logging
from utils.config import load_config, validate_config


def setup_distributed():
    """Initialize distributed training."""
    if 'SLURM_PROCID' in os.environ:
        # SLURM environment
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])

        # Set up process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )

        torch.cuda.set_device(local_rank)

    elif 'RANK' in os.environ:
        # Manual distributed setup
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )

        torch.cuda.set_device(local_rank)
    else:
        # Single GPU setup
        rank = 0
        world_size = 1
        local_rank = 0

    return rank, world_size, local_rank


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create and initialize the MoE model."""
    model_config = config['model']

    model = MoEModel(
        input_dim=model_config.get('input_dim', 320),
        hidden_dim=model_config.get('hidden_dim', 512),
        num_experts=model_config.get('num_experts', 4),
        k=model_config.get('top_k', 2),
        num_layers=model_config.get('num_layers', 2),
        output_dim=model_config.get('output_dim', 1),
        dropout=model_config.get('dropout', 0.1)
    )

    model = model.to(device)

    # Wrap in DDP if using distributed training
    if dist.is_initialized():
        model = DDP(model, device_ids=[device.index])

    return model


def create_loss_function(config: Dict[str, Any]) -> nn.Module:
    """Create the loss function."""
    loss_config = config['training']['loss']

    ranking_loss = PairwiseRankingLoss(
        margin=loss_config.get('margin', 0.1),
        n_pairs=loss_config.get('n_pairs', 1000)
    )

    losses = {
        'ranking': (ranking_loss, loss_config.get('ranking_weight', 1.0))
    }

    # Add MSE loss if specified
    if loss_config.get('use_mse', False):
        mse_loss = nn.MSELoss()
        losses['mse'] = (mse_loss, loss_config.get('mse_weight', 0.1))

    return CombinedLoss(losses, aux_weight=loss_config.get('aux_weight', 0.01))


def create_data_loaders(config: Dict[str, Any], rank: int, world_size: int) -> Dict[str, DataLoader]:
    """Create data loaders for training and validation."""
    data_config = config['data']

    # Create data module
    data_module = ProteinDataModule(
        data_dir=data_config['data_dir'],
        embedding_dir=data_config.get('embedding_dir'),
        batch_size=data_config['batch_size'],
        num_workers=data_config.get('num_workers', 4),
        support_size=data_config.get('support_size', 128),
        query_size=data_config.get('query_size', 72)
    )

    # Get embedding files
    train_embedding_dir = os.path.join(data_config['embedding_dir'], 'train')
    val_embedding_dir = os.path.join(data_config['embedding_dir'], 'test')

    train_files = [os.path.join(train_embedding_dir, f)
                   for f in os.listdir(train_embedding_dir) if f.endswith('.pt')]
    val_files = [os.path.join(val_embedding_dir, f)
                 for f in os.listdir(val_embedding_dir) if f.endswith('.pt')]

    # Create datasets
    train_dataset = EmbeddingDataset(train_files[:data_config.get('max_train_files')])
    val_dataset = EmbeddingDataset(val_files[:data_config.get('max_val_files')])

    # Create samplers for distributed training
    train_sampler = None
    val_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        sampler=val_sampler,
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'train_sampler': train_sampler,
        'val_sampler': val_sampler
    }


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
    writer: Optional[SummaryWriter] = None,
    rank: int = 0
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_aux_loss = 0.0
    num_batches = 0

    # Only show progress bar on rank 0
    iterator = tqdm(data_loader, desc=f"Epoch {epoch}") if rank == 0 else data_loader

    for batch_idx, batch in enumerate(iterator):
        # Move data to device
        embeddings = batch['embedding'].to(device)
        scores = batch['score'].to(device)

        # Forward pass
        predictions, aux_loss = model(embeddings)
        predictions = predictions.squeeze(-1)  # [batch, 1] -> [batch]

        # Compute loss
        loss, loss_dict = loss_fn(predictions, scores, aux_loss)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if config['training'].get('grad_clip_norm'):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['grad_clip_norm']
            )

        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        total_aux_loss += aux_loss.item() if aux_loss is not None else 0.0
        num_batches += 1

        # Log to tensorboard
        if writer and rank == 0 and batch_idx % config['logging']['log_interval'] == 0:
            global_step = epoch * len(data_loader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/AuxLoss', aux_loss.item() if aux_loss else 0.0, global_step)

        # Memory cleanup
        del embeddings, scores, predictions, loss
        if aux_loss is not None:
            del aux_loss
        torch.cuda.empty_cache()

    return {
        'loss': total_loss / num_batches,
        'aux_loss': total_aux_loss / num_batches
    }


def validate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    rank: int = 0
) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_targets = []
    num_batches = 0

    with torch.no_grad():
        iterator = tqdm(data_loader, desc=f"Validation") if rank == 0 else data_loader

        for batch in iterator:
            embeddings = batch['embedding'].to(device)
            scores = batch['score'].to(device)

            # Forward pass
            predictions, aux_loss = model(embeddings)
            predictions = predictions.squeeze(-1)

            # Compute loss
            loss, _ = loss_fn(predictions, scores, aux_loss)

            total_loss += loss.item()
            num_batches += 1

            # Collect predictions for metrics
            all_predictions.append(predictions.cpu())
            all_targets.append(scores.cpu())

            # Memory cleanup
            del embeddings, scores, predictions, loss
            if aux_loss is not None:
                del aux_loss
            torch.cuda.empty_cache()

    # Compute metrics
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    metrics = compute_ranking_metrics(all_predictions, all_targets)
    metrics['loss'] = total_loss / num_batches

    # Log metrics
    if writer and rank == 0:
        for metric_name, metric_value in metrics.items():
            writer.add_scalar(f'Val/{metric_name}', metric_value, epoch)

    return metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train MoE model for protein functionality prediction')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory for logs and checkpoints')

    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    # Load configuration
    config = load_config(args.config)
    validate_config(config)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging (only on rank 0)
    if rank == 0:
        setup_logging(output_dir / 'train.log')
        logging.info(f"Starting training with config: {args.config}")
        logging.info(f"World size: {world_size}, Rank: {rank}")
        logging.info(f"Output directory: {output_dir}")

    # Set device
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    # Set random seeds for reproducibility
    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['training']['seed'])

    # Create model
    model = create_model(config, device)

    # Create loss function
    loss_fn = create_loss_function(config)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.01)
    )

    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training'].get('min_lr', 1e-6)
    )

    # Create data loaders
    data_loaders = create_data_loaders(config, rank, world_size)

    # Setup tensorboard writer (only on rank 0)
    writer = None
    if rank == 0:
        writer = SummaryWriter(output_dir / 'tensorboard')

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if rank == 0:
            logging.info(f"Resumed from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, config['training']['epochs']):
        # Set epoch for distributed sampler
        if data_loaders['train_sampler'] is not None:
            data_loaders['train_sampler'].set_epoch(epoch)

        # Train epoch
        train_metrics = train_epoch(
            model, data_loaders['train'], loss_fn, optimizer, device,
            epoch, config, writer, rank
        )

        # Validate epoch
        val_metrics = validate_epoch(
            model, data_loaders['val'], loss_fn, device,
            epoch, writer, rank
        )

        # Update learning rate
        scheduler.step()

        # Log metrics (only on rank 0)
        if rank == 0:
            logging.info(f"Epoch {epoch}")
            logging.info(f"Train Loss: {train_metrics['loss']:.6f}")
            logging.info(f"Val Loss: {val_metrics['loss']:.6f}")
            logging.info(f"Val Spearman: {val_metrics.get('spearman', 0.0):.4f}")
            logging.info(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint (only on rank 0)
        if rank == 0:
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']

            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config,
                'val_metrics': val_metrics
            }, is_best, output_dir)

            # Save regular checkpoint every N epochs
            if epoch % config['training'].get('save_interval', 10) == 0:
                checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'config': config
                }, checkpoint_path)

    # Cleanup
    if writer:
        writer.close()

    if dist.is_initialized():
        dist.destroy_process_group()

    if rank == 0:
        logging.info("Training completed!")


if __name__ == '__main__':
    main()