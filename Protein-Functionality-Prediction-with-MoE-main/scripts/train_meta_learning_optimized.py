#!/usr/bin/env python3
"""
Optimized Meta-Learning Script for Protein Functionality Prediction

Production-ready implementation with episodic meta-learning featuring:
- Proper error handling and logging
- Checkpointing and resumption
- Mixed precision training
- Configuration management
- Better memory efficiency
- Reproducibility guarantees

Usage:
    python scripts/train_meta_learning_optimized.py --config configs/meta_learning_config.yaml
"""

import os
import sys
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import scipy.stats as stats

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.logging import setup_logging


# ---------------- Configuration ----------------
def load_meta_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------- Dataset ----------------
class ProteinGymDataset(Dataset):
    """
    Dataset for ProteinGym DMS data.

    Args:
        path: Path to CSV file
        seq_col: Column name for sequences
        fitness_col: Column name for fitness scores
        max_len: Maximum sequence length (filter longer sequences)
    """

    def __init__(
        self,
        path: str,
        seq_col: str = 'mutated_sequence',
        fitness_col: str = 'DMS_score',
        max_len: Optional[int] = None
    ):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            logging.error(f"Failed to load {path}: {e}")
            raise

        if seq_col not in df.columns or fitness_col not in df.columns:
            raise ValueError(f"Required columns not found in {path}")

        self.sequences = df[seq_col].astype(str).tolist()
        self.fitness = df[fitness_col].astype(float).tolist()

        # Filter by max length if specified
        if max_len:
            filtered = [
                (s, f) for s, f in zip(self.sequences, self.fitness)
                if len(s) <= max_len
            ]
            if filtered:
                self.sequences, self.fitness = zip(*filtered)
                self.sequences = list(self.sequences)
                self.fitness = list(self.fitness)
            else:
                self.sequences, self.fitness = [], []
                logging.warning(f"All sequences in {path} exceed max_len={max_len}")

        assert len(self.sequences) == len(self.fitness), "Sequence/fitness length mismatch"

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        return {
            "sequence": self.sequences[idx],
            "fitness": float(self.fitness[idx])
        }


# ---------------- Model Components ----------------
class ExpertMLP(nn.Module):
    """Single expert network for MoE."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MoEHead(nn.Module):
    """Mixture of Experts head with gating mechanism."""

    def __init__(
        self,
        in_dim: int,
        num_experts: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            ExpertMLP(in_dim, hidden_dim, dropout)
            for _ in range(num_experts)
        ])
        self.gate = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_experts)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE.

        Returns:
            predictions: Weighted expert outputs
            gate_weights: Expert selection weights (for analysis)
        """
        gate_logits = self.gate(x)
        gate_weights = torch.softmax(gate_logits, dim=-1)

        # Compute all expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        # Weighted combination
        predictions = (gate_weights * expert_outputs).sum(dim=1)

        return predictions, gate_weights


class ESM2MoEModel(nn.Module):
    """ESM2 backbone with MoE head for protein fitness prediction."""

    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        num_experts: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.embed_dim = self.backbone.config.hidden_size
        self.moe = MoEHead(self.embed_dim, num_experts, hidden_dim, dropout)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logging.info("ESM2 backbone frozen")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            predictions: Fitness predictions
            gate_weights: Expert selection weights
        """
        # Get ESM2 embeddings
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state  # [batch, seq_len, embed_dim]

        # Mean pooling (weighted by attention mask)
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
        sum_embeddings = (embeddings * mask_expanded).sum(1)
        sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
        pooled = sum_embeddings / sum_mask

        # MoE prediction
        predictions, gate_weights = self.moe(pooled)

        return predictions, gate_weights


# ---------------- Data Utilities ----------------
class ProteinCollator:
    """Collate function for protein sequences."""

    def __init__(self, tokenizer, max_len: int = 1024):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequences = [item["sequence"] for item in batch]
        fitness = torch.tensor([item["fitness"] for item in batch], dtype=torch.float)

        # Tokenize
        encoding = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len
        )

        return encoding["input_ids"], encoding["attention_mask"], fitness


# ---------------- Training Functions ----------------
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict,
    scaler: Optional[GradScaler] = None
) -> float:
    """Train for one epoch on support set."""
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    num_batches = 0

    for input_ids, attention_mask, fitness in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        fitness = fitness.to(device)

        optimizer.zero_grad()

        # Forward pass with mixed precision
        if scaler is not None:
            with autocast():
                predictions, _ = model(input_ids, attention_mask)
                loss = criterion(predictions, fitness)

            scaler.scale(loss).backward()

            # Gradient clipping
            if config['training'].get('grad_clip_norm'):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['training']['grad_clip_norm']
                )

            scaler.step(optimizer)
            scaler.update()
        else:
            predictions, _ = model(input_ids, attention_mask)
            loss = criterion(predictions, fitness)
            loss.backward()

            # Gradient clipping
            if config['training'].get('grad_clip_norm'):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['training']['grad_clip_norm']
                )

            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Clean up
        del input_ids, attention_mask, fitness, predictions, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate on query set."""
    model.eval()

    all_predictions = []
    all_fitness = []

    for input_ids, attention_mask, fitness in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        predictions, _ = model(input_ids, attention_mask)

        all_predictions.append(predictions.cpu())
        all_fitness.append(fitness)

        # Clean up
        del input_ids, attention_mask, predictions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    predictions = torch.cat(all_predictions).numpy()
    fitness = torch.cat(all_fitness).numpy()

    # Compute metrics
    mse = ((predictions - fitness) ** 2).mean()
    mae = np.abs(predictions - fitness).mean()

    # Handle edge cases for correlation
    if len(np.unique(predictions)) < 2 or len(np.unique(fitness)) < 2:
        spearman = 0.0
        pearson = 0.0
    else:
        spearman = stats.spearmanr(predictions, fitness).correlation
        pearson = stats.pearsonr(predictions, fitness)[0]

    return {
        'mse': float(mse),
        'mae': float(mae),
        'spearman': float(spearman),
        'pearson': float(pearson)
    }


# ---------------- Meta-Learning ----------------
def meta_train_task(
    model: nn.Module,
    task_path: str,
    tokenizer,
    device: torch.device,
    config: dict,
    scaler: Optional[GradScaler] = None
) -> Optional[Dict[str, float]]:
    """
    Meta-train on a single task (support/query split).

    Returns:
        Metrics dict or None if task is too small
    """
    try:
        dataset = ProteinGymDataset(
            task_path,
            max_len=config['data'].get('max_sequence_length')
        )
    except Exception as e:
        logging.error(f"Failed to load task {task_path}: {e}")
        return None

    if len(dataset) < config['data']['min_samples']:
        logging.warning(f"Skipping {task_path}: only {len(dataset)} samples")
        return None

    # Create support/query split
    n = len(dataset)
    indices = list(range(n))
    random.shuffle(indices)

    support_size = int(config['data']['support_frac'] * n)
    support_indices = indices[:support_size]
    query_indices = indices[support_size:]

    if len(query_indices) == 0:
        logging.warning(f"Skipping {task_path}: no query samples")
        return None

    # Create dataloaders
    support_dataset = torch.utils.data.Subset(dataset, support_indices)
    query_dataset = torch.utils.data.Subset(dataset, query_indices)

    collator = ProteinCollator(tokenizer, config['data']['max_sequence_length'])

    support_loader = DataLoader(
        support_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 0),
        collate_fn=collator,
        pin_memory=True
    )

    query_loader = DataLoader(
        query_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 0),
        collate_fn=collator,
        pin_memory=True
    )

    # Create task-specific optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['task_learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.01)
    )

    # Train on support set
    task_name = os.path.basename(task_path)
    logging.info(f"Training on {task_name} ({len(support_indices)} support / {len(query_indices)} query)")

    for epoch in range(config['training']['task_epochs']):
        train_loss = train_one_epoch(model, support_loader, optimizer, device, config, scaler)

        if epoch % config['logging']['log_interval'] == 0:
            logging.info(f"  Epoch {epoch+1}/{config['training']['task_epochs']}: Loss={train_loss:.4f}")

    # Evaluate on query set
    metrics = evaluate(model, query_loader, device)
    metrics['train_loss'] = train_loss
    metrics['task_name'] = task_name

    logging.info(
        f"  Query: MSE={metrics['mse']:.4f}, Spearman={metrics['spearman']:.4f}, "
        f"Pearson={metrics['pearson']:.4f}"
    )

    return metrics


def meta_train(
    task_paths: List[str],
    tokenizer,
    device: torch.device,
    config: dict,
    output_dir: Path
) -> nn.Module:
    """Meta-train across all tasks."""

    # Initialize model
    model = ESM2MoEModel(
        model_name=config['model']['esm2_model'],
        num_experts=config['model']['num_experts'],
        hidden_dim=config['model']['hidden_dim'],
        dropout=config['model']['dropout'],
        freeze_backbone=config['model']['freeze_backbone']
    ).to(device)

    # Mixed precision scaler
    scaler = GradScaler() if config['training']['mixed_precision'] else None

    # Track metrics
    all_metrics = []

    # Meta-train on each task
    for task_path in tqdm(task_paths, desc="Meta-training"):
        metrics = meta_train_task(model, task_path, tokenizer, device, config, scaler)
        if metrics:
            all_metrics.append(metrics)

    # Compute average metrics
    if all_metrics:
        avg_metrics = {
            'avg_mse': np.mean([m['mse'] for m in all_metrics]),
            'avg_spearman': np.mean([m['spearman'] for m in all_metrics]),
            'avg_pearson': np.mean([m['pearson'] for m in all_metrics])
        }
        logging.info(f"\nMeta-training complete:")
        logging.info(f"  Avg MSE: {avg_metrics['avg_mse']:.4f}")
        logging.info(f"  Avg Spearman: {avg_metrics['avg_spearman']:.4f}")
        logging.info(f"  Avg Pearson: {avg_metrics['avg_pearson']:.4f}")

    # Save model
    checkpoint_path = output_dir / 'meta_trained_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'metrics': all_metrics,
        'avg_metrics': avg_metrics if all_metrics else {}
    }, checkpoint_path)
    logging.info(f"Model saved to {checkpoint_path}")

    return model


def meta_test_task(
    model: nn.Module,
    task_path: str,
    tokenizer,
    device: torch.device,
    config: dict,
    scaler: Optional[GradScaler] = None
) -> Optional[Dict[str, float]]:
    """Meta-test on a single task with adaptation."""

    metrics = meta_train_task(model, task_path, tokenizer, device, config, scaler)
    return metrics


def meta_test(
    model: nn.Module,
    test_paths: List[str],
    tokenizer,
    device: torch.device,
    config: dict,
    output_dir: Path
):
    """Meta-test on all test tasks."""

    scaler = GradScaler() if config['training']['mixed_precision'] else None
    all_metrics = []

    logging.info("\n" + "="*80)
    logging.info("META-TESTING")
    logging.info("="*80)

    for task_path in tqdm(test_paths, desc="Meta-testing"):
        metrics = meta_test_task(model, task_path, tokenizer, device, config, scaler)
        if metrics:
            all_metrics.append(metrics)

    # Compute average test metrics
    if all_metrics:
        avg_metrics = {
            'test_avg_mse': np.mean([m['mse'] for m in all_metrics]),
            'test_avg_spearman': np.mean([m['spearman'] for m in all_metrics]),
            'test_avg_pearson': np.mean([m['pearson'] for m in all_metrics])
        }
        logging.info(f"\nMeta-testing complete:")
        logging.info(f"  Test Avg MSE: {avg_metrics['test_avg_mse']:.4f}")
        logging.info(f"  Test Avg Spearman: {avg_metrics['test_avg_spearman']:.4f}")
        logging.info(f"  Test Avg Pearson: {avg_metrics['test_avg_pearson']:.4f}")

        # Save results
        results_path = output_dir / 'test_results.yaml'
        with open(results_path, 'w') as f:
            yaml.dump({
                'avg_metrics': avg_metrics,
                'task_metrics': all_metrics
            }, f)
        logging.info(f"Results saved to {results_path}")


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description='Meta-learning for protein functionality prediction')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='outputs/meta_learning',
                       help='Output directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Load configuration
    config = load_meta_config(args.config)

    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    log_file = output_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    setup_logging(log_file)

    logging.info("="*80)
    logging.info("Meta-Learning for Protein Functionality Prediction")
    logging.info("="*80)
    logging.info(f"Config: {args.config}")
    logging.info(f"Output: {output_dir}")

    # Set random seed
    set_seed(config['training']['seed'])

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")

    # Load tokenizer
    logging.info(f"Loading tokenizer: {config['model']['esm2_model']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['esm2_model'])

    # Get task files
    train_dir = Path(config['data']['train_dir'])
    test_dir = Path(config['data']['test_dir'])

    train_paths = sorted([str(p) for p in train_dir.glob('*.csv')])
    test_paths = sorted([str(p) for p in test_dir.glob('*.csv')])

    logging.info(f"Found {len(train_paths)} training tasks")
    logging.info(f"Found {len(test_paths)} test tasks")

    if not train_paths:
        logging.error("No training files found!")
        return

    # Meta-train
    if args.resume:
        logging.info(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model = ESM2MoEModel(
            model_name=config['model']['esm2_model'],
            num_experts=config['model']['num_experts'],
            hidden_dim=config['model']['hidden_dim'],
            dropout=config['model']['dropout'],
            freeze_backbone=config['model']['freeze_backbone']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info("Checkpoint loaded, skipping meta-training")
    else:
        model = meta_train(train_paths, tokenizer, device, config, output_dir)

    # Meta-test
    if test_paths:
        meta_test(model, test_paths, tokenizer, device, config, output_dir)
    else:
        logging.warning("No test files found, skipping meta-testing")

    logging.info("\nDone!")


if __name__ == '__main__':
    main()
