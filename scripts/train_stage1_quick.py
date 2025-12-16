#!/usr/bin/env python3
"""
Stage 1 Training Script - Quick Version for Testing

Uses ESM2-35M (faster than 650M) with Stage 1 optimizations:
- Ranking loss for Spearman optimization
- EMA for training stability
- Multi-epoch training
- Cosine LR schedule

Usage:
    python scripts/train_stage1_quick.py --max-tasks 10  # Test on 10 proteins
    python scripts/train_stage1_quick.py  # Full training on all proteins
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import all required libraries first
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np
import yaml
import random
from scipy.stats import spearmanr

# Import Stage 1 components
from src.models import CombinedLoss, EMAWrapper
from src.models.multimodal_fusion import MultiModalMoEModel

# Self-contained utilities
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_spearman(predictions, targets):
    if len(predictions) < 2:
        return 0.0
    return float(spearmanr(predictions, targets)[0])

class ProteinGymDataset(torch.utils.data.Dataset):
    def __init__(self, path, seq_col='mutated_sequence', fitness_col='DMS_score', max_len=None):
        df = pd.read_csv(path)
        self.sequences = df[seq_col].astype(str).tolist()
        self.fitness = df[fitness_col].astype(float).tolist()

        if max_len:
            filtered = [(s, f) for s, f in zip(self.sequences, self.fitness) if len(s) <= max_len]
            if filtered:
                self.sequences, self.fitness = zip(*filtered)
                self.sequences = list(self.sequences)
                self.fitness = list(self.fitness)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], torch.tensor(self.fitness[idx], dtype=torch.float32)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def meta_train_task_stage1(model, task_path, tokenizer, device, config, scaler, criterion, ema):
    """
    Meta-train on a single task with Stage 1 optimizations.
    """
    logging.info(f"Training on {Path(task_path).name}")

    try:
        # Load dataset
        dataset = ProteinGymDataset(
            path=task_path,
            max_len=config['data']['max_sequence_length']
        )

        if len(dataset) < config['data']['min_samples']:
            logging.warning(f"Skipping {task_path}: insufficient samples")
            return None

        # Support/query split
        support_size = int(len(dataset) * config['data']['support_frac'])
        query_size = len(dataset) - support_size

        support_dataset, query_dataset = torch.utils.data.random_split(
            dataset, [support_size, query_size]
        )

        logging.info(f"  {len(support_dataset)} support / {len(query_dataset)} query")

        # Create dataloaders
        support_loader = DataLoader(
            support_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=0
        )

        query_loader = DataLoader(
            query_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=0
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['task_learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        # Training loop (multiple epochs for Stage 1)
        model.train()
        task_epochs = config['training'].get('task_epochs', 1)

        for epoch in range(task_epochs):
            total_loss = 0
            num_batches = 0

            for batch_idx, (sequences, targets) in enumerate(support_loader):
                sequences = list(sequences)
                targets = targets.to(device)

                # Tokenize
                inputs = tokenizer(
                    sequences,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=config['data']['max_sequence_length']
                ).to(device)

                # Forward with AMP
                with autocast():
                    predictions = model(inputs['input_ids'], inputs['attention_mask'])
                    loss, loss_dict = criterion(predictions, targets)

                # Backward
                scaler.scale(loss).backward()

                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['training']['grad_clip_norm']
                )

                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Update EMA
                if ema is not None:
                    ema.update()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            logging.info(f"  Epoch {epoch+1}/{task_epochs}: Loss={avg_loss:.4f}")

        # Evaluate on query set with EMA
        model.eval()
        all_predictions = []
        all_targets = []

        eval_context = ema if ema is not None else torch.no_grad()

        with eval_context:
            for sequences, targets in query_loader:
                sequences = list(sequences)

                inputs = tokenizer(
                    sequences,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=config['data']['max_sequence_length']
                ).to(device)

                with torch.no_grad():
                    predictions = model(inputs['input_ids'], inputs['attention_mask'])

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.numpy())

        # Compute metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        mse = float(np.mean((all_predictions - all_targets) ** 2))
        spearman = compute_spearman(all_predictions, all_targets)
        pearson = float(np.corrcoef(all_predictions, all_targets)[0, 1])

        logging.info(f"  Query: MSE={mse:.4f}, Spearman={spearman:.4f}, Pearson={pearson:.4f}")

        return {
            'task_name': Path(task_path).name,
            'train_loss': avg_loss,
            'mse': mse,
            'spearman': spearman,
            'pearson': pearson
        }

    except Exception as e:
        logging.error(f"Error processing {task_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Stage 1 Training (Quick)')
    parser.add_argument('--max-tasks', type=int, default=None,
                        help='Maximum number of tasks to train on (for testing)')
    parser.add_argument('--output-dir', default='outputs/stage1_quick_run',
                        help='Output directory')
    args = parser.parse_args()

    # Create modified config for Stage 1 quick test
    config = {
        'model': {
            'esm2_model': 'facebook/esm2_t12_35M_UR50D',  # 35M: faster than 650M
            'num_experts': 8,
            'expert_hidden_dim': 512,
            'dropout': 0.15,
            'freeze_esm2': False,
            'use_msa': False  # Stage 1: no MSA
        },
        'data': {
            'train_dir': 'data/raw/Train_split',
            'test_dir': 'data/raw/Test_split',
            'max_sequence_length': 4096,
            'min_samples': 10,
            'support_frac': 0.85,
        },
        'training': {
            'task_epochs': 2,  # 2 epochs for speed
            'batch_size': 4,
            'task_learning_rate': 0.00001,
            'weight_decay': 0.02,
            'grad_clip_norm': 1.0,
            'mixed_precision': True,
            'seed': 42,
            'mse_weight': 0.5,
            'ranking_weight': 0.5,
            'use_ema': True,
            'ema_decay': 0.999
        }
    }

    # Set seed
    set_seed(config['training']['seed'])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['esm2_model'])

    # Create Stage 1 model
    logging.info("Creating Stage 1 model...")
    model = MultiModalMoEModel(
        esm2_model_name=config['model']['esm2_model'],
        num_experts=config['model']['num_experts'],
        expert_hidden_dim=config['model']['expert_hidden_dim'],
        dropout=config['model']['dropout'],
        freeze_esm2=config['model']['freeze_esm2'],
        use_msa=False  # Stage 1
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model has {num_params:,} parameters")

    # Create combined loss (MSE + Ranking)
    criterion = CombinedLoss(
        mse_weight=config['training']['mse_weight'],
        ranking_weight=config['training']['ranking_weight']
    )

    # Create EMA
    ema = EMAWrapper(model, decay=config['training']['ema_decay'], enabled=True)

    # Create scaler for mixed precision
    scaler = GradScaler() if config['training']['mixed_precision'] else None

    # Get task list
    train_dir = Path(config['data']['train_dir'])
    train_tasks = sorted(list(train_dir.glob('*.csv')))

    if args.max_tasks:
        train_tasks = train_tasks[:args.max_tasks]
        logging.info(f"Using {len(train_tasks)} tasks for testing")

    logging.info(f"Found {len(train_tasks)} training tasks")

    # Training loop
    all_results = []

    for task_idx, task_path in enumerate(tqdm(train_tasks, desc="Training")):
        result = meta_train_task_stage1(
            model, task_path, tokenizer, device, config, scaler, criterion, ema
        )

        if result is not None:
            all_results.append(result)

        # Save checkpoint every 10 tasks
        if (task_idx + 1) % 10 == 0:
            checkpoint_path = output_dir / f'checkpoint_task{task_idx+1}.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.state_dict() if ema else None,
                'task_idx': task_idx,
                'config': config
            }, checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")

    # Save final model with EMA weights
    if ema:
        ema.ema.apply_shadow()

    final_model_path = output_dir / 'final_model.pth'
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model saved to {final_model_path}")

    # Compute average metrics
    if all_results:
        avg_spearman = np.mean([r['spearman'] for r in all_results])
        avg_pearson = np.mean([r['pearson'] for r in all_results])
        avg_mse = np.mean([r['mse'] for r in all_results])

        logging.info("\n" + "="*70)
        logging.info("Stage 1 Training Complete!")
        logging.info(f"  Average Spearman: {avg_spearman:.4f}")
        logging.info(f"  Average Pearson: {avg_pearson:.4f}")
        logging.info(f"  Average MSE: {avg_mse:.4f}")
        logging.info("="*70)

        # Save results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_dir / 'training_results.csv', index=False)

        # Save summary
        summary = {
            'avg_spearman': float(avg_spearman),
            'avg_pearson': float(avg_pearson),
            'avg_mse': float(avg_mse),
            'num_tasks': len(all_results),
            'config': config
        }

        with open(output_dir / 'summary.yaml', 'w') as f:
            yaml.dump(summary, f)

        logging.info(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()
