#!/usr/bin/env python3
"""
Baseline Training Scaled to ESM2-35M

Simply scales the working baseline (0.4264) to ESM2-35M.
NO fancy tricks - just proven approach with larger backbone.

Target: 0.45-0.48 Spearman (+5-10% improvement)
"""

import os
import sys
import argparse
import logging
import random
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(predictions, targets):
    """Compute Spearman, Pearson, and MSE"""
    if len(predictions) < 2:
        return 0.0, 0.0, 0.0

    spearman = float(spearmanr(predictions, targets)[0])
    pearson = float(pearsonr(predictions, targets)[0])
    mse = float(np.mean((np.array(predictions) - np.array(targets)) ** 2))

    return spearman, pearson, mse


class ProteinGymDataset(Dataset):
    def __init__(self, path, max_len=None):
        df = pd.read_csv(path)
        self.sequences = df['mutated_sequence'].astype(str).tolist()
        self.fitness = df['DMS_score'].astype(float).tolist()

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


# Model Components (from baseline)
class ExpertMLP(nn.Module):
    """Single expert network for MoE."""
    def __init__(self, in_dim, hidden_dim=256, dropout=0.1):
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

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MoEHead(nn.Module):
    """Mixture of Experts head with gating mechanism."""
    def __init__(self, in_dim, num_experts=8, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            ExpertMLP(in_dim, hidden_dim, dropout) for _ in range(num_experts)
        ])
        self.gate = nn.Sequential(
            nn.Linear(in_dim, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # Expert predictions
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        # Gate weights
        gate_weights = self.gate(x)

        # Weighted combination
        predictions = (expert_outputs * gate_weights).sum(dim=1)

        return predictions, gate_weights


class ESM2MoEModel(nn.Module):
    """ESM2 backbone with MoE head for protein fitness prediction."""
    def __init__(self, model_name, num_experts=8, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.embed_dim = self.backbone.config.hidden_size
        self.moe = MoEHead(self.embed_dim, num_experts, hidden_dim, dropout)

    def forward(self, input_ids, attention_mask):
        # Get ESM2 embeddings
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state

        # Mean pooling (weighted by attention mask)
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
        sum_embeddings = (embeddings * mask_expanded).sum(1)
        sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
        pooled = sum_embeddings / sum_mask

        # MoE prediction
        predictions, gate_weights = self.moe(pooled)

        return predictions


def collate_fn(batch, tokenizer, max_len=1024):
    """Collate function for batching"""
    sequences, fitness = zip(*batch)

    encoding = tokenizer(
        list(sequences),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len
    )

    return encoding["input_ids"], encoding["attention_mask"], torch.stack(fitness)


def train_task(model, task_path, tokenizer, device, config):
    """Train on one task (protein family)"""
    logging.info(f"Training on {Path(task_path).name}")

    # Load dataset
    dataset = ProteinGymDataset(task_path, max_len=config['max_len'])

    if len(dataset) < 10:
        logging.warning(f"  Skipping - too few samples ({len(dataset)})")
        return None

    # Split support/query (80/20 like baseline)
    support_size = int(len(dataset) * 0.8)
    query_size = len(dataset) - support_size
    support_set, query_set = torch.utils.data.random_split(dataset, [support_size, query_size])

    logging.info(f"  {support_size} support / {query_size} query")

    # Create loaders
    support_loader = DataLoader(
        support_set,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, config['max_len'])
    )

    query_loader = DataLoader(
        query_set,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, config['max_len'])
    )

    # Setup optimizer (task-specific)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    criterion = nn.MSELoss()
    scaler = GradScaler()

    # Train on support set (1 epoch like baseline)
    model.train()
    total_loss = 0.0
    num_batches = 0

    for input_ids, attention_mask, fitness in support_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        fitness = fitness.to(device)

        optimizer.zero_grad()

        with autocast():
            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, fitness)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    logging.info(f"  Train Loss: {avg_loss:.4f}")

    # Evaluate on query set
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for input_ids, attention_mask, fitness in query_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            with autocast():
                predictions = model(input_ids, attention_mask)

            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(fitness.numpy())

    # Compute metrics
    spearman, pearson, mse = compute_metrics(all_preds, all_targets)
    logging.info(f"  Query: MSE={mse:.4f}, Spearman={spearman:.4f}, Pearson={pearson:.4f}")

    return {
        'task': Path(task_path).name,
        'train_loss': avg_loss,
        'mse': mse,
        'spearman': spearman,
        'pearson': pearson,
        'n_support': support_size,
        'n_query': query_size
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='facebook/esm2_t12_35M_UR50D', help='ESM2 model')
    parser.add_argument('--data-dir', default='data/raw/Train_split', help='Training data directory')
    parser.add_argument('--output-dir', default='outputs/baseline_scaled', help='Output directory')
    parser.add_argument('--max-tasks', type=int, default=None, help='Max tasks for testing')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--max-len', type=int, default=1024, help='Max sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and tokenizer
    logging.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = ESM2MoEModel(
        model_name=args.model,
        num_experts=8,
        hidden_dim=256,
        dropout=0.1
    ).to(device)

    logging.info(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Config
    config = {
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': 0.01,
        'grad_clip': 1.0,
        'max_len': args.max_len
    }

    # Get task files
    task_files = sorted(Path(args.data_dir).glob('*.csv'))
    if args.max_tasks:
        task_files = task_files[:args.max_tasks]
        logging.info(f"Using {len(task_files)} tasks for testing")
    else:
        logging.info(f"Found {len(task_files)} training tasks")

    # Train on each task
    results = []
    for task_file in tqdm(task_files, desc="Training"):
        result = train_task(model, task_file, tokenizer, device, config)
        if result:
            results.append(result)

        # Save checkpoint every 20 tasks
        if len(results) % 20 == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_{len(results)}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_path)
    logging.info(f"Saved final model to {final_path}")

    # Summary statistics
    if results:
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(args.output_dir, 'results.csv'), index=False)

        logging.info("\n" + "="*50)
        logging.info("TRAINING COMPLETE")
        logging.info("="*50)
        logging.info(f"Tasks completed: {len(results)}")
        logging.info(f"Average Spearman: {df['spearman'].mean():.4f} ± {df['spearman'].std():.4f}")
        logging.info(f"Average Pearson: {df['pearson'].mean():.4f} ± {df['pearson'].std():.4f}")
        logging.info(f"Average MSE: {df['mse'].mean():.4f} ± {df['mse'].std():.4f}")
        logging.info(f"Results saved to {args.output_dir}/results.csv")


if __name__ == '__main__':
    main()
