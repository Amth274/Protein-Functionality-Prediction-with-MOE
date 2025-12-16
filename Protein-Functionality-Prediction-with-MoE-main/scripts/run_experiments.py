#!/usr/bin/env python3
"""
Comprehensive experiments for publication:
1. Multiple seeds for statistical significance
2. Ablation studies (model size, head architecture, meta-learning vs standard)
3. Protein category analysis
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import logging
import json
import argparse
from datetime import datetime
import random

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/experiments.log')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Model Definitions
# ============================================================================

class SimpleHead(nn.Module):
    """Simple linear head"""
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.head(x).squeeze(-1)


class MLPHead(nn.Module):
    """MLP head (default)"""
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, x):
        return self.head(x).squeeze(-1)


class DeepHead(nn.Module):
    """Deeper MLP head"""
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, x):
        return self.head(x).squeeze(-1)


class ProteinModel(nn.Module):
    """Flexible protein fitness prediction model"""
    def __init__(self, model_name, head_type='mlp', dropout=0.1, use_gradient_checkpointing=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.encoder.config.hidden_size

        if use_gradient_checkpointing and hasattr(self.encoder, 'gradient_checkpointing_enable'):
            self.encoder.gradient_checkpointing_enable()

        # Select head type
        if head_type == 'simple':
            self.head = SimpleHead(self.hidden_dim, dropout)
        elif head_type == 'mlp':
            self.head = MLPHead(self.hidden_dim, dropout)
        elif head_type == 'deep':
            self.head = DeepHead(self.hidden_dim, dropout)
        else:
            raise ValueError(f"Unknown head type: {head_type}")

    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden_states.mean(dim=1)

        return self.head(pooled)


# ============================================================================
# Dataset
# ============================================================================

class ProteinDataset(Dataset):
    def __init__(self, path, max_len=1024):
        df = pd.read_csv(path)
        self.sequences = df['mutated_sequence'].astype(str).tolist()
        self.fitness = df['DMS_score'].astype(float).tolist()

        filtered = [(s, f) for s, f in zip(self.sequences, self.fitness) if len(s) <= max_len]
        if filtered:
            self.sequences, self.fitness = zip(*filtered)
            self.sequences = list(self.sequences)
            self.fitness = list(self.fitness)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], torch.tensor(self.fitness[idx], dtype=torch.float32)


def collate_fn(batch, tokenizer, max_len=1024):
    sequences, fitness = zip(*batch)
    encoding = tokenizer(list(sequences), return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    return encoding["input_ids"], encoding["attention_mask"], torch.stack(fitness)


# ============================================================================
# Training Functions
# ============================================================================

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def meta_train_task(model, task_path, tokenizer, device, optimizer, scaler, batch_size=4, max_len=1024):
    """Meta-learning training on one protein"""
    dataset = ProteinDataset(task_path, max_len=max_len)

    if len(dataset) < 10:
        return None

    support_size = int(len(dataset) * 0.8)
    query_size = len(dataset) - support_size
    support_set, query_set = torch.utils.data.random_split(dataset, [support_size, query_size])

    support_loader = DataLoader(
        support_set, batch_size=batch_size, shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_len)
    )
    query_loader = DataLoader(
        query_set, batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_len)
    )

    # Train on support set
    model.train()
    for input_ids, attention_mask, targets in support_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        with autocast():
            predictions = model(input_ids, attention_mask)
            loss = F.mse_loss(predictions, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    # Evaluate on query set
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for input_ids, attention_mask, targets in query_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            with autocast():
                predictions = model(input_ids, attention_mask)

            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(targets.numpy())

    if len(all_preds) < 2:
        return None

    spearman = spearmanr(all_preds, all_targets)[0]
    return float(spearman) if not np.isnan(spearman) else 0.0


def standard_train_task(model, task_path, tokenizer, device, optimizer, scaler, batch_size=4, max_len=1024):
    """Standard training (no meta-learning split)"""
    dataset = ProteinDataset(task_path, max_len=max_len)

    if len(dataset) < 10:
        return None

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_len)
    )

    # Train on full dataset
    model.train()
    for input_ids, attention_mask, targets in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        with autocast():
            predictions = model(input_ids, attention_mask)
            loss = F.mse_loss(predictions, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    # Evaluate on same data (will overfit, but for comparison)
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for input_ids, attention_mask, targets in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            with autocast():
                predictions = model(input_ids, attention_mask)

            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(targets.numpy())

    spearman = spearmanr(all_preds, all_targets)[0]
    return float(spearman) if not np.isnan(spearman) else 0.0


def test_task(model, task_path, tokenizer, device, checkpoint, batch_size=2, max_len=1024, use_meta=True):
    """Test on one protein with optional meta-learning"""
    dataset = ProteinDataset(task_path, max_len=max_len)

    if len(dataset) < 10:
        return None

    if use_meta:
        # Meta-learning: fine-tune on support, eval on query
        support_size = int(len(dataset) * 0.8)
        query_size = len(dataset) - support_size
        support_set, query_set = torch.utils.data.random_split(dataset, [support_size, query_size])

        support_loader = DataLoader(
            support_set, batch_size=batch_size, shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, tokenizer, max_len)
        )
        query_loader = DataLoader(
            query_set, batch_size=batch_size, shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, tokenizer, max_len)
        )

        # Reload and fine-tune
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
        model.train()

        for input_ids, attention_mask, targets in support_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            with autocast():
                predictions = model(input_ids, attention_mask)
                loss = F.mse_loss(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        eval_loader = query_loader
    else:
        # Standard: just evaluate
        eval_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, tokenizer, max_len)
        )

    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for input_ids, attention_mask, targets in eval_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            with autocast():
                predictions = model(input_ids, attention_mask)

            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(targets.numpy())

    if len(all_preds) < 2:
        return None

    spearman = spearmanr(all_preds, all_targets)[0]
    return float(spearman) if not np.isnan(spearman) else 0.0


# ============================================================================
# Experiment Functions
# ============================================================================

def run_seed_experiment(model_name, seeds, output_dir, num_train=50, num_test=10):
    """Run experiment with multiple seeds"""
    logger.info(f"\n{'='*60}")
    logger.info(f"SEED EXPERIMENT: {model_name}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"{'='*60}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dir = Path('data/raw/Train_split')
    test_dir = Path('data/raw/Test_split')
    train_files = sorted(train_dir.glob('*.csv'))[:num_train]
    test_files = sorted(test_dir.glob('*.csv'))[:num_test]

    all_results = []

    for seed in seeds:
        logger.info(f"\n--- Seed {seed} ---")
        set_seed(seed)

        # Initialize model
        model = ProteinModel(model_name, head_type='mlp', dropout=0.1).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
        scaler = GradScaler()

        # Train
        train_results = []
        for task_file in tqdm(train_files, desc=f"Training (seed={seed})"):
            result = meta_train_task(model, task_file, tokenizer, device, optimizer, scaler, batch_size=4)
            if result is not None:
                train_results.append(result)

        train_avg = np.mean(train_results) if train_results else 0
        logger.info(f"Train avg Spearman: {train_avg:.4f}")

        # Save checkpoint
        checkpoint = {'model_state_dict': model.state_dict(), 'seed': seed}

        # Test
        test_results = []
        for task_file in tqdm(test_files, desc=f"Testing (seed={seed})"):
            model.load_state_dict(checkpoint['model_state_dict'])
            result = test_task(model, task_file, tokenizer, device, checkpoint, batch_size=2, use_meta=True)
            if result is not None:
                test_results.append({'protein': task_file.stem, 'spearman': result})

        test_avg = np.mean([r['spearman'] for r in test_results]) if test_results else 0
        logger.info(f"Test avg Spearman: {test_avg:.4f}")

        all_results.append({
            'seed': seed,
            'train_avg': train_avg,
            'test_avg': test_avg,
            'test_results': test_results
        })

        # Clean up
        del model, optimizer, scaler
        torch.cuda.empty_cache()

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/seed_experiment.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary statistics
    test_avgs = [r['test_avg'] for r in all_results]
    summary = {
        'model': model_name,
        'seeds': seeds,
        'mean': np.mean(test_avgs),
        'std': np.std(test_avgs),
        'min': np.min(test_avgs),
        'max': np.max(test_avgs)
    }

    logger.info(f"\nSEED EXPERIMENT SUMMARY:")
    logger.info(f"Mean: {summary['mean']:.4f} ± {summary['std']:.4f}")
    logger.info(f"Range: [{summary['min']:.4f}, {summary['max']:.4f}]")

    return summary, all_results


def run_model_size_ablation(output_dir, num_train=30, num_test=10):
    """Ablation study on model size"""
    logger.info(f"\n{'='*60}")
    logger.info("MODEL SIZE ABLATION")
    logger.info(f"{'='*60}")

    models = [
        ('facebook/esm2_t6_8M_UR50D', '8M', 320),
        ('facebook/esm2_t12_35M_UR50D', '35M', 480),
        ('facebook/esm2_t30_150M_UR50D', '150M', 640),
        ('facebook/esm2_t33_650M_UR50D', '650M', 1280),
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dir = Path('data/raw/Train_split')
    test_dir = Path('data/raw/Test_split')
    train_files = sorted(train_dir.glob('*.csv'))[:num_train]
    test_files = sorted(test_dir.glob('*.csv'))[:num_test]

    results = []
    set_seed(42)

    for model_name, size_name, hidden_dim in models:
        logger.info(f"\n--- {size_name} ({model_name}) ---")

        try:
            model = ProteinModel(model_name, head_type='mlp', dropout=0.1).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
            scaler = GradScaler()

            # Train
            train_results = []
            for task_file in tqdm(train_files, desc=f"Training {size_name}"):
                result = meta_train_task(model, task_file, tokenizer, device, optimizer, scaler, batch_size=4)
                if result is not None:
                    train_results.append(result)

            train_avg = np.mean(train_results) if train_results else 0

            # Save checkpoint
            checkpoint = {'model_state_dict': model.state_dict()}

            # Test
            test_results = []
            for task_file in tqdm(test_files, desc=f"Testing {size_name}"):
                model.load_state_dict(checkpoint['model_state_dict'])
                result = test_task(model, task_file, tokenizer, device, checkpoint, batch_size=2, use_meta=True)
                if result is not None:
                    test_results.append(result)

            test_avg = np.mean(test_results) if test_results else 0

            results.append({
                'model': model_name,
                'size': size_name,
                'hidden_dim': hidden_dim,
                'train_avg': train_avg,
                'test_avg': test_avg
            })

            logger.info(f"{size_name}: Train={train_avg:.4f}, Test={test_avg:.4f}")

            del model, optimizer, scaler
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error with {size_name}: {e}")
            results.append({
                'model': model_name,
                'size': size_name,
                'hidden_dim': hidden_dim,
                'train_avg': None,
                'test_avg': None,
                'error': str(e)
            })

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/model_size_ablation.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_head_ablation(model_name, output_dir, num_train=30, num_test=10):
    """Ablation study on head architecture"""
    logger.info(f"\n{'='*60}")
    logger.info("HEAD ARCHITECTURE ABLATION")
    logger.info(f"{'='*60}")

    head_types = ['simple', 'mlp', 'deep']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dir = Path('data/raw/Train_split')
    test_dir = Path('data/raw/Test_split')
    train_files = sorted(train_dir.glob('*.csv'))[:num_train]
    test_files = sorted(test_dir.glob('*.csv'))[:num_test]

    results = []
    set_seed(42)

    for head_type in head_types:
        logger.info(f"\n--- Head: {head_type} ---")

        model = ProteinModel(model_name, head_type=head_type, dropout=0.1).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
        scaler = GradScaler()

        # Train
        train_results = []
        for task_file in tqdm(train_files, desc=f"Training {head_type}"):
            result = meta_train_task(model, task_file, tokenizer, device, optimizer, scaler, batch_size=4)
            if result is not None:
                train_results.append(result)

        train_avg = np.mean(train_results) if train_results else 0

        # Save checkpoint
        checkpoint = {'model_state_dict': model.state_dict()}

        # Test
        test_results = []
        for task_file in tqdm(test_files, desc=f"Testing {head_type}"):
            model.load_state_dict(checkpoint['model_state_dict'])
            result = test_task(model, task_file, tokenizer, device, checkpoint, batch_size=2, use_meta=True)
            if result is not None:
                test_results.append(result)

        test_avg = np.mean(test_results) if test_results else 0

        results.append({
            'head_type': head_type,
            'train_avg': train_avg,
            'test_avg': test_avg
        })

        logger.info(f"{head_type}: Train={train_avg:.4f}, Test={test_avg:.4f}")

        del model, optimizer, scaler
        torch.cuda.empty_cache()

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/head_ablation.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_meta_vs_standard_ablation(model_name, output_dir, num_train=30, num_test=10):
    """Ablation: Meta-learning vs standard training"""
    logger.info(f"\n{'='*60}")
    logger.info("META-LEARNING VS STANDARD ABLATION")
    logger.info(f"{'='*60}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dir = Path('data/raw/Train_split')
    test_dir = Path('data/raw/Test_split')
    train_files = sorted(train_dir.glob('*.csv'))[:num_train]
    test_files = sorted(test_dir.glob('*.csv'))[:num_test]

    results = []

    for use_meta, name in [(True, 'meta-learning'), (False, 'standard')]:
        logger.info(f"\n--- {name} ---")
        set_seed(42)

        model = ProteinModel(model_name, head_type='mlp', dropout=0.1).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
        scaler = GradScaler()

        # Train
        train_results = []
        train_fn = meta_train_task if use_meta else standard_train_task
        for task_file in tqdm(train_files, desc=f"Training ({name})"):
            result = train_fn(model, task_file, tokenizer, device, optimizer, scaler, batch_size=4)
            if result is not None:
                train_results.append(result)

        train_avg = np.mean(train_results) if train_results else 0

        # Save checkpoint
        checkpoint = {'model_state_dict': model.state_dict()}

        # Test
        test_results = []
        for task_file in tqdm(test_files, desc=f"Testing ({name})"):
            model.load_state_dict(checkpoint['model_state_dict'])
            result = test_task(model, task_file, tokenizer, device, checkpoint, batch_size=2, use_meta=use_meta)
            if result is not None:
                test_results.append(result)

        test_avg = np.mean(test_results) if test_results else 0

        results.append({
            'method': name,
            'use_meta': use_meta,
            'train_avg': train_avg,
            'test_avg': test_avg
        })

        logger.info(f"{name}: Train={train_avg:.4f}, Test={test_avg:.4f}")

        del model, optimizer, scaler
        torch.cuda.empty_cache()

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/meta_vs_standard.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


def analyze_protein_categories(test_results_path, output_dir):
    """Analyze performance by protein category"""
    logger.info(f"\n{'='*60}")
    logger.info("PROTEIN CATEGORY ANALYSIS")
    logger.info(f"{'='*60}")

    # Load test results
    df = pd.read_csv(test_results_path)

    # Define categories based on protein names
    categories = {
        'viral': ['HIV', 'HV1', 'INFA', 'DEN', 'LAMBD', 'BPT7', 'AAV'],
        'human': ['HUMAN'],
        'bacterial': ['ECOLI', 'STRSG', 'THEMA', 'MYCTU', 'PSEAI'],
        'yeast': ['YEAST'],
        'plant': ['ARATH'],
        'other': []
    }

    # Categorize each protein
    df['category'] = 'other'
    for cat, keywords in categories.items():
        if cat != 'other':
            for kw in keywords:
                df.loc[df['protein'].str.contains(kw, case=False), 'category'] = cat

    # Calculate statistics by category
    category_stats = df.groupby('category').agg({
        'spearman': ['mean', 'std', 'count', 'min', 'max']
    }).round(4)

    category_stats.columns = ['mean', 'std', 'count', 'min', 'max']
    category_stats = category_stats.reset_index()

    logger.info("\nPerformance by Category:")
    logger.info(category_stats.to_string())

    # Identify problematic proteins
    low_performers = df[df['spearman'] < 0.3].sort_values('spearman')
    high_performers = df[df['spearman'] > 0.8].sort_values('spearman', ascending=False)

    logger.info(f"\nLow performers (<0.3 Spearman):")
    for _, row in low_performers.iterrows():
        logger.info(f"  {row['protein']}: {row['spearman']:.4f} ({row['category']})")

    logger.info(f"\nHigh performers (>0.8 Spearman):")
    for _, row in high_performers.head(10).iterrows():
        logger.info(f"  {row['protein']}: {row['spearman']:.4f} ({row['category']})")

    # Save analysis
    os.makedirs(output_dir, exist_ok=True)

    analysis = {
        'category_stats': category_stats.to_dict('records'),
        'low_performers': low_performers.to_dict('records'),
        'high_performers': high_performers.head(10).to_dict('records'),
        'overall': {
            'mean': df['spearman'].mean(),
            'std': df['spearman'].std(),
            'median': df['spearman'].median()
        }
    }

    with open(f"{output_dir}/protein_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)

    df.to_csv(f"{output_dir}/test_results_categorized.csv", index=False)

    return analysis


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive experiments')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['all', 'seeds', 'model_size', 'head', 'meta_vs_standard', 'protein_analysis'],
                       help='Which experiment to run')
    parser.add_argument('--output_dir', type=str, default='outputs/experiments',
                       help='Output directory')
    parser.add_argument('--num_train', type=int, default=50, help='Number of training proteins')
    parser.add_argument('--num_test', type=int, default=15, help='Number of test proteins')
    parser.add_argument('--seeds', type=str, default='42,123,456', help='Comma-separated seeds')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(',')]
    model_650m = 'facebook/esm2_t33_650M_UR50D'
    model_35m = 'facebook/esm2_t12_35M_UR50D'  # Faster for ablations

    results = {}

    if args.experiment in ['all', 'seeds']:
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT 1: MULTIPLE SEEDS")
        logger.info("="*80)
        summary, seed_results = run_seed_experiment(
            model_650m, seeds, args.output_dir,
            num_train=args.num_train, num_test=args.num_test
        )
        results['seeds'] = {'summary': summary, 'details': seed_results}

    if args.experiment in ['all', 'model_size']:
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT 2: MODEL SIZE ABLATION")
        logger.info("="*80)
        size_results = run_model_size_ablation(
            args.output_dir, num_train=args.num_train, num_test=args.num_test
        )
        results['model_size'] = size_results

    if args.experiment in ['all', 'head']:
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT 3: HEAD ARCHITECTURE ABLATION")
        logger.info("="*80)
        head_results = run_head_ablation(
            model_35m, args.output_dir, num_train=args.num_train, num_test=args.num_test
        )
        results['head'] = head_results

    if args.experiment in ['all', 'meta_vs_standard']:
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT 4: META-LEARNING VS STANDARD")
        logger.info("="*80)
        meta_results = run_meta_vs_standard_ablation(
            model_35m, args.output_dir, num_train=args.num_train, num_test=args.num_test
        )
        results['meta_vs_standard'] = meta_results

    if args.experiment in ['all', 'protein_analysis']:
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT 5: PROTEIN CATEGORY ANALYSIS")
        logger.info("="*80)
        test_results_path = 'outputs/esm650m/test_results.csv'
        if os.path.exists(test_results_path):
            analysis = analyze_protein_categories(test_results_path, args.output_dir)
            results['protein_analysis'] = analysis
        else:
            logger.warning(f"Test results not found at {test_results_path}")

    # Save all results
    with open(f"{args.output_dir}/all_experiments.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("\n" + "="*80)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info(f"Results saved to {args.output_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
