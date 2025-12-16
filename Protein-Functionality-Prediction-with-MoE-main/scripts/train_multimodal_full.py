#!/usr/bin/env python3
"""
Multi-Modal Training: ESM2-35M + MSA Features
Full dataset training (173 proteins) targeting 0.55-0.65 Spearman

Features:
- ESM2-35M backbone (35M params)
- MSA evolutionary features (PSSM, conservation)
- Cross-attention fusion
- MoE architecture
- Full ProteinGym dataset

Target: 0.55-0.65 Spearman (+29-52% over baseline 0.4264)
"""

import os
import sys
import argparse
import logging
import random
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

# Import our multimodal components
from src.models.multimodal_fusion import MultiModalMoEModel
from src.data.msa_utils import simple_msa_features

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


class ProteinGymMSADataset(Dataset):
    """Dataset with MSA features"""
    def __init__(self, path, use_msa=True, max_len=None):
        df = pd.read_csv(path)
        self.sequences = df['mutated_sequence'].astype(str).tolist()
        self.fitness = df['DMS_score'].astype(float).tolist()
        self.use_msa = use_msa

        if max_len:
            filtered = [(s, f) for s, f in zip(self.sequences, self.fitness) if len(s) <= max_len]
            if filtered:
                self.sequences, self.fitness = zip(*filtered)
                self.sequences = list(self.sequences)
                self.fitness = list(self.fitness)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        fitness = torch.tensor(self.fitness[idx], dtype=torch.float32)

        # Generate MSA features (simple PSSM-like)
        if self.use_msa:
            msa_feat = simple_msa_features(sequence, num_homologs=50)
            msa_feat = torch.tensor(msa_feat, dtype=torch.float32)
        else:
            msa_feat = None

        return sequence, msa_feat, fitness


def collate_fn(batch, tokenizer, max_len=1024):
    """Collate function for batching with MSA"""
    sequences, msa_features, fitness = zip(*batch)

    # Tokenize sequences
    encoding = tokenizer(
        list(sequences),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len
    )

    # Stack MSA features (pad to max length in batch)
    if msa_features[0] is not None:
        max_seq_len = encoding["input_ids"].shape[1]
        batch_size = len(msa_features)

        # Pad MSA features
        padded_msa = []
        for msa in msa_features:
            if len(msa) < max_seq_len:
                # Pad with zeros
                padding = torch.zeros(max_seq_len - len(msa), msa.shape[1])
                msa = torch.cat([msa, padding], dim=0)
            elif len(msa) > max_seq_len:
                # Truncate
                msa = msa[:max_seq_len]
            padded_msa.append(msa)

        msa_batch = torch.stack(padded_msa)
    else:
        msa_batch = None

    return encoding["input_ids"], encoding["attention_mask"], msa_batch, torch.stack(fitness)


def train_task(model, task_path, tokenizer, device, config):
    """Train on one task (protein family)"""
    task_name = Path(task_path).name
    logging.info(f"Training on {task_name}")

    # Load dataset
    dataset = ProteinGymMSADataset(
        task_path,
        use_msa=config['use_msa'],
        max_len=config['max_len']
    )

    if len(dataset) < 10:
        logging.warning(f"  Skipping - too few samples ({len(dataset)})")
        return None

    # Split support/query (85/15 for more training data)
    support_size = int(len(dataset) * 0.85)
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

    # Train on support set (1 epoch)
    model.train()
    total_loss = 0.0
    num_batches = 0

    for input_ids, attention_mask, msa_features, fitness in support_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        fitness = fitness.to(device)

        if msa_features is not None:
            msa_features = msa_features.to(device)

        optimizer.zero_grad()

        with autocast():
            predictions = model(input_ids, attention_mask, msa_features)
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
        for input_ids, attention_mask, msa_features, fitness in query_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            if msa_features is not None:
                msa_features = msa_features.to(device)

            with autocast():
                predictions = model(input_ids, attention_mask, msa_features)

            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(fitness.numpy())

    # Compute metrics
    spearman, pearson, mse = compute_metrics(all_preds, all_targets)
    logging.info(f"  Query: MSE={mse:.4f}, Spearman={spearman:.4f}, Pearson={pearson:.4f}")

    return {
        'task': task_name,
        'train_loss': avg_loss,
        'mse': mse,
        'spearman': spearman,
        'pearson': pearson,
        'n_support': support_size,
        'n_query': query_size
    }


def main():
    parser = argparse.ArgumentParser(description='Multi-modal training with MSA features')
    parser.add_argument('--model', default='facebook/esm2_t12_35M_UR50D', help='ESM2 model')
    parser.add_argument('--data-dir', default='data/raw/Train_split', help='Training data directory')
    parser.add_argument('--output-dir', default='outputs/multimodal_full', help='Output directory')
    parser.add_argument('--use-msa', action='store_true', default=True, help='Use MSA features')
    parser.add_argument('--no-msa', action='store_false', dest='use_msa', help='Disable MSA features')
    parser.add_argument('--max-tasks', type=int, default=None, help='Max tasks (for testing)')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--max-len', type=int, default=1024, help='Max sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    logging.info(f"MSA features: {'ENABLED' if args.use_msa else 'DISABLED'}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and tokenizer
    logging.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = MultiModalMoEModel(
        esm2_model_name=args.model,
        num_experts=8,
        expert_hidden_dim=256,
        use_msa=args.use_msa,
        msa_dim=20,
        dropout=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model: {total_params:,} total params, {trainable_params:,} trainable")

    # Config
    config = {
        'use_msa': args.use_msa,
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
        logging.info(f"Found {len(task_files)} training tasks (FULL DATASET)")

    # Train on each task
    results = []
    for i, task_file in enumerate(tqdm(task_files, desc="Training")):
        result = train_task(model, task_file, tokenizer, device, config)
        if result:
            results.append(result)

        # Save checkpoint every 20 tasks
        if (i + 1) % 20 == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_{i+1}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'tasks_completed': i + 1,
                'results': results
            }, checkpoint_path)
            logging.info(f"Saved checkpoint: {checkpoint_path}")

            # Intermediate statistics
            if results:
                df_temp = pd.DataFrame(results)
                logging.info(f"  Progress: {i+1}/{len(task_files)} tasks")
                logging.info(f"  Current avg Spearman: {df_temp['spearman'].mean():.4f}")

    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'results': results
    }, final_path)
    logging.info(f"Saved final model to {final_path}")

    # Summary statistics
    if results:
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(args.output_dir, 'results.csv'), index=False)

        logging.info("\n" + "="*60)
        logging.info("TRAINING COMPLETE")
        logging.info("="*60)
        logging.info(f"Tasks completed: {len(results)}/{len(task_files)}")
        logging.info(f"Average Spearman: {df['spearman'].mean():.4f} ± {df['spearman'].std():.4f}")
        logging.info(f"Median Spearman: {df['spearman'].median():.4f}")
        logging.info(f"Average Pearson: {df['pearson'].mean():.4f} ± {df['pearson'].std():.4f}")
        logging.info(f"Average MSE: {df['mse'].mean():.4f} ± {df['mse'].std():.4f}")
        logging.info(f"Best task: {df.loc[df['spearman'].idxmax(), 'task']} (Spearman: {df['spearman'].max():.4f})")
        logging.info(f"Results saved to {args.output_dir}/results.csv")

        # Comparison to baseline
        baseline_spearman = 0.4264
        improvement = ((df['spearman'].mean() / baseline_spearman) - 1) * 100
        logging.info(f"\nBaseline Spearman: {baseline_spearman:.4f}")
        logging.info(f"Improvement: {improvement:+.1f}%")


if __name__ == '__main__':
    main()
