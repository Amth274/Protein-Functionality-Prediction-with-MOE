#!/usr/bin/env python3
"""
Optimized MSA-Enhanced Training Script

Fixes from failure analysis:
1. Proper test/train split evaluation
2. Correct learning rate for ESM2-35M (5e-6)
3. MSA evolutionary features for performance boost
4. Full 173-protein training

Target: 0.50-0.58 Spearman on test set (+17-36% over baseline 0.4264)
"""

import os
import sys
import argparse
import logging
import random
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings('ignore')

# Add project root
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

# Our components
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
    if len(predictions) < 2:
        return 0.0, 0.0, 0.0

    spearman = float(spearmanr(predictions, targets)[0])
    pearson = float(pearsonr(predictions, targets)[0])
    mse = float(np.mean((np.array(predictions) - np.array(targets)) ** 2))

    return spearman, pearson, mse


class ProteinGymMSADataset(Dataset):
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

        if self.use_msa:
            msa_feat = simple_msa_features(sequence, num_homologs=50)
            msa_feat = torch.tensor(msa_feat, dtype=torch.float32)
        else:
            msa_feat = None

        return sequence, msa_feat, fitness


def collate_fn(batch, tokenizer, max_len=1024):
    sequences, msa_features, fitness = zip(*batch)

    encoding = tokenizer(
        list(sequences),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len
    )

    if msa_features[0] is not None:
        max_seq_len = encoding["input_ids"].shape[1]

        padded_msa = []
        for msa in msa_features:
            if len(msa) < max_seq_len:
                padding = torch.zeros(max_seq_len - len(msa), msa.shape[1])
                msa = torch.cat([msa, padding], dim=0)
            elif len(msa) > max_seq_len:
                msa = msa[:max_seq_len]
            padded_msa.append(msa)

        msa_batch = torch.stack(padded_msa)
    else:
        msa_batch = None

    return encoding["input_ids"], encoding["attention_mask"], msa_batch, torch.stack(fitness)


def train_task(model, task_path, tokenizer, device, config):
    """Train on one task"""
    task_name = Path(task_path).name

    dataset = ProteinGymMSADataset(
        task_path,
        use_msa=config['use_msa'],
        max_len=config['max_len']
    )

    if len(dataset) < 10:
        return None

    # 80/20 split like baseline
    support_size = int(len(dataset) * 0.8)
    query_size = len(dataset) - support_size
    support_set, query_set = torch.utils.data.random_split(dataset, [support_size, query_size])

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    criterion = nn.MSELoss()
    scaler = GradScaler()

    # Training
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

    # Evaluation
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

    spearman, pearson, mse = compute_metrics(all_preds, all_targets)

    return {
        'task': task_name,
        'train_loss': avg_loss,
        'mse': mse,
        'spearman': spearman,
        'pearson': pearson,
        'n_support': support_size,
        'n_query': query_size
    }


def meta_test(model, test_dir, tokenizer, device, config):
    """Evaluate on held-out test set (like baseline)"""
    logging.info("\n" + "="*60)
    logging.info("META-TESTING ON HELD-OUT TEST SET")
    logging.info("="*60)

    test_files = sorted(Path(test_dir).glob('*.csv'))
    logging.info(f"Found {len(test_files)} test tasks")

    results = []

    for test_file in tqdm(test_files, desc="Meta-testing"):
        result = train_task(model, test_file, tokenizer, device, config)
        if result:
            results.append(result)
            logging.info(f"  {result['task']}: Spearman={result['spearman']:.4f}")

    if results:
        df = pd.DataFrame(results)

        logging.info("\n" + "="*60)
        logging.info("TEST SET RESULTS (comparable to baseline)")
        logging.info("="*60)
        logging.info(f"Tasks: {len(results)}")
        logging.info(f"Avg Spearman: {df['spearman'].mean():.4f} ± {df['spearman'].std():.4f}")
        logging.info(f"Median Spearman: {df['spearman'].median():.4f}")
        logging.info(f"Avg Pearson: {df['pearson'].mean():.4f}")
        logging.info(f"Avg MSE: {df['mse'].mean():.4f}")

        # Comparison
        baseline = 0.4264
        improvement = ((df['spearman'].mean() / baseline) - 1) * 100
        logging.info(f"\nBaseline: {baseline:.4f}")
        logging.info(f"Improvement: {improvement:+.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='facebook/esm2_t12_35M_UR50D')
    parser.add_argument('--train-dir', default='data/raw/Train_split')
    parser.add_argument('--test-dir', default='data/raw/Test_split')
    parser.add_argument('--output-dir', default='outputs/msa_optimized')
    parser.add_argument('--use-msa', action='store_true', default=True)
    parser.add_argument('--no-msa', action='store_false', dest='use_msa')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-6, help='Lower LR for larger model')
    parser.add_argument('--max-len', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info("="*60)
    logging.info("MSA-ENHANCED TRAINING")
    logging.info("="*60)
    logging.info(f"Device: {device}")
    logging.info(f"Model: {args.model}")
    logging.info(f"MSA features: {'ENABLED' if args.use_msa else 'DISABLED'}")
    logging.info(f"Learning rate: {args.lr}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
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
    logging.info(f"Parameters: {total_params:,}")

    config = {
        'use_msa': args.use_msa,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': 0.01,
        'grad_clip': 1.0,
        'max_len': args.max_len
    }

    # Train on 173 proteins
    train_files = sorted(Path(args.train_dir).glob('*.csv'))
    logging.info(f"\nTraining on {len(train_files)} proteins...")

    train_results = []
    for i, train_file in enumerate(tqdm(train_files, desc="Training")):
        result = train_task(model, train_file, tokenizer, device, config)
        if result:
            train_results.append(result)

        # Checkpoints
        if (i + 1) % 50 == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_{i+1}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'train_results': train_results
            }, checkpoint_path)

            if train_results:
                df_temp = pd.DataFrame(train_results)
                logging.info(f"\n  Progress: {i+1}/{len(train_files)}")
                logging.info(f"  Train avg Spearman: {df_temp['spearman'].mean():.4f}")

    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'train_results': train_results
    }, final_path)

    # Save training results
    if train_results:
        df_train = pd.DataFrame(train_results)
        df_train.to_csv(os.path.join(args.output_dir, 'train_results.csv'), index=False)
        logging.info(f"\nTraining complete: {len(train_results)} proteins")
        logging.info(f"Train avg Spearman: {df_train['spearman'].mean():.4f}")

    # CRITICAL: Test on held-out 44 proteins
    test_results = meta_test(model, args.test_dir, tokenizer, device, config)

    if test_results:
        df_test = pd.DataFrame(test_results)
        df_test.to_csv(os.path.join(args.output_dir, 'test_results.csv'), index=False)

        # Save summary
        summary = {
            'model': args.model,
            'msa_enabled': args.use_msa,
            'learning_rate': args.lr,
            'train_proteins': len(train_results),
            'test_proteins': len(test_results),
            'train_spearman': float(df_train['spearman'].mean()),
            'test_spearman': float(df_test['spearman'].mean()),
            'test_pearson': float(df_test['pearson'].mean()),
            'baseline_spearman': 0.4264,
            'improvement_pct': float(((df_test['spearman'].mean() / 0.4264) - 1) * 100)
        }

        with open(os.path.join(args.output_dir, 'summary.yaml'), 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)

        logging.info(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
