#!/usr/bin/env python3
"""
Full test evaluation for ESM2-650M model.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/test_esm650m.log')
    ]
)
logger = logging.getLogger(__name__)


class ESM650MModel(nn.Module):
    def __init__(self, model_name='facebook/esm2_t33_650M_UR50D', dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.encoder.config.hidden_size

        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 4, 1)
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden_states.mean(dim=1)

        output = self.head(pooled)
        return output.squeeze(-1)


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


def test_task(model, task_path, tokenizer, device, batch_size=2, max_len=1024):
    """Evaluate on one protein with meta-learning style (train on support, eval on query)."""
    dataset = ProteinDataset(task_path, max_len=max_len)

    if len(dataset) < 10:
        return None

    # 80/20 split
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

    # Quick fine-tune on support set
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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load model
    model_path = 'outputs/esm650m/final_model.pt'
    logger.info(f"Loading model from {model_path}")

    model = ESM650MModel(model_name='facebook/esm2_t33_650M_UR50D', dropout=0.1).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Model loaded successfully")

    tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')

    # Get test files
    test_dir = Path('data/raw/Test_split')
    test_files = sorted(test_dir.glob('*.csv'))
    logger.info(f"Found {len(test_files)} test files")

    # Run full test
    logger.info("\n" + "="*60)
    logger.info("FULL META-TESTING ON 44 TEST PROTEINS")
    logger.info("="*60)

    results = []
    for test_file in tqdm(test_files, desc="Testing"):
        # Reload model weights for each protein (meta-learning)
        model.load_state_dict(checkpoint['model_state_dict'])

        spearman = test_task(model, test_file, tokenizer, device)
        if spearman is not None:
            results.append({
                'protein': test_file.stem,
                'spearman': spearman
            })
            logger.info(f"  {test_file.name}: Spearman={spearman:.4f}")

    # Calculate statistics
    results_df = pd.DataFrame(results)
    avg_spearman = results_df['spearman'].mean()
    median_spearman = results_df['spearman'].median()
    std_spearman = results_df['spearman'].std()

    logger.info("\n" + "="*60)
    logger.info("FINAL TEST RESULTS - ESM2-650M")
    logger.info("="*60)
    logger.info(f"Test proteins evaluated: {len(results)}")
    logger.info(f"Average Spearman: {avg_spearman:.4f}")
    logger.info(f"Median Spearman: {median_spearman:.4f}")
    logger.info(f"Std Spearman: {std_spearman:.4f}")
    logger.info(f"Baseline: 0.4264")
    logger.info(f"SOTA (SaProt+TTT): 0.62")
    logger.info(f"Improvement vs Baseline: +{((avg_spearman - 0.4264) / 0.4264 * 100):.1f}%")
    logger.info(f"Improvement vs SOTA: +{((avg_spearman - 0.62) / 0.62 * 100):.1f}%")
    logger.info("="*60)

    # Save results
    results_df.to_csv('outputs/esm650m/test_results.csv', index=False)

    with open('outputs/esm650m/final_summary.txt', 'w') as f:
        f.write("ESM2-650M Final Test Results\n")
        f.write("="*40 + "\n")
        f.write(f"Test proteins: {len(results)}\n")
        f.write(f"Average Spearman: {avg_spearman:.4f}\n")
        f.write(f"Median Spearman: {median_spearman:.4f}\n")
        f.write(f"Std Spearman: {std_spearman:.4f}\n")
        f.write(f"Baseline: 0.4264\n")
        f.write(f"SOTA: 0.62\n")
        f.write(f"Improvement vs Baseline: +{((avg_spearman - 0.4264) / 0.4264 * 100):.1f}%\n")
        f.write(f"Improvement vs SOTA: +{((avg_spearman - 0.62) / 0.62 * 100):.1f}%\n")

    # Top and bottom performers
    results_df_sorted = results_df.sort_values('spearman', ascending=False)
    logger.info("\nTop 10 performers:")
    for _, row in results_df_sorted.head(10).iterrows():
        logger.info(f"  {row['protein']}: {row['spearman']:.4f}")

    logger.info("\nBottom 5 performers:")
    for _, row in results_df_sorted.tail(5).iterrows():
        logger.info(f"  {row['protein']}: {row['spearman']:.4f}")


if __name__ == '__main__':
    main()
