#!/usr/bin/env python3
"""
Fast test evaluation for ESM2-650M - No model reloading per protein.
Direct inference without per-protein fine-tuning (zero-shot style).
"""

import os
import sys
import torch
import torch.nn as nn
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
        logging.FileHandler('logs/test_esm650m_fast.log')
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


def evaluate_protein(model, data_path, tokenizer, device, batch_size=8, max_len=1024):
    """Fast evaluation - direct inference without fine-tuning."""
    df = pd.read_csv(data_path)

    sequences = df['mutated_sequence'].astype(str).tolist()
    targets = df['DMS_score'].astype(float).tolist()

    # Filter by length
    filtered = [(s, t) for s, t in zip(sequences, targets) if len(s) <= max_len]
    if len(filtered) < 10:
        return None

    sequences, targets = zip(*filtered)
    sequences = list(sequences)
    targets = list(targets)

    model.eval()
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]

            encoding = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len
            )

            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            with autocast():
                preds = model(input_ids, attention_mask)

            all_preds.extend(preds.cpu().numpy())

    spearman = spearmanr(all_preds, targets)[0]
    return float(spearman) if not np.isnan(spearman) else 0.0


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load model ONCE
    model_path = 'outputs/esm650m/final_model.pt'
    logger.info(f"Loading model from {model_path}")

    model = ESM650MModel(model_name='facebook/esm2_t33_650M_UR50D', dropout=0.1).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("Model loaded successfully")

    tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')

    # Test files
    test_dir = Path('data/raw/Test_split')
    test_files = sorted(test_dir.glob('*.csv'))
    logger.info(f"Found {len(test_files)} test files")

    logger.info("\n" + "="*60)
    logger.info("FAST TEST EVALUATION (Direct Inference)")
    logger.info("="*60)

    results = []
    for test_file in tqdm(test_files, desc="Testing"):
        spearman = evaluate_protein(model, test_file, tokenizer, device, batch_size=8)
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
    logger.info("FINAL TEST RESULTS - ESM2-650M (Direct Inference)")
    logger.info("="*60)
    logger.info(f"Test proteins: {len(results)}")
    logger.info(f"Average Spearman: {avg_spearman:.4f}")
    logger.info(f"Median Spearman: {median_spearman:.4f}")
    logger.info(f"Std Spearman: {std_spearman:.4f}")
    logger.info(f"Baseline: 0.4264")
    logger.info(f"SOTA (SaProt+TTT): 0.62")
    logger.info(f"Improvement vs Baseline: +{((avg_spearman - 0.4264) / 0.4264 * 100):.1f}%")
    logger.info("="*60)

    # Save results
    results_df.to_csv('outputs/esm650m/test_results_fast.csv', index=False)

    # Top performers
    results_df_sorted = results_df.sort_values('spearman', ascending=False)
    logger.info("\nTop 10 performers:")
    for _, row in results_df_sorted.head(10).iterrows():
        logger.info(f"  {row['protein']}: {row['spearman']:.4f}")


if __name__ == '__main__':
    main()
