#!/usr/bin/env python3
"""
ESM2-650M Training Script - Target 0.60-0.62 Spearman

Upgrades from ESM2-35M (480 dims) to ESM2-650M (1280 dims)
with simplified head based on SOTA research findings.

Key changes:
1. ESM2-650M backbone (1280 dim embeddings)
2. Simplified prediction head (research shows complex heads don't help)
3. Lower learning rate for larger model (1e-6)
4. Gradient checkpointing for memory efficiency
"""

import os
import sys
import argparse
import logging
import random
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/train_esm650m.log')
    ]
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ESM650MModel(nn.Module):
    """
    ESM2-650M with simplified prediction head.

    Research shows simpler heads often outperform complex MoE architectures.
    Using: ESM2-650M (1280 dims) -> LayerNorm -> Linear -> ReLU -> Linear -> 1
    """

    def __init__(
        self,
        model_name='facebook/esm2_t33_650M_UR50D',
        dropout=0.1,
        use_gradient_checkpointing=True
    ):
        super().__init__()

        # Load ESM2-650M
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.encoder.config.hidden_size  # 1280

        # Enable gradient checkpointing for memory efficiency
        if use_gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        # Simplified prediction head (research-backed)
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),  # 1280 -> 320
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 4, 1)  # 320 -> 1
        )

        # Initialize head weights
        self._init_weights()

    def _init_weights(self):
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None):
        # Get ESM2 embeddings
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, 1280)

        # Mean pooling with attention mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden_states.mean(dim=1)

        # Prediction
        output = self.head(pooled)
        return output.squeeze(-1)


class ProteinDataset(Dataset):
    def __init__(self, path, max_len=1024):
        df = pd.read_csv(path)
        self.sequences = df['mutated_sequence'].astype(str).tolist()
        self.fitness = df['DMS_score'].astype(float).tolist()

        # Filter by length
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

    encoding = tokenizer(
        list(sequences),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len
    )

    return encoding["input_ids"], encoding["attention_mask"], torch.stack(fitness)


def train_task(model, task_path, tokenizer, device, config, optimizer, scaler):
    """Meta-learning style training on one protein task."""
    dataset = ProteinDataset(task_path, max_len=config['max_len'])

    if len(dataset) < 10:
        return None

    # 80/20 split
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

    # Training on support set
    model.train()
    accum_steps = config.get('gradient_accumulation_steps', 8)

    optimizer.zero_grad()
    step = 0

    for input_ids, attention_mask, targets in support_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        with autocast():
            predictions = model(input_ids, attention_mask)
            loss = F.mse_loss(predictions, targets)
            loss = loss / accum_steps

        scaler.scale(loss).backward()
        step += 1

        if step % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    # Final gradient step
    if step % accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    # Evaluation on query set
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='data/raw/Train_split')
    parser.add_argument('--test_dir', type=str, default='data/raw/Test_split')
    parser.add_argument('--output_dir', type=str, default='outputs/esm650m')
    parser.add_argument('--batch_size', type=int, default=2)  # Smaller for 650M
    parser.add_argument('--lr', type=float, default=1e-6)  # Lower for large model
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--max_len', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Initialize model
    logger.info("Loading ESM2-650M model...")
    model = ESM650MModel(
        model_name='facebook/esm2_t33_650M_UR50D',
        dropout=0.1,
        use_gradient_checkpointing=True
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Embedding dimension: {model.hidden_dim}")

    tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')

    # Optimizer with lower LR for pretrained model
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scaler = GradScaler()

    config = {
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'grad_clip': args.grad_clip,
        'max_len': args.max_len,
        'gradient_accumulation_steps': args.gradient_accumulation_steps
    }

    logger.info(f"Config: {config}")

    # Get training files
    train_dir = Path(args.train_dir)
    train_files = sorted(train_dir.glob('*.csv'))
    logger.info(f"Found {len(train_files)} training files")

    # Training loop
    results = []
    for i, train_file in enumerate(tqdm(train_files, desc="Training")):
        spearman = train_task(model, train_file, tokenizer, device, config, optimizer, scaler)

        if spearman is not None:
            results.append({
                'protein': train_file.stem,
                'spearman': spearman
            })

        # Checkpoint every 50 tasks
        if (i + 1) % 50 == 0:
            avg_spearman = np.mean([r['spearman'] for r in results])
            logger.info(f"\nProgress: {i+1}/{len(train_files)}")
            logger.info(f"Train avg Spearman: {avg_spearman:.4f}")

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'results': results
            }, f"{args.output_dir}/checkpoint_{i+1}.pt")

    # Save final model
    final_spearman = np.mean([r['spearman'] for r in results])
    logger.info(f"\nTraining complete: {len(results)} proteins")
    logger.info(f"Train avg Spearman: {final_spearman:.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'train_results': results
    }, f"{args.output_dir}/final_model.pt")

    pd.DataFrame(results).to_csv(f"{args.output_dir}/train_results.csv", index=False)

    # Meta-testing on held-out test set
    logger.info("\n" + "="*60)
    logger.info("META-TESTING ON HELD-OUT TEST SET")
    logger.info("="*60)

    test_dir = Path(args.test_dir)
    test_files = sorted(test_dir.glob('*.csv'))
    logger.info(f"Found {len(test_files)} test tasks")

    test_results = []
    for test_file in tqdm(test_files, desc="Meta-testing"):
        spearman = train_task(model, test_file, tokenizer, device, config, optimizer, scaler)
        if spearman is not None:
            test_results.append({
                'protein': test_file.stem,
                'spearman': spearman
            })
            logger.info(f"  {test_file.name}: Spearman={spearman:.4f}")

    test_avg = np.mean([r['spearman'] for r in test_results])
    test_median = np.median([r['spearman'] for r in test_results])

    logger.info(f"\n" + "="*60)
    logger.info("FINAL RESULTS")
    logger.info("="*60)
    logger.info(f"Test proteins: {len(test_results)}")
    logger.info(f"Test Avg Spearman: {test_avg:.4f}")
    logger.info(f"Test Median Spearman: {test_median:.4f}")
    logger.info(f"Baseline: 0.4264")
    logger.info(f"Improvement: +{((test_avg - 0.4264) / 0.4264 * 100):.1f}%")
    logger.info("="*60)

    pd.DataFrame(test_results).to_csv(f"{args.output_dir}/test_results.csv", index=False)

    # Summary
    with open(f"{args.output_dir}/summary.txt", 'w') as f:
        f.write("ESM2-650M Model Results\n")
        f.write("="*40 + "\n")
        f.write(f"Model: ESM2-650M (1280 dim embeddings)\n")
        f.write(f"Train proteins: {len(results)}\n")
        f.write(f"Train Avg Spearman: {final_spearman:.4f}\n")
        f.write(f"Test proteins: {len(test_results)}\n")
        f.write(f"Test Avg Spearman: {test_avg:.4f}\n")
        f.write(f"Test Median Spearman: {test_median:.4f}\n")
        f.write(f"Baseline: 0.4264\n")
        f.write(f"Improvement: +{((test_avg - 0.4264) / 0.4264 * 100):.1f}%\n")


if __name__ == '__main__':
    main()
