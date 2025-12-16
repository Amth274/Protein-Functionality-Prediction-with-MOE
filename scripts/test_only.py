#!/usr/bin/env python3
"""Run test evaluation only using saved model."""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class MSAEncoder(nn.Module):
    def __init__(self, msa_dim=20, hidden_dim=320):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(msa_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
    def forward(self, x):
        return self.encoder(x)

class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim=320, num_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
    def forward(self, seq_features, msa_features):
        attn_out, _ = self.cross_attn(seq_features, msa_features, msa_features)
        return self.norm(seq_features + attn_out)

class ExpertMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class MultiModalMoEModel(nn.Module):
    def __init__(self, esm2_model_name, num_experts=8, expert_hidden_dim=256, use_msa=True, msa_dim=20, dropout=0.1):
        super().__init__()
        self.esm2_encoder = AutoModel.from_pretrained(esm2_model_name)
        self.hidden_dim = self.esm2_encoder.config.hidden_size
        self.use_msa = use_msa
        if use_msa:
            self.msa_encoder = MSAEncoder(msa_dim=msa_dim, hidden_dim=self.hidden_dim)
            self.fusion = CrossAttentionFusion(hidden_dim=self.hidden_dim)
        self.experts = nn.ModuleList([
            ExpertMLP(self.hidden_dim, expert_hidden_dim, 1, dropout)
            for _ in range(num_experts)
        ])
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, input_ids, attention_mask, msa_features=None):
        outputs = self.esm2_encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq_features = outputs.last_hidden_state
        if self.use_msa and msa_features is not None:
            msa_encoded = self.msa_encoder(msa_features)
            seq_features = self.fusion(seq_features, msa_encoded)
        pooled = seq_features.mean(dim=1)
        gate_weights = self.gate(pooled)
        expert_outputs = torch.stack([expert(pooled) for expert in self.experts], dim=1)
        output = (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
        return output.squeeze(-1)

def simple_msa_features(sequence, num_positions=20):
    aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    seq_len = len(sequence)
    pssm = np.zeros((seq_len, 20))
    for pos, aa in enumerate(sequence):
        if aa in aa_to_idx:
            idx = aa_to_idx[aa]
            pssm[pos, idx] = 0.7
            noise = np.random.random(20) * 0.1
            pssm[pos] += noise
            pssm[pos] /= pssm[pos].sum()
        else:
            pssm[pos] = np.ones(20) / 20
    return pssm

def compute_spearman(predictions, targets):
    if len(predictions) < 2:
        return 0.0
    corr, _ = spearmanr(predictions, targets)
    return corr if not np.isnan(corr) else 0.0

def test_task(model, data_file, tokenizer, device, max_length=512):
    df = pd.read_csv(data_file)
    if 'mutant' not in df.columns or 'DMS_score' not in df.columns:
        return None, None
    
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for _, row in df.iterrows():
            seq = row.get('mutated_sequence', row.get('sequence', ''))
            if not seq or len(seq) > max_length:
                continue
            
            inputs = tokenizer(seq, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            msa_feat = simple_msa_features(seq)
            msa_tensor = torch.tensor(msa_feat, dtype=torch.float32).unsqueeze(0).to(device)
            
            pred = model(input_ids, attention_mask, msa_tensor)
            predictions.append(pred.item())
            targets.append(row['DMS_score'])
    
    if len(predictions) < 10:
        return None, None
    
    spearman = compute_spearman(predictions, targets)
    return spearman, len(predictions)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model_path = 'outputs/msa_optimized_full/final_model.pt'
    print(f"Loading model from {model_path}")
    
    model = MultiModalMoEModel(
        esm2_model_name='facebook/esm2_t12_35M_UR50D',
        num_experts=8,
        expert_hidden_dim=256,
        use_msa=True,
        msa_dim=20,
        dropout=0.1
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")
    
    tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')
    
    # Get test files
    test_dir = Path('data/ProteinGym/test')
    test_files = sorted(test_dir.glob('*.csv'))
    print(f"Found {len(test_files)} test files")
    
    # Run test
    results = []
    for test_file in tqdm(test_files, desc="Testing"):
        spearman, n_samples = test_task(model, test_file, tokenizer, device)
        if spearman is not None:
            results.append({
                'protein': test_file.stem,
                'spearman': spearman,
                'n_samples': n_samples
            })
            print(f"  {test_file.name}: Spearman={spearman:.4f} (n={n_samples})")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('outputs/msa_optimized_full/test_results.csv', index=False)
    
    # Summary
    avg_spearman = results_df['spearman'].mean()
    median_spearman = results_df['spearman'].median()
    
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    print(f"Test proteins evaluated: {len(results)}")
    print(f"Average Spearman: {avg_spearman:.4f}")
    print(f"Median Spearman: {median_spearman:.4f}")
    print(f"Baseline: 0.4264")
    print(f"Improvement: +{((avg_spearman - 0.4264) / 0.4264 * 100):.1f}%")
    print("="*60)
    
    # Save summary
    with open('outputs/msa_optimized_full/summary.txt', 'w') as f:
        f.write("MSA-Optimized Model Test Results\n")
        f.write("="*40 + "\n")
        f.write(f"Test proteins: {len(results)}\n")
        f.write(f"Average Spearman: {avg_spearman:.4f}\n")
        f.write(f"Median Spearman: {median_spearman:.4f}\n")
        f.write(f"Baseline: 0.4264\n")
        f.write(f"Improvement: +{((avg_spearman - 0.4264) / 0.4264 * 100):.1f}%\n")

if __name__ == '__main__':
    main()
