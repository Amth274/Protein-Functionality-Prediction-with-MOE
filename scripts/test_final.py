#!/usr/bin/env python3
"""Run test evaluation using saved model with correct architecture."""
import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.stats import spearmanr

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from src.models.multimodal_fusion import MultiModalMoEModel
from src.data.msa_utils import simple_msa_features

def compute_spearman(predictions, targets):
    if len(predictions) < 2:
        return 0.0
    corr, _ = spearmanr(predictions, targets)
    return corr if not np.isnan(corr) else 0.0

def test_task(model, data_file, tokenizer, device, max_length=1024):
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
    
    # Load model using the exact same class
    model_path = 'outputs/msa_optimized_full/final_model.pt'
    print(f"Loading model from {model_path}")
    
    # Create model with same architecture
    model = MultiModalMoEModel(
        esm2_model_name='facebook/esm2_t12_35M_UR50D',
        num_experts=8,
        expert_hidden_dim=256,
        use_msa=True,
        msa_dim=20,
        dropout=0.1
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")
    
    tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')
    
    # Get test files
    test_dir = Path('data/raw/Test_split')
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
