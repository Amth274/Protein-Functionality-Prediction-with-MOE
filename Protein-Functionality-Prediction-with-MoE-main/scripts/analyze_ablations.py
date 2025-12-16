#!/usr/bin/env python3
"""
Analyze ablation study results and generate comprehensive report.

Usage:
    python scripts/analyze_ablations.py --output reports/ablation_report.md
"""

import argparse
import yaml
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime


def load_experiment_results(exp_id):
    """Load results from an experiment."""
    results_path = Path(f'outputs/ablations/{exp_id}/test_results.yaml')

    if not results_path.exists():
        return None

    with open(results_path) as f:
        results = yaml.safe_load(f)

    # Extract metrics
    metrics = {
        'experiment_id': exp_id,
        'spearman': float(results['avg_metrics']['test_avg_spearman']),
        'pearson': float(results['avg_metrics']['test_avg_pearson']),
        'mse': float(results['avg_metrics']['test_avg_mse']),
    }

    # Calculate per-protein statistics
    task_metrics = results['task_metrics']
    spearmans = [t['spearman'] for t in task_metrics]
    metrics['spearman_std'] = float(np.std(spearmans))
    metrics['spearman_median'] = float(np.median(spearmans))
    metrics['spearman_min'] = float(np.min(spearmans))
    metrics['spearman_max'] = float(np.max(spearmans))

    # Count proteins above thresholds
    metrics['proteins_above_0.6'] = sum(1 for s in spearmans if s > 0.6)
    metrics['proteins_above_0.7'] = sum(1 for s in spearmans if s > 0.7)
    metrics['proteins_above_0.8'] = sum(1 for s in spearmans if s > 0.8)

    return metrics


def load_experiment_config(exp_id):
    """Load configuration for an experiment."""
    config_path = Path(f'configs/ablations/{exp_id}.yaml')

    if not config_path.exists():
        return None

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return {
        'id': exp_id,
        'description': config.get('experiment', {}).get('description', 'N/A'),
        'num_experts': config.get('model', {}).get('num_experts'),
        'hidden_dim': config.get('model', {}).get('hidden_dim'),
        'esm2_model': config.get('model', {}).get('esm2_model'),
        'freeze_backbone': config.get('model', {}).get('freeze_backbone'),
        'dropout': config.get('model', {}).get('dropout'),
        'support_frac': config.get('data', {}).get('support_frac'),
        'task_epochs': config.get('training', {}).get('task_epochs'),
        'weight_decay': config.get('training', {}).get('weight_decay'),
    }


def analyze_ablations():
    """Analyze all completed ablation experiments."""

    # Load experiment index
    index_path = Path('configs/ablations/index.yaml')
    with open(index_path) as f:
        index = yaml.safe_load(f)

    experiments = index['experiments']

    # Collect results
    all_results = []
    all_configs = []

    print("Loading experiment results...\n")

    for exp in experiments:
        exp_id = exp['id']

        # Load results
        results = load_experiment_results(exp_id)
        config = load_experiment_config(exp_id)

        if results and config:
            all_results.append(results)
            all_configs.append(config)
            print(f"✓ {exp_id}: Spearman = {results['spearman']:.4f}")
        else:
            print(f"⚠ {exp_id}: No results found")

    if not all_results:
        print("\nNo results to analyze!")
        return None

    # Create DataFrames
    df_results = pd.DataFrame(all_results)
    df_configs = pd.DataFrame(all_configs)

    # Merge
    df = pd.merge(df_configs, df_results, left_on='id', right_on='experiment_id')

    # Sort by spearman
    df = df.sort_values('spearman', ascending=False)

    return df


def generate_report(df, output_path):
    """Generate comprehensive markdown report."""

    baseline_spearman = 0.4264  # From A1.1
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    report = f"""# Ablation Study Results
**Generated**: {timestamp}
**Baseline**: A1.1 (Spearman: {baseline_spearman:.4f})
**Experiments Completed**: {len(df)}

---

## Executive Summary

### Top 5 Configurations

| Rank | Exp ID | Spearman | Δ from Baseline | Description |
|------|--------|----------|-----------------|-------------|
"""

    for i, row in df.head(5).iterrows():
        delta = row['spearman'] - baseline_spearman
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        report += f"| {i+1} | {row['id']} | {row['spearman']:.4f} | {delta_str} | {row['description']} |\n"

    report += f"""

### Bottom 5 Configurations

| Rank | Exp ID | Spearman | Δ from Baseline | Description |
|------|--------|----------|-----------------|-------------|
"""

    for i, row in df.tail(5).iterrows():
        delta = row['spearman'] - baseline_spearman
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        rank = len(df) - list(df.tail(5).index).index(i)
        report += f"| {rank} | {row['id']} | {row['spearman']:.4f} | {delta_str} | {row['description']} |\n"

    report += f"""

### Key Findings

- **Best Performance**: {df.iloc[0]['id']} with Spearman = {df.iloc[0]['spearman']:.4f}
- **Worst Performance**: {df.iloc[-1]['id']} with Spearman = {df.iloc[-1]['spearman']:.4f}
- **Performance Range**: {df['spearman'].min():.4f} to {df['spearman'].max():.4f}
- **Average Performance**: {df['spearman'].mean():.4f} ± {df['spearman'].std():.4f}

---

## Study 1: Architecture Ablations (MoE Components)

"""

    # Filter A1 experiments
    a1_df = df[df['id'].str.startswith('A1')].copy()
    if len(a1_df) > 0:
        report += "### Effect of Number of Experts\n\n"
        report += "| Exp ID | Num Experts | Spearman | Δ from Baseline |\n"
        report += "|--------|-------------|----------|------------------|\n"

        for _, row in a1_df.iterrows():
            delta = row['spearman'] - baseline_spearman
            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            report += f"| {row['id']} | {row['num_experts']} | {row['spearman']:.4f} | {delta_str} |\n"

        report += f"\n**Finding**: "
        if len(a1_df) >= 2:
            best_a1 = a1_df.iloc[0]
            report += f"Optimal number of experts appears to be {best_a1['num_experts']} ({best_a1['id']}). "
    else:
        report += "*No A1 experiments completed yet.*\n"

    report += "\n---\n\n## Study 2: Expert Capacity Ablations\n\n"

    # Filter A2 experiments
    a2_df = df[df['id'].str.startswith('A2')].copy()
    if len(a2_df) > 0:
        report += "### Effect of Expert Hidden Dimension\n\n"
        report += "| Exp ID | Hidden Dim | Spearman | Δ from Baseline |\n"
        report += "|--------|------------|----------|------------------|\n"

        for _, row in a2_df.iterrows():
            delta = row['spearman'] - baseline_spearman
            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            report += f"| {row['id']} | {row['hidden_dim']} | {row['spearman']:.4f} | {delta_str} |\n"

        best_a2 = a2_df.iloc[0]
        report += f"\n**Finding**: Optimal hidden dimension appears to be {best_a2['hidden_dim']} ({best_a2['id']}). "
    else:
        report += "*No A2 experiments completed yet.*\n"

    report += "\n---\n\n## Study 3: Backbone Model Ablations\n\n"

    # Filter A3 experiments
    a3_df = df[df['id'].str.startswith('A3')].copy()
    if len(a3_df) > 0:
        report += "### Effect of ESM2 Model Size\n\n"
        report += "| Exp ID | ESM2 Model | Frozen | Spearman | Δ from Baseline |\n"
        report += "|--------|------------|--------|----------|------------------|\n"

        for _, row in a3_df.iterrows():
            delta = row['spearman'] - baseline_spearman
            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            frozen = "Yes" if row['freeze_backbone'] else "No"
            model_name = row['esm2_model'].split('/')[-1].replace('esm2_', '')
            report += f"| {row['id']} | {model_name} | {frozen} | {row['spearman']:.4f} | {delta_str} |\n"

        best_a3 = a3_df.iloc[0]
        report += f"\n**Finding**: Best backbone configuration is {best_a3['id']}. "
    else:
        report += "*No A3 experiments completed yet.*\n"

    report += "\n---\n\n## Study 4: Training Strategy Ablations\n\n"

    # Filter A4 experiments
    a4_df = df[df['id'].str.startswith('A4')].copy()
    if len(a4_df) > 0:
        report += "### Effect of Support/Query Split\n\n"
        report += "| Exp ID | Support % | Epochs | Spearman | Δ from Baseline |\n"
        report += "|--------|-----------|--------|----------|------------------|\n"

        for _, row in a4_df.iterrows():
            delta = row['spearman'] - baseline_spearman
            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            support_pct = int(row['support_frac'] * 100)
            report += f"| {row['id']} | {support_pct}% | {row['task_epochs']} | {row['spearman']:.4f} | {delta_str} |\n"

        best_a4 = a4_df.iloc[0]
        report += f"\n**Finding**: Optimal training configuration is {best_a4['id']}. "
    else:
        report += "*No A4 experiments completed yet.*\n"

    report += "\n---\n\n## Study 6: Regularization Ablations\n\n"

    # Filter A6 experiments
    a6_df = df[df['id'].str.startswith('A6')].copy()
    if len(a6_df) > 0:
        report += "### Effect of Regularization\n\n"
        report += "| Exp ID | Dropout | Weight Decay | Spearman | Δ from Baseline |\n"
        report += "|--------|---------|--------------|----------|------------------|\n"

        for _, row in a6_df.iterrows():
            delta = row['spearman'] - baseline_spearman
            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            report += f"| {row['id']} | {row['dropout']} | {row['weight_decay']} | {row['spearman']:.4f} | {delta_str} |\n"

        best_a6 = a6_df.iloc[0]
        report += f"\n**Finding**: Best regularization is {best_a6['id']}. "
    else:
        report += "*No A6 experiments completed yet.*\n"

    report += "\n---\n\n## Study 7: Data Efficiency\n\n"

    # Filter A7 experiments
    a7_df = df[df['id'].str.startswith('A7')].copy()
    if len(a7_df) > 0:
        report += "### Effect of Training Data Size\n\n"
        report += "| Exp ID | Training % | Spearman | Δ from Baseline |\n"
        report += "|--------|------------|----------|------------------|\n"

        for _, row in a7_df.iterrows():
            delta = row['spearman'] - baseline_spearman
            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            # Extract train fraction from experiment description
            desc = row['description']
            pct = desc.split('%')[0].split()[-1] if '%' in desc else "?"
            report += f"| {row['id']} | {pct}% | {row['spearman']:.4f} | {delta_str} |\n"

        report += f"\n**Finding**: Performance scales with training data size. "
    else:
        report += "*No A7 experiments completed yet.*\n"

    report += "\n---\n\n## All Results (Full Table)\n\n"
    report += "| Rank | Exp ID | Spearman | Pearson | MSE | Description |\n"
    report += "|------|--------|----------|---------|-----|-------------|\n"

    for rank, (_, row) in enumerate(df.iterrows(), 1):
        report += f"| {rank} | {row['id']} | {row['spearman']:.4f} | {row['pearson']:.4f} | {row['mse']:.2f} | {row['description']} |\n"

    report += f"""

---

## Recommendations

Based on the ablation study results:

### Optimal Configuration

The best-performing configuration combines:
- **Experiment**: {df.iloc[0]['id']}
- **Performance**: Spearman = {df.iloc[0]['spearman']:.4f} ({((df.iloc[0]['spearman'] - baseline_spearman) / baseline_spearman * 100):.1f}% improvement)
- **Description**: {df.iloc[0]['description']}

### Key Hyperparameters

"""

    best_row = df.iloc[0]
    report += f"- **Num Experts**: {best_row['num_experts']}\n"
    report += f"- **Hidden Dim**: {best_row['hidden_dim']}\n"
    report += f"- **ESM2 Model**: {best_row['esm2_model']}\n"
    report += f"- **Freeze Backbone**: {best_row['freeze_backbone']}\n"
    report += f"- **Dropout**: {best_row['dropout']}\n"
    report += f"- **Support Fraction**: {best_row['support_frac']}\n"
    report += f"- **Task Epochs**: {best_row['task_epochs']}\n"

    report += f"""

### Next Steps

1. Retrain with optimal configuration for longer
2. Investigate why certain configurations underperformed
3. Consider ensemble methods combining top configurations
4. Explore additional features (MSA, structure) for further gains

---

*Report generated by analyze_ablations.py*
*Timestamp: {timestamp}*
"""

    # Write report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"\n✓ Report saved to: {output_path}")

    # Also save CSV
    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ CSV saved to: {csv_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Analyze ablation study results')
    parser.add_argument('--output', default='reports/ablation_report.md',
                        help='Output path for report (default: reports/ablation_report.md)')

    args = parser.parse_args()

    # Analyze
    df = analyze_ablations()

    if df is not None:
        # Generate report
        generate_report(df, args.output)

        print(f"\n{'='*70}")
        print(f"Analysis complete!")
        print(f"{'='*70}\n")
    else:
        print("No results to analyze. Run some ablation experiments first.")


if __name__ == '__main__':
    main()
