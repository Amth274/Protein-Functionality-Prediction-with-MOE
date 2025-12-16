#!/usr/bin/env python3
"""
Simplified EDA for Protein Functionality Prediction
This version works without complex imports and focuses on basic data analysis.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
from tqdm import tqdm
from pathlib import Path


def load_dms_data_simple(file_path):
    """Simple function to load and validate DMS data."""
    df = pd.read_csv(file_path)
    required_columns = ['mutated_sequence', 'DMS_score']

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found")

    return df.dropna(subset=required_columns)


def get_protein_statistics_simple(df):
    """Get basic statistics for a protein dataset."""
    return {
        'num_sequences': len(df),
        'avg_sequence_length': df['mutated_sequence'].str.len().mean(),
        'min_sequence_length': df['mutated_sequence'].str.len().min(),
        'max_sequence_length': df['mutated_sequence'].str.len().max(),
        'avg_score': df['DMS_score'].mean(),
        'std_score': df['DMS_score'].std(),
        'min_score': df['DMS_score'].min(),
        'max_score': df['DMS_score'].max(),
    }


def analyze_dataset_overview(data_dir, split_name):
    """Analyze overview statistics for a dataset split."""
    print(f"Analyzing {split_name} dataset...")

    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files in {split_name}")

    overview_stats = {
        'num_files': len(csv_files),
        'total_sequences': 0,
        'valid_files': 0,
        'file_stats': []
    }

    for csv_file in tqdm(csv_files, desc=f"Processing {split_name} files"):
        file_path = os.path.join(data_dir, csv_file)
        try:
            df = load_dms_data_simple(file_path)
            stats = get_protein_statistics_simple(df)
            stats['file_name'] = csv_file
            stats['protein_id'] = csv_file.split('_')[0]

            overview_stats['file_stats'].append(stats)
            overview_stats['total_sequences'] += stats['num_sequences']
            overview_stats['valid_files'] += 1

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

    return overview_stats


def create_summary_plots(train_stats, test_stats, output_dir):
    """Create summary visualization plots."""
    plt.style.use('default')

    # Dataset overview
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Number of sequences per dataset
    train_sizes = [stats['num_sequences'] for stats in train_stats]
    test_sizes = [stats['num_sequences'] for stats in test_stats]

    axes[0, 0].hist(train_sizes, bins=30, alpha=0.7, label='Train', density=True)
    axes[0, 0].hist(test_sizes, bins=30, alpha=0.7, label='Test', density=True)
    axes[0, 0].axvline(200, color='red', linestyle='--', label='Meta-learning threshold')
    axes[0, 0].set_xlabel('Number of Sequences')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Dataset Size Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Sequence length distribution
    train_lengths = [stats['avg_sequence_length'] for stats in train_stats]
    test_lengths = [stats['avg_sequence_length'] for stats in test_stats]

    axes[0, 1].hist(train_lengths, bins=30, alpha=0.7, label='Train', density=True)
    axes[0, 1].hist(test_lengths, bins=30, alpha=0.7, label='Test', density=True)
    axes[0, 1].set_xlabel('Average Sequence Length')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Sequence Length Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Score distribution
    train_scores = [stats['avg_score'] for stats in train_stats]
    test_scores = [stats['avg_score'] for stats in test_stats]

    axes[0, 2].hist(train_scores, bins=30, alpha=0.7, label='Train', density=True)
    axes[0, 2].hist(test_scores, bins=30, alpha=0.7, label='Test', density=True)
    axes[0, 2].set_xlabel('Average DMS Score')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].set_title('Score Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Score variability
    train_std = [stats['std_score'] for stats in train_stats]
    test_std = [stats['std_score'] for stats in test_stats]

    axes[1, 0].hist(train_std, bins=30, alpha=0.7, label='Train', density=True)
    axes[1, 0].hist(test_std, bins=30, alpha=0.7, label='Test', density=True)
    axes[1, 0].set_xlabel('Score Standard Deviation')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Score Variability')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Meta-learning suitability
    min_required = 200
    train_suitable = len([s for s in train_stats if s['num_sequences'] >= min_required])
    test_suitable = len([s for s in test_stats if s['num_sequences'] >= min_required])

    categories = ['Train\nSuitable', 'Train\nToo Small', 'Test\nSuitable', 'Test\nToo Small']
    values = [train_suitable, len(train_stats) - train_suitable,
              test_suitable, len(test_stats) - test_suitable]
    colors = ['green', 'red', 'lightgreen', 'lightcoral']

    axes[1, 1].bar(categories, values, color=colors)
    axes[1, 1].set_ylabel('Number of Datasets')
    axes[1, 1].set_title('Meta-Learning Suitability')
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Protein family distribution (top 15)
    protein_types = defaultdict(int)
    for stats in train_stats + test_stats:
        protein_type = stats['protein_id'][:4]  # First 4 characters
        protein_types[protein_type] += 1

    top_types = dict(sorted(protein_types.items(), key=lambda x: x[1], reverse=True)[:15])
    axes[1, 2].bar(range(len(top_types)), list(top_types.values()))
    axes[1, 2].set_xticks(range(len(top_types)))
    axes[1, 2].set_xticklabels(list(top_types.keys()), rotation=45)
    axes[1, 2].set_xlabel('Protein Type (First 4 chars)')
    axes[1, 2].set_ylabel('Number of Datasets')
    axes[1, 2].set_title('Top Protein Types')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'eda_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    return train_suitable, test_suitable


def generate_eda_report(train_overview, test_overview, output_dir):
    """Generate comprehensive EDA report."""
    print("Generating EDA report...")

    train_stats = train_overview['file_stats']
    test_stats = test_overview['file_stats']

    # Basic statistics
    report = {
        'dataset_overview': {
            'train_files': train_overview['num_files'],
            'test_files': test_overview['num_files'],
            'train_valid_files': train_overview['valid_files'],
            'test_valid_files': test_overview['valid_files'],
            'total_train_sequences': train_overview['total_sequences'],
            'total_test_sequences': test_overview['total_sequences']
        },
        'sequence_statistics': {
            'train_avg_length': np.mean([s['avg_sequence_length'] for s in train_stats]),
            'test_avg_length': np.mean([s['avg_sequence_length'] for s in test_stats]),
            'train_length_std': np.std([s['avg_sequence_length'] for s in train_stats]),
            'test_length_std': np.std([s['avg_sequence_length'] for s in test_stats]),
            'overall_min_length': min([s['min_sequence_length'] for s in train_stats + test_stats]),
            'overall_max_length': max([s['max_sequence_length'] for s in train_stats + test_stats])
        },
        'score_statistics': {
            'train_avg_score': np.mean([s['avg_score'] for s in train_stats]),
            'test_avg_score': np.mean([s['avg_score'] for s in test_stats]),
            'train_score_std': np.std([s['avg_score'] for s in train_stats]),
            'test_score_std': np.std([s['avg_score'] for s in test_stats]),
            'overall_min_score': min([s['min_score'] for s in train_stats + test_stats]),
            'overall_max_score': max([s['max_score'] for s in train_stats + test_stats])
        }
    }

    # Meta-learning analysis
    min_required = 200
    train_suitable = len([s for s in train_stats if s['num_sequences'] >= min_required])
    test_suitable = len([s for s in test_stats if s['num_sequences'] >= min_required])

    report['meta_learning'] = {
        'train_suitable_tasks': train_suitable,
        'test_suitable_tasks': test_suitable,
        'train_unsuitable_tasks': len(train_stats) - train_suitable,
        'test_unsuitable_tasks': len(test_stats) - test_suitable,
        'meta_learning_threshold': min_required,
        'suitability_percentage': (train_suitable + test_suitable) / (len(train_stats) + len(test_stats)) * 100
    }

    # Save detailed file statistics
    detailed_stats = {
        'train_files': train_stats,
        'test_files': test_stats
    }

    # Save reports
    with open(output_dir / 'eda_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    with open(output_dir / 'detailed_stats.json', 'w') as f:
        json.dump(detailed_stats, f, indent=2, default=str)

    return report


def main():
    """Main EDA function."""
    # Setup
    data_dir = 'data/raw'
    output_dir = Path('outputs/eda')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== PROTEIN FUNCTIONALITY EDA ===")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Check directories
    train_dir = os.path.join(data_dir, 'Train_split')
    test_dir = os.path.join(data_dir, 'Test_split')

    if not os.path.exists(train_dir):
        print(f"ERROR: Training directory not found: {train_dir}")
        return

    if not os.path.exists(test_dir):
        print(f"ERROR: Test directory not found: {test_dir}")
        return

    # Analyze datasets
    train_overview = analyze_dataset_overview(train_dir, 'train')
    test_overview = analyze_dataset_overview(test_dir, 'test')

    # Create plots
    print("Creating visualization plots...")
    train_suitable, test_suitable = create_summary_plots(
        train_overview['file_stats'],
        test_overview['file_stats'],
        output_dir
    )

    # Generate report
    report = generate_eda_report(train_overview, test_overview, output_dir)

    # Print summary
    print("\n=== EDA SUMMARY ===")
    print(f"Training files: {train_overview['valid_files']}/{train_overview['num_files']}")
    print(f"Test files: {test_overview['valid_files']}/{test_overview['num_files']}")
    print(f"Total training sequences: {train_overview['total_sequences']:,}")
    print(f"Total test sequences: {test_overview['total_sequences']:,}")
    print(f"Meta-learning suitable tasks: {train_suitable} train, {test_suitable} test")
    print(f"Average sequence length: {report['sequence_statistics']['train_avg_length']:.1f} (train), {report['sequence_statistics']['test_avg_length']:.1f} (test)")
    print(f"Average DMS scores: {report['score_statistics']['train_avg_score']:.3f} (train), {report['score_statistics']['test_avg_score']:.3f} (test)")
    print(f"Suitability for meta-learning: {report['meta_learning']['suitability_percentage']:.1f}%")
    print(f"\nEDA complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()