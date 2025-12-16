#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for Protein Functionality Prediction

This script performs comprehensive EDA on the ProteinGym DMS datasets to understand:
- Data distribution and quality
- Sequence characteristics
- Score distributions
- Task complexity analysis

Usage:
    python scripts/run_eda.py --output-dir outputs/eda
"""

import argparse
import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.utils import load_dms_data, get_protein_statistics, clean_sequences


def setup_eda_logging(output_dir):
    """Setup logging for EDA."""
    log_file = output_dir / 'eda.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def analyze_dataset_overview(data_dir, split_name, logger):
    """Analyze overview statistics for a dataset split."""
    logger.info(f"Analyzing {split_name} dataset overview...")

    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    logger.info(f"Found {len(csv_files)} CSV files in {split_name}")

    overview_stats = {
        'num_files': len(csv_files),
        'files': csv_files,
        'total_sequences': 0,
        'total_mutations': 0,
        'valid_files': 0,
        'file_stats': []
    }

    for csv_file in tqdm(csv_files, desc=f"Processing {split_name} files"):
        file_path = os.path.join(data_dir, csv_file)
        try:
            df = load_dms_data(file_path)
            df_clean = clean_sequences(df)

            stats = get_protein_statistics(df_clean)
            stats['file_name'] = csv_file
            stats['protein_id'] = csv_file.split('_')[0]

            overview_stats['file_stats'].append(stats)
            overview_stats['total_sequences'] += stats['num_sequences']
            overview_stats['valid_files'] += 1

            logger.debug(f"{csv_file}: {stats['num_sequences']} sequences")

        except Exception as e:
            logger.error(f"Error processing {csv_file}: {e}")

    return overview_stats


def create_sequence_length_analysis(file_stats, output_dir, split_name):
    """Create sequence length distribution analysis."""
    seq_lengths = []
    protein_ids = []

    for stats in file_stats:
        seq_lengths.extend([stats['avg_sequence_length']] * int(stats['num_sequences']))
        protein_ids.append(stats['protein_id'])

    # Overall distribution
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.hist(seq_lengths, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.title(f'{split_name}: Sequence Length Distribution')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    avg_lengths = [stats['avg_sequence_length'] for stats in file_stats]
    plt.boxplot(avg_lengths)
    plt.ylabel('Average Sequence Length')
    plt.title(f'{split_name}: Avg Length per Protein')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    num_sequences = [stats['num_sequences'] for stats in file_stats]
    plt.hist(num_sequences, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Sequences')
    plt.ylabel('Number of Proteins')
    plt.title(f'{split_name}: Sequences per Protein')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.scatter(avg_lengths, num_sequences, alpha=0.6)
    plt.xlabel('Average Sequence Length')
    plt.ylabel('Number of Sequences')
    plt.title(f'{split_name}: Length vs Count')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{split_name}_sequence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_score_distribution_analysis(file_stats, output_dir, split_name):
    """Create score distribution analysis."""
    plt.figure(figsize=(15, 10))

    # Collect all scores and statistics
    all_avg_scores = [stats['avg_score'] for stats in file_stats]
    all_std_scores = [stats['std_score'] for stats in file_stats]
    all_min_scores = [stats['min_score'] for stats in file_stats]
    all_max_scores = [stats['max_score'] for stats in file_stats]

    plt.subplot(2, 3, 1)
    plt.hist(all_avg_scores, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Average DMS Score')
    plt.ylabel('Number of Proteins')
    plt.title(f'{split_name}: Average Score Distribution')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 2)
    plt.hist(all_std_scores, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Score Standard Deviation')
    plt.ylabel('Number of Proteins')
    plt.title(f'{split_name}: Score Variability')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 3)
    score_ranges = [max_s - min_s for max_s, min_s in zip(all_max_scores, all_min_scores)]
    plt.hist(score_ranges, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Score Range (Max - Min)')
    plt.ylabel('Number of Proteins')
    plt.title(f'{split_name}: Score Range Distribution')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 4)
    plt.scatter(all_avg_scores, all_std_scores, alpha=0.6)
    plt.xlabel('Average Score')
    plt.ylabel('Score Std Dev')
    plt.title(f'{split_name}: Score Mean vs Variability')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 5)
    plt.boxplot([all_min_scores, all_avg_scores, all_max_scores],
                labels=['Min', 'Avg', 'Max'])
    plt.ylabel('DMS Score')
    plt.title(f'{split_name}: Score Statistics')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 6)
    num_sequences = [stats['num_sequences'] for stats in file_stats]
    plt.scatter(score_ranges, num_sequences, alpha=0.6)
    plt.xlabel('Score Range')
    plt.ylabel('Number of Sequences')
    plt.title(f'{split_name}: Range vs Sample Size')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{split_name}_score_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_protein_family_analysis(file_stats, output_dir, split_name):
    """Create protein family analysis based on protein IDs."""
    protein_ids = [stats['protein_id'] for stats in file_stats]

    # Extract organism/family patterns
    organism_patterns = defaultdict(int)
    protein_types = defaultdict(int)

    for protein_id in protein_ids:
        # Extract organism codes (usually after first underscore)
        parts = protein_id.split('_')
        if len(parts) > 1:
            organism_patterns[parts[1][:5]] += 1  # First 5 chars of organism

        # Extract protein type patterns (first part)
        protein_types[parts[0][:4]] += 1  # First 4 chars of protein

    # Create visualizations
    plt.figure(figsize=(15, 10))

    # Top organisms
    plt.subplot(2, 2, 1)
    top_organisms = dict(sorted(organism_patterns.items(), key=lambda x: x[1], reverse=True)[:15])
    plt.bar(range(len(top_organisms)), list(top_organisms.values()))
    plt.xticks(range(len(top_organisms)), list(top_organisms.keys()), rotation=45)
    plt.xlabel('Organism Pattern')
    plt.ylabel('Number of Proteins')
    plt.title(f'{split_name}: Top Organism Patterns')
    plt.grid(True, alpha=0.3)

    # Top protein types
    plt.subplot(2, 2, 2)
    top_proteins = dict(sorted(protein_types.items(), key=lambda x: x[1], reverse=True)[:15])
    plt.bar(range(len(top_proteins)), list(top_proteins.values()))
    plt.xticks(range(len(top_proteins)), list(top_proteins.keys()), rotation=45)
    plt.xlabel('Protein Type Pattern')
    plt.ylabel('Number of Proteins')
    plt.title(f'{split_name}: Top Protein Types')
    plt.grid(True, alpha=0.3)

    # Sequence length by protein type
    plt.subplot(2, 2, 3)
    type_lengths = defaultdict(list)
    for stats in file_stats:
        protein_type = stats['protein_id'].split('_')[0][:4]
        type_lengths[protein_type].append(stats['avg_sequence_length'])

    # Top 10 protein types by frequency
    top_types = sorted(protein_types.items(), key=lambda x: x[1], reverse=True)[:10]
    lengths_data = [type_lengths[ptype] for ptype, _ in top_types]
    labels = [ptype for ptype, _ in top_types]

    plt.boxplot(lengths_data, labels=labels)
    plt.xticks(rotation=45)
    plt.ylabel('Average Sequence Length')
    plt.title(f'{split_name}: Length by Protein Type')
    plt.grid(True, alpha=0.3)

    # Dataset size by protein type
    plt.subplot(2, 2, 4)
    type_sizes = defaultdict(list)
    for stats in file_stats:
        protein_type = stats['protein_id'].split('_')[0][:4]
        type_sizes[protein_type].append(stats['num_sequences'])

    sizes_data = [type_sizes[ptype] for ptype, _ in top_types]
    plt.boxplot(sizes_data, labels=labels)
    plt.xticks(rotation=45)
    plt.ylabel('Number of Sequences')
    plt.title(f'{split_name}: Dataset Size by Type')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{split_name}_family_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    return organism_patterns, protein_types


def create_meta_learning_analysis(train_stats, test_stats, output_dir):
    """Analyze meta-learning suitability of the data."""
    plt.figure(figsize=(15, 12))

    # Support/Query split analysis
    support_size = 128
    query_size = 72
    min_required = support_size + query_size

    train_suitable = [stats for stats in train_stats if stats['num_sequences'] >= min_required]
    test_suitable = [stats for stats in test_stats if stats['num_sequences'] >= min_required]

    plt.subplot(2, 3, 1)
    train_sizes = [stats['num_sequences'] for stats in train_stats]
    test_sizes = [stats['num_sequences'] for stats in test_stats]

    plt.hist(train_sizes, bins=50, alpha=0.7, label='Train', density=True)
    plt.hist(test_sizes, bins=50, alpha=0.7, label='Test', density=True)
    plt.axvline(min_required, color='red', linestyle='--', label=f'Min Required ({min_required})')
    plt.xlabel('Number of Sequences')
    plt.ylabel('Density')
    plt.title('Dataset Size Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 2)
    suitability_data = [
        len(train_suitable), len(train_stats) - len(train_suitable),
        len(test_suitable), len(test_stats) - len(test_suitable)
    ]
    labels = ['Train Suitable', 'Train Too Small', 'Test Suitable', 'Test Too Small']
    colors = ['green', 'red', 'lightgreen', 'lightcoral']
    plt.bar(labels, suitability_data, color=colors)
    plt.ylabel('Number of Datasets')
    plt.title('Meta-Learning Suitability')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Score diversity analysis
    plt.subplot(2, 3, 3)
    train_diversity = [stats['std_score'] / (abs(stats['avg_score']) + 1e-8) for stats in train_suitable]
    test_diversity = [stats['std_score'] / (abs(stats['avg_score']) + 1e-8) for stats in test_suitable]

    plt.hist(train_diversity, bins=30, alpha=0.7, label='Train', density=True)
    plt.hist(test_diversity, bins=30, alpha=0.7, label='Test', density=True)
    plt.xlabel('Score Coefficient of Variation')
    plt.ylabel('Density')
    plt.title('Score Diversity Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Sequence length compatibility
    plt.subplot(2, 3, 4)
    train_lengths = [stats['avg_sequence_length'] for stats in train_suitable]
    test_lengths = [stats['avg_sequence_length'] for stats in test_suitable]

    plt.boxplot([train_lengths, test_lengths], labels=['Train', 'Test'])
    plt.ylabel('Average Sequence Length')
    plt.title('Sequence Length Distribution')
    plt.grid(True, alpha=0.3)

    # Task complexity scatter
    plt.subplot(2, 3, 5)
    train_complexity = [(stats['num_sequences'], stats['std_score']) for stats in train_suitable]
    test_complexity = [(stats['num_sequences'], stats['std_score']) for stats in test_suitable]

    if train_complexity:
        train_x, train_y = zip(*train_complexity)
        plt.scatter(train_x, train_y, alpha=0.6, label='Train', s=20)
    if test_complexity:
        test_x, test_y = zip(*test_complexity)
        plt.scatter(test_x, test_y, alpha=0.6, label='Test', s=20)

    plt.xlabel('Number of Sequences')
    plt.ylabel('Score Standard Deviation')
    plt.title('Task Complexity')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Summary statistics
    plt.subplot(2, 3, 6)
    summary_stats = [
        f"Train Tasks: {len(train_stats)}",
        f"Test Tasks: {len(test_stats)}",
        f"Train Suitable: {len(train_suitable)}",
        f"Test Suitable: {len(test_suitable)}",
        f"Total Sequences (Train): {sum(train_sizes):,}",
        f"Total Sequences (Test): {sum(test_sizes):,}",
        f"Avg Length (Train): {np.mean(train_lengths):.1f}",
        f"Avg Length (Test): {np.mean(test_lengths):.1f}"
    ]

    plt.text(0.1, 0.9, '\n'.join(summary_stats), transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.axis('off')
    plt.title('Summary Statistics')

    plt.tight_layout()
    plt.savefig(output_dir / 'meta_learning_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    return len(train_suitable), len(test_suitable)


def generate_eda_report(train_overview, test_overview, output_dir, logger):
    """Generate comprehensive EDA report."""
    logger.info("Generating EDA report...")

    report = {
        'dataset_overview': {
            'train': {
                'num_files': train_overview['num_files'],
                'valid_files': train_overview['valid_files'],
                'total_sequences': train_overview['total_sequences']
            },
            'test': {
                'num_files': test_overview['num_files'],
                'valid_files': test_overview['valid_files'],
                'total_sequences': test_overview['total_sequences']
            }
        },
        'sequence_statistics': {},
        'score_statistics': {},
        'recommendations': []
    }

    # Calculate aggregate statistics
    train_stats = train_overview['file_stats']
    test_stats = test_overview['file_stats']

    # Sequence statistics
    train_seq_lengths = [stats['avg_sequence_length'] for stats in train_stats]
    test_seq_lengths = [stats['avg_sequence_length'] for stats in test_stats]

    report['sequence_statistics'] = {
        'train': {
            'mean_length': float(np.mean(train_seq_lengths)),
            'std_length': float(np.std(train_seq_lengths)),
            'min_length': float(np.min(train_seq_lengths)),
            'max_length': float(np.max(train_seq_lengths))
        },
        'test': {
            'mean_length': float(np.mean(test_seq_lengths)),
            'std_length': float(np.std(test_seq_lengths)),
            'min_length': float(np.min(test_seq_lengths)),
            'max_length': float(np.max(test_seq_lengths))
        }
    }

    # Score statistics
    train_scores = [stats['avg_score'] for stats in train_stats]
    test_scores = [stats['avg_score'] for stats in test_stats]

    report['score_statistics'] = {
        'train': {
            'mean_score': float(np.mean(train_scores)),
            'std_score': float(np.std(train_scores)),
            'min_score': float(np.min(train_scores)),
            'max_score': float(np.max(train_scores))
        },
        'test': {
            'mean_score': float(np.mean(test_scores)),
            'std_score': float(np.std(test_scores)),
            'min_score': float(np.min(test_scores)),
            'max_score': float(np.max(test_scores))
        }
    }

    # Generate recommendations
    min_meta_learning = 200
    suitable_train = len([s for s in train_stats if s['num_sequences'] >= min_meta_learning])
    suitable_test = len([s for s in test_stats if s['num_sequences'] >= min_meta_learning])

    report['recommendations'] = [
        f"Total usable tasks for meta-learning: {suitable_train} train, {suitable_test} test",
        f"Average sequence length: {np.mean(train_seq_lengths + test_seq_lengths):.1f} amino acids",
        f"Recommended batch size: 32-64 (based on sequence lengths)",
        f"ESM2 tokenizer max_length should be set to: {int(np.percentile(train_seq_lengths + test_seq_lengths, 95)) + 50}",
        "Consider stratified sampling based on protein families for better generalization"
    ]

    # Save report
    with open(output_dir / 'eda_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    return report


def main():
    """Main EDA function."""
    parser = argparse.ArgumentParser(description='Run EDA on protein functionality data')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                        help='Directory containing raw data')
    parser.add_argument('--output-dir', type=str, default='outputs/eda',
                        help='Output directory for EDA results')
    parser.add_argument('--create-plots', action='store_true', default=True,
                        help='Create visualization plots')

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_eda_logging(output_dir)

    logger.info("Starting Protein Functionality EDA")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Set style for plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # Analyze training data
    train_dir = os.path.join(args.data_dir, 'Train_split')
    test_dir = os.path.join(args.data_dir, 'Test_split')

    if not os.path.exists(train_dir):
        logger.error(f"Training directory not found: {train_dir}")
        return

    if not os.path.exists(test_dir):
        logger.error(f"Test directory not found: {test_dir}")
        return

    # Analyze datasets
    train_overview = analyze_dataset_overview(train_dir, 'train', logger)
    test_overview = analyze_dataset_overview(test_dir, 'test', logger)

    if args.create_plots:
        logger.info("Creating visualization plots...")

        # Sequence analysis
        create_sequence_length_analysis(train_overview['file_stats'], output_dir, 'train')
        create_sequence_length_analysis(test_overview['file_stats'], output_dir, 'test')

        # Score analysis
        create_score_distribution_analysis(train_overview['file_stats'], output_dir, 'train')
        create_score_distribution_analysis(test_overview['file_stats'], output_dir, 'test')

        # Family analysis
        train_orgs, train_types = create_protein_family_analysis(train_overview['file_stats'], output_dir, 'train')
        test_orgs, test_types = create_protein_family_analysis(test_overview['file_stats'], output_dir, 'test')

        # Meta-learning analysis
        train_suitable, test_suitable = create_meta_learning_analysis(
            train_overview['file_stats'], test_overview['file_stats'], output_dir
        )

        logger.info("Plots saved to output directory")

    # Generate comprehensive report
    report = generate_eda_report(train_overview, test_overview, output_dir, logger)

    # Print summary
    logger.info("=== EDA SUMMARY ===")
    logger.info(f"Training files: {train_overview['valid_files']}/{train_overview['num_files']}")
    logger.info(f"Test files: {test_overview['valid_files']}/{test_overview['num_files']}")
    logger.info(f"Total training sequences: {train_overview['total_sequences']:,}")
    logger.info(f"Total test sequences: {test_overview['total_sequences']:,}")

    if args.create_plots:
        logger.info(f"Meta-learning suitable: {train_suitable} train, {test_suitable} test tasks")

    logger.info(f"EDA complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()