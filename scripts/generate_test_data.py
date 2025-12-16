#!/usr/bin/env python3
"""
Generate synthetic protein data for testing the training pipeline.

This script creates synthetic protein sequences and fitness scores
for testing purposes when the full ProteinGym dataset is not available.

Usage:
    python scripts/generate_test_data.py --num-files 5 --sequences-per-file 300
    python scripts/generate_test_data.py --quick  # Quick test with 2 files
"""

import argparse
import random
import pandas as pd
from pathlib import Path


def generate_random_sequence(length=100):
    """Generate a random amino acid sequence."""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    return ''.join(random.choices(amino_acids, k=length))


def generate_protein_dataset(num_sequences=300, seq_length=100):
    """Generate a synthetic protein dataset."""
    data = {
        'mutant': [f'M{i}' for i in range(num_sequences)],
        'mutated_sequence': [generate_random_sequence(seq_length) for _ in range(num_sequences)],
        'DMS_score': [random.gauss(0, 1) for _ in range(num_sequences)],
        'DMS_score_bin': [random.randint(0, 2) for _ in range(num_sequences)]
    }

    # Add some structure to scores
    for i in range(num_sequences):
        # Make some mutations clearly beneficial/detrimental
        if i % 10 == 0:
            data['DMS_score'][i] = random.gauss(2, 0.5)  # Beneficial
            data['DMS_score_bin'][i] = 2
        elif i % 10 == 5:
            data['DMS_score'][i] = random.gauss(-2, 0.5)  # Detrimental
            data['DMS_score_bin'][i] = 0

    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic protein test data')
    parser.add_argument('--num-files', type=int, default=5,
                       help='Number of CSV files to generate')
    parser.add_argument('--sequences-per-file', type=int, default=300,
                       help='Number of sequences per file')
    parser.add_argument('--seq-length', type=int, default=100,
                       help='Length of each sequence')
    parser.add_argument('--train-dir', type=str, default='data/raw/Train_split',
                       help='Output directory for training files')
    parser.add_argument('--test-dir', type=str, default='data/raw/Test_split',
                       help='Output directory for test files')
    parser.add_argument('--test-split', type=float, default=0.2,
                       help='Fraction of files for testing')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: generate 2 train + 1 test files')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Quick mode
    if args.quick:
        args.num_files = 3
        args.sequences_per_file = 250
        print("Quick mode: generating 2 train + 1 test files with 250 sequences each")

    # Create output directories
    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Calculate split
    num_test = max(1, int(args.num_files * args.test_split))
    num_train = args.num_files - num_test

    print(f"Generating synthetic protein data...")
    print(f"  Training files: {num_train}")
    print(f"  Test files: {num_test}")
    print(f"  Sequences per file: {args.sequences_per_file}")
    print(f"  Sequence length: {args.seq_length}")
    print()

    # Generate training files
    print("Creating training files:")
    for i in range(num_train):
        df = generate_protein_dataset(args.sequences_per_file, args.seq_length)
        output_path = train_dir / f'synthetic_protein_train_{i:03d}.csv'
        df.to_csv(output_path, index=False)
        print(f"  ✓ {output_path.name}")

    # Generate test files
    print("\nCreating test files:")
    for i in range(num_test):
        df = generate_protein_dataset(args.sequences_per_file, args.seq_length)
        output_path = test_dir / f'synthetic_protein_test_{i:03d}.csv'
        df.to_csv(output_path, index=False)
        print(f"  ✓ {output_path.name}")

    print("\n" + "="*70)
    print("✓ Synthetic data generation complete!")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  Training: {train_dir}/ ({num_train} files)")
    print(f"  Testing:  {test_dir}/ ({num_test} files)")
    print(f"\nTotal sequences: {args.num_files * args.sequences_per_file:,}")
    print(f"\nYou can now run training:")
    print(f"  python scripts/train_meta_learning_optimized.py \\")
    print(f"      --config configs/meta_learning_config.yaml")
    print()
    print("Note: This is synthetic data for testing only.")
    print("      For real experiments, use ProteinGym dataset.")
    print("      See DATA_GUIDE.md for download instructions.")


if __name__ == '__main__':
    main()
