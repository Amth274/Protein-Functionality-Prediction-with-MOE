#!/usr/bin/env python3
"""
Data preprocessing script for protein functionality prediction.

This script converts the original notebook functionality into a clean command-line
script for preprocessing DMS data and generating embeddings.

Usage:
    python scripts/preprocess_data.py --input-dir data/raw/Train_split --output-dir data/processed --embeddings-dir data/embeddings

Author: Generated for Protein Functionality Research
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.preprocessing import DataPreprocessor, EmbeddingGenerator
from utils.logging import setup_logging
from utils.config import load_config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess protein data for MoE training')

    parser.add_argument('--input-dir', type=str, required=True,
                        help='Input directory containing CSV files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for processed files')
    parser.add_argument('--embeddings-dir', type=str, default=None,
                        help='Output directory for embeddings (optional)')
    parser.add_argument('--config', type=str, default=None,
                        help='Configuration file path')

    # Preprocessing options
    parser.add_argument('--min-samples', type=int, default=200,
                        help='Minimum number of samples per file')
    parser.add_argument('--score-bins', type=int, default=3,
                        help='Number of bins for score discretization')

    # Embedding generation options
    parser.add_argument('--model-name', type=str, default='facebook/esm2_t6_8M_UR50D',
                        help='ESM model name for embeddings')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for embedding generation')
    parser.add_argument('--max-length', type=int, default=1024,
                        help='Maximum sequence length')
    parser.add_argument('--support-size', type=int, default=128,
                        help='Support set size')
    parser.add_argument('--query-size', type=int, default=72,
                        help='Query set size')

    # Processing options
    parser.add_argument('--skip-preprocessing', action='store_true',
                        help='Skip data preprocessing step')
    parser.add_argument('--skip-embeddings', action='store_true',
                        help='Skip embedding generation step')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for embedding generation')

    return parser.parse_args()


def main():
    """Main preprocessing function."""
    args = parse_arguments()

    # Setup logging
    setup_logging(level='INFO')
    logger = logging.getLogger(__name__)

    logger.info("Starting data preprocessing pipeline")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.embeddings_dir:
        embeddings_dir = Path(args.embeddings_dir)
        embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration if provided
    config = None
    if args.config:
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)

    # Step 1: Data Preprocessing
    if not args.skip_preprocessing:
        logger.info("Step 1: Preprocessing DMS data")

        preprocessor = DataPreprocessor(
            score_bins=args.score_bins
        )

        preprocessor.process_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            min_samples=args.min_samples
        )

        logger.info("Data preprocessing completed")
    else:
        logger.info("Skipping data preprocessing")

    # Step 2: Embedding Generation
    if not args.skip_embeddings and args.embeddings_dir:
        logger.info("Step 2: Generating protein embeddings")

        embedding_generator = EmbeddingGenerator(
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length
        )

        # Determine which directory to use for embedding generation
        source_dir = args.output_dir if not args.skip_preprocessing else args.input_dir

        # Generate embeddings
        split_name = Path(args.input_dir).name.lower()  # 'train_split' -> 'train'
        split_name = split_name.replace('_split', '')

        embedding_generator.generate_directory_embeddings(
            input_dir=source_dir,
            output_dir=args.embeddings_dir,
            split_name=split_name
        )

        logger.info("Embedding generation completed")
    else:
        if args.skip_embeddings:
            logger.info("Skipping embedding generation")
        else:
            logger.info("No embeddings directory specified, skipping embedding generation")

    logger.info("Preprocessing pipeline completed successfully")


if __name__ == '__main__':
    main()