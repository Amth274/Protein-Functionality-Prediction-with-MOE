#!/usr/bin/env python3
"""
Model evaluation script for protein functionality prediction.

This script evaluates trained MoE models on test data and generates comprehensive
metrics and visualizations.

Usage:
    python scripts/evaluate_model.py --model-path outputs/best_model.pth --test-dir data/embeddings/test --config configs/moe_config.yaml

Author: Generated for Protein Functionality Research
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import MoEModel
from data import EmbeddingDataset
from utils.logging import setup_logging
from utils.config import load_config
from utils.metrics import compute_ranking_metrics, aggregate_metrics_across_tasks


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate MoE model for protein functionality prediction')

    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--test-dir', type=str, required=True,
                        help='Directory containing test embeddings')
    parser.add_argument('--config', type=str, required=True,
                        help='Configuration file path')
    parser.add_argument('--output-dir', type=str, default='outputs/evaluation',
                        help='Output directory for evaluation results')

    # Evaluation options
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for evaluation')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save predictions for each task')
    parser.add_argument('--create-plots', action='store_true',
                        help='Create evaluation plots')

    return parser.parse_args()


def load_model(model_path: str, config: Dict, device: torch.device) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    # Create model
    model_config = config['model']
    model = MoEModel(
        input_dim=model_config.get('input_dim', 320),
        hidden_dim=model_config.get('hidden_dim', 512),
        num_experts=model_config.get('num_experts', 4),
        k=model_config.get('top_k', 2),
        num_layers=model_config.get('num_layers', 2),
        output_dim=model_config.get('output_dim', 1),
        dropout=model_config.get('dropout', 0.1)
    )

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Handle potential DataParallel/DistributedDataParallel wrapper
    state_dict = checkpoint['model_state_dict']
    if any(key.startswith('module.') for key in state_dict.keys()):
        # Remove 'module.' prefix
        state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


def evaluate_single_task(
    model: torch.nn.Module,
    embedding_file: str,
    device: torch.device,
    batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Evaluate model on a single task."""
    # Load embeddings
    data = torch.load(embedding_file, map_location='cpu')

    if isinstance(data, list):
        # Handle list of batches
        all_embeddings = []
        all_scores = []

        for batch in data:
            embeddings = batch.get('embedding', batch.get('embeddings'))
            scores = batch.get('score', batch.get('scores'))

            if torch.is_tensor(embeddings):
                all_embeddings.append(embeddings)
            if torch.is_tensor(scores):
                all_scores.append(scores)

        if all_embeddings:
            embeddings = torch.cat(all_embeddings, dim=0)
            scores = torch.cat(all_scores, dim=0)
        else:
            return np.array([]), np.array([]), {}

    elif isinstance(data, dict):
        # Handle single batch
        embeddings = data.get('embedding', data.get('embeddings'))
        scores = data.get('score', data.get('scores'))

        if not torch.is_tensor(embeddings) or not torch.is_tensor(scores):
            return np.array([]), np.array([]), {}

    else:
        return np.array([]), np.array([]), {}

    # Move to device and evaluate
    embeddings = embeddings.to(device)
    scores = scores.to(device)

    predictions = []
    targets = []

    with torch.no_grad():
        # Process in batches
        for i in range(0, len(embeddings), batch_size):
            batch_embeddings = embeddings[i:i+batch_size]
            batch_scores = scores[i:i+batch_size]

            # Forward pass
            batch_predictions, _ = model(batch_embeddings)
            batch_predictions = batch_predictions.squeeze(-1)

            predictions.append(batch_predictions.cpu())
            targets.append(batch_scores.cpu())

    # Concatenate results
    predictions = torch.cat(predictions).numpy()
    targets = torch.cat(targets).numpy()

    # Compute metrics
    metrics = compute_ranking_metrics(predictions, targets)

    return predictions, targets, metrics


def create_evaluation_plots(
    all_predictions: List[np.ndarray],
    all_targets: List[np.ndarray],
    task_names: List[str],
    output_dir: Path
):
    """Create evaluation plots."""
    # Create plots directory
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # 1. Scatter plot of predictions vs targets (per task)
    n_tasks = min(len(all_predictions), 9)  # Show up to 9 tasks
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for i in range(n_tasks):
        ax = axes[i]
        ax.scatter(all_targets[i], all_predictions[i], alpha=0.6, s=20)
        ax.plot([all_targets[i].min(), all_targets[i].max()],
                [all_targets[i].min(), all_targets[i].max()], 'r--', alpha=0.8)
        ax.set_xlabel('True Scores')
        ax.set_ylabel('Predicted Scores')
        ax.set_title(f'{task_names[i][:20]}...' if len(task_names[i]) > 20 else task_names[i])

    # Hide unused subplots
    for i in range(n_tasks, 9):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(plots_dir / 'predictions_vs_targets.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Overall correlation plot
    all_preds_concat = np.concatenate(all_predictions)
    all_targets_concat = np.concatenate(all_targets)

    plt.figure(figsize=(8, 8))
    plt.scatter(all_targets_concat, all_preds_concat, alpha=0.3, s=10)
    plt.plot([all_targets_concat.min(), all_targets_concat.max()],
             [all_targets_concat.min(), all_targets_concat.max()], 'r--', alpha=0.8)
    plt.xlabel('True Scores')
    plt.ylabel('Predicted Scores')
    plt.title('Overall Predictions vs Targets')

    # Add correlation info
    from scipy.stats import spearmanr, pearsonr
    spearman_corr, _ = spearmanr(all_targets_concat, all_preds_concat)
    pearson_corr, _ = pearsonr(all_targets_concat, all_preds_concat)

    plt.text(0.05, 0.95, f'Spearman: {spearman_corr:.3f}\nPearson: {pearson_corr:.3f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(plots_dir / 'overall_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Metrics distribution
    from utils.metrics import compute_ranking_metrics

    task_metrics = []
    for preds, targets in zip(all_predictions, all_targets):
        if len(preds) > 0:
            metrics = compute_ranking_metrics(preds, targets)
            task_metrics.append(metrics)

    if task_metrics:
        metrics_df = pd.DataFrame(task_metrics)

        # Select key metrics for plotting
        key_metrics = ['spearman', 'pearson', 'mse', 'mae']
        available_metrics = [m for m in key_metrics if m in metrics_df.columns]

        if available_metrics:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()

            for i, metric in enumerate(available_metrics[:4]):
                ax = axes[i]
                metrics_df[metric].hist(bins=20, alpha=0.7, ax=ax)
                ax.set_xlabel(metric.capitalize())
                ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution of {metric.capitalize()}')
                ax.axvline(metrics_df[metric].mean(), color='red', linestyle='--',
                          label=f'Mean: {metrics_df[metric].mean():.3f}')
                ax.legend()

            # Hide unused subplots
            for i in range(len(available_metrics), 4):
                axes[i].set_visible(False)

            plt.tight_layout()
            plt.savefig(plots_dir / 'metrics_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()


def main():
    """Main evaluation function."""
    args = parse_arguments()

    # Setup logging
    setup_logging(level='INFO')
    logger = logging.getLogger(__name__)

    logger.info("Starting model evaluation")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Test directory: {args.test_dir}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config = load_config(args.config)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load model
    logger.info("Loading model...")
    model = load_model(args.model_path, config, device)
    logger.info("Model loaded successfully")

    # Get test files
    test_dir = Path(args.test_dir)
    test_files = list(test_dir.glob('*.pt'))

    if not test_files:
        logger.error(f"No .pt files found in {test_dir}")
        return

    logger.info(f"Found {len(test_files)} test files")

    # Evaluate on each task
    all_predictions = []
    all_targets = []
    all_metrics = []
    task_names = []

    logger.info("Evaluating tasks...")
    for test_file in tqdm(test_files):
        task_name = test_file.stem
        task_names.append(task_name)

        try:
            predictions, targets, metrics = evaluate_single_task(
                model, str(test_file), device, args.batch_size
            )

            if len(predictions) > 0:
                all_predictions.append(predictions)
                all_targets.append(targets)
                all_metrics.append(metrics)

                logger.info(f"Task {task_name}: Spearman = {metrics.get('spearman', 0.0):.3f}")

                # Save predictions if requested
                if args.save_predictions:
                    pred_df = pd.DataFrame({
                        'predictions': predictions,
                        'targets': targets
                    })
                    pred_df.to_csv(output_dir / f'{task_name}_predictions.csv', index=False)

            else:
                logger.warning(f"No valid data found for task {task_name}")

        except Exception as e:
            logger.error(f"Error evaluating task {task_name}: {e}")

    # Aggregate metrics
    if all_metrics:
        logger.info("Computing aggregate metrics...")
        aggregate_metrics = aggregate_metrics_across_tasks(all_metrics)

        # Save metrics
        metrics_file = output_dir / 'evaluation_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(aggregate_metrics, f, indent=2)

        # Create summary
        summary = {
            'num_tasks': len(all_metrics),
            'num_total_samples': sum(len(targets) for targets in all_targets),
            'aggregate_metrics': aggregate_metrics
        }

        summary_file = output_dir / 'evaluation_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Print key results
        logger.info("=== Evaluation Results ===")
        logger.info(f"Number of tasks evaluated: {len(all_metrics)}")
        logger.info(f"Total samples: {sum(len(targets) for targets in all_targets)}")
        logger.info(f"Mean Spearman correlation: {aggregate_metrics.get('spearman_mean', 0.0):.3f} ± {aggregate_metrics.get('spearman_std', 0.0):.3f}")
        logger.info(f"Mean Pearson correlation: {aggregate_metrics.get('pearson_mean', 0.0):.3f} ± {aggregate_metrics.get('pearson_std', 0.0):.3f}")
        logger.info(f"Mean MSE: {aggregate_metrics.get('mse_mean', 0.0):.6f} ± {aggregate_metrics.get('mse_std', 0.0):.6f}")

        # Create plots if requested
        if args.create_plots:
            logger.info("Creating evaluation plots...")
            create_evaluation_plots(all_predictions, all_targets, task_names, output_dir)

        logger.info(f"Evaluation results saved to: {output_dir}")

    else:
        logger.error("No successful evaluations completed")


if __name__ == '__main__':
    main()