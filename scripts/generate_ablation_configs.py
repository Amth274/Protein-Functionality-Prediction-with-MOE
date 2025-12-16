#!/usr/bin/env python3
"""
Generate YAML configuration files for ablation study experiments.
Creates 35+ config files for systematic evaluation of model components.
"""

import os
import yaml
from pathlib import Path


# Base configuration (A1.1 - Baseline)
BASE_CONFIG = {
    'model': {
        'esm2_model': 'facebook/esm2_t6_8M_UR50D',
        'num_experts': 8,
        'hidden_dim': 256,
        'dropout': 0.1,
        'freeze_backbone': False,
    },
    'data': {
        'train_dir': 'data/raw/Train_split',
        'test_dir': 'data/raw/Test_split',
        'max_sequence_length': 4096,
        'min_samples': 10,
        'support_frac': 0.8,
        'num_workers': 2,
    },
    'training': {
        'task_epochs': 1,
        'batch_size': 4,
        'task_learning_rate': 0.00001,
        'weight_decay': 0.01,
        'grad_clip_norm': 1.0,
        'mixed_precision': True,
        'seed': 42,
    },
    'logging': {
        'log_interval': 1,
        'save_interval': 10,
    }
}


def create_config(exp_id, description, modifications):
    """Create a configuration by applying modifications to base config."""
    config = {
        'experiment': {
            'id': exp_id,
            'description': description,
            'baseline': 'A1.1'
        }
    }

    # Deep copy base config
    import copy
    config.update(copy.deepcopy(BASE_CONFIG))

    # Apply modifications
    for key_path, value in modifications.items():
        keys = key_path.split('.')
        current = config
        for key in keys[:-1]:
            current = current[key]
        current[keys[-1]] = value

    return config


def save_config(config, filepath):
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Created: {filepath}")


def generate_all_configs():
    """Generate all ablation study configurations."""

    output_dir = Path('configs/ablations')
    configs = []

    # =========================================================================
    # Study 1: Architecture Ablations (MoE Components)
    # =========================================================================

    # A1.1: Baseline (already trained)
    configs.append(('A1.1', 'Baseline - 8 experts, 256 dim', {}))

    # A1.2: Single Expert
    configs.append(('A1.2', 'Single Expert - No specialization', {
        'model.num_experts': 1
    }))

    # A1.3: Linear Head (will need code modification - skip for now)
    # configs.append(('A1.3', 'Linear Head - Simplest baseline', {
    #     'model.num_experts': 0  # Special case
    # }))

    # A1.4: Few Experts
    configs.append(('A1.4', 'Few Experts - 2 experts', {
        'model.num_experts': 2
    }))

    # A1.5: Medium Experts
    configs.append(('A1.5', 'Medium Experts - 4 experts', {
        'model.num_experts': 4
    }))

    # A1.6: Many Experts
    configs.append(('A1.6', 'Many Experts - 16 experts', {
        'model.num_experts': 16
    }))

    # A1.7: Huge Experts
    configs.append(('A1.7', 'Huge Experts - 32 experts', {
        'model.num_experts': 32
    }))

    # =========================================================================
    # Study 2: Expert Capacity Ablations
    # =========================================================================

    # A2.2: Tiny Experts
    configs.append(('A2.2', 'Tiny Experts - 64 dim', {
        'model.hidden_dim': 64
    }))

    # A2.3: Small Experts
    configs.append(('A2.3', 'Small Experts - 128 dim', {
        'model.hidden_dim': 128
    }))

    # A2.4: Large Experts
    configs.append(('A2.4', 'Large Experts - 512 dim', {
        'model.hidden_dim': 512
    }))

    # A2.5: Huge Experts
    configs.append(('A2.5', 'Huge Experts - 1024 dim', {
        'model.hidden_dim': 1024
    }))

    # =========================================================================
    # Study 3: Backbone Model Ablations
    # =========================================================================

    # A3.2: Frozen 8M
    configs.append(('A3.2', 'Frozen ESM2-8M backbone', {
        'model.freeze_backbone': True
    }))

    # A3.3: ESM2-35M
    configs.append(('A3.3', 'ESM2-35M backbone', {
        'model.esm2_model': 'facebook/esm2_t12_35M_UR50D'
    }))

    # A3.4: ESM2-150M
    configs.append(('A3.4', 'ESM2-150M backbone', {
        'model.esm2_model': 'facebook/esm2_t30_150M_UR50D'
    }))

    # A3.5: ESM2-650M
    configs.append(('A3.5', 'ESM2-650M backbone', {
        'model.esm2_model': 'facebook/esm2_t33_650M_UR50D',
        'training.batch_size': 2,  # Reduce batch size for memory
    }))

    # A3.6: Frozen 650M
    configs.append(('A3.6', 'Frozen ESM2-650M backbone', {
        'model.esm2_model': 'facebook/esm2_t33_650M_UR50D',
        'model.freeze_backbone': True,
        'training.batch_size': 2,
    }))

    # =========================================================================
    # Study 4: Training Strategy Ablations
    # =========================================================================

    # A4.3: More Query (50% support)
    configs.append(('A4.3', 'More Query - 50% support split', {
        'data.support_frac': 0.5
    }))

    # A4.4: Balanced (70% support)
    configs.append(('A4.4', 'Balanced - 70% support split', {
        'data.support_frac': 0.7
    }))

    # A4.5: Less Query (90% support)
    configs.append(('A4.5', 'Less Query - 90% support split', {
        'data.support_frac': 0.9
    }))

    # A4.6: Multi-Epoch (3 epochs)
    configs.append(('A4.6', 'Multi-Epoch - 3 epochs per task', {
        'training.task_epochs': 3
    }))

    # =========================================================================
    # Study 6: Regularization and Optimization
    # =========================================================================

    # A6.2: No Dropout
    configs.append(('A6.2', 'No Dropout', {
        'model.dropout': 0.0
    }))

    # A6.3: High Dropout
    configs.append(('A6.3', 'High Dropout - 0.3', {
        'model.dropout': 0.3
    }))

    # A6.4: No Weight Decay
    configs.append(('A6.4', 'No Weight Decay', {
        'training.weight_decay': 0.0
    }))

    # A6.5: Strong Weight Decay
    configs.append(('A6.5', 'Strong Weight Decay - 0.1', {
        'training.weight_decay': 0.1
    }))

    # A6.6: No Gradient Clipping
    configs.append(('A6.6', 'No Gradient Clipping', {
        'training.grad_clip_norm': None
    }))

    # =========================================================================
    # Study 7: Data Efficiency
    # =========================================================================

    # Note: These require modifying the data loader to sample tasks
    # We can use a different random seed and train_fraction parameter

    # A7.2: 25% data
    configs.append(('A7.2', '25% Training Data - 43 tasks', {
        'training.seed': 43,  # Different seed for sampling
        'experiment.train_fraction': 0.25
    }))

    # A7.3: 50% data
    configs.append(('A7.3', '50% Training Data - 86 tasks', {
        'training.seed': 44,
        'experiment.train_fraction': 0.50
    }))

    # A7.4: 75% data
    configs.append(('A7.4', '75% Training Data - 130 tasks', {
        'training.seed': 45,
        'experiment.train_fraction': 0.75
    }))

    # =========================================================================
    # Generate all config files
    # =========================================================================

    print(f"\nGenerating {len(configs)} ablation configurations...\n")

    for exp_id, description, modifications in configs:
        config = create_config(exp_id, description, modifications)
        filepath = output_dir / f"{exp_id}.yaml"
        save_config(config, filepath)

    print(f"\n✓ Generated {len(configs)} configuration files in {output_dir}/")

    # Create experiment index
    index = {
        'total_experiments': len(configs),
        'studies': {
            'A1': 'Architecture Ablations (MoE Components)',
            'A2': 'Expert Capacity Ablations',
            'A3': 'Backbone Model Ablations',
            'A4': 'Training Strategy Ablations',
            'A6': 'Regularization and Optimization',
            'A7': 'Data Efficiency'
        },
        'experiments': [
            {
                'id': exp_id,
                'description': description,
                'modifications': modifications
            }
            for exp_id, description, modifications in configs
        ]
    }

    index_file = output_dir / 'index.yaml'
    save_config(index, index_file)
    print(f"✓ Created experiment index: {index_file}")

    return len(configs)


if __name__ == '__main__':
    num_configs = generate_all_configs()
    print(f"\n{'='*70}")
    print(f"Ready to run ablation studies!")
    print(f"Total experiments: {num_configs}")
    print(f"Estimated GPU time: ~{num_configs * 2.7:.1f} hours")
    print(f"{'='*70}\n")
