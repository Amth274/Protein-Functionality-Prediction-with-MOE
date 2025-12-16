"""
Configuration utilities for protein functionality prediction.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path (str): Path to configuration file

    Returns:
        Dict[str, Any]: Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Expand environment variables and relative paths
    config = _expand_paths(config)

    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.

    Args:
        config (Dict[str, Any]): Configuration dictionary
        config_path (str): Path to save configuration file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.

    Args:
        config (Dict[str, Any]): Configuration to validate

    Returns:
        bool: True if valid

    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ['model', 'data', 'training']
    missing_sections = [section for section in required_sections if section not in config]

    if missing_sections:
        raise ValueError(f"Missing required configuration sections: {missing_sections}")

    # Validate model configuration
    model_config = config['model']
    required_model_keys = ['input_dim', 'hidden_dim', 'num_experts']
    missing_model_keys = [key for key in required_model_keys if key not in model_config]

    if missing_model_keys:
        raise ValueError(f"Missing required model configuration keys: {missing_model_keys}")

    # Validate data configuration
    data_config = config['data']
    required_data_keys = ['data_dir', 'batch_size']
    missing_data_keys = [key for key in required_data_keys if key not in data_config]

    if missing_data_keys:
        raise ValueError(f"Missing required data configuration keys: {missing_data_keys}")

    # Validate training configuration
    training_config = config['training']
    required_training_keys = ['epochs', 'learning_rate']
    missing_training_keys = [key for key in required_training_keys if key not in training_config]

    if missing_training_keys:
        raise ValueError(f"Missing required training configuration keys: {missing_training_keys}")

    # Validate paths exist
    data_dir = Path(data_config['data_dir'])
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    return True


def _expand_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively expand environment variables and relative paths in config.

    Args:
        config (Dict[str, Any]): Configuration dictionary

    Returns:
        Dict[str, Any]: Configuration with expanded paths
    """
    if isinstance(config, dict):
        return {key: _expand_paths(value) for key, value in config.items()}
    elif isinstance(config, list):
        return [_expand_paths(item) for item in config]
    elif isinstance(config, str):
        # Expand environment variables
        expanded = os.path.expandvars(config)
        # Expand user home directory
        expanded = os.path.expanduser(expanded)
        return expanded
    else:
        return config


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.

    Returns:
        Dict[str, Any]: Default configuration dictionary
    """
    return {
        'model': {
            'input_dim': 320,
            'hidden_dim': 512,
            'num_experts': 4,
            'top_k': 2,
            'num_layers': 2,
            'output_dim': 1,
            'dropout': 0.1
        },
        'data': {
            'data_dir': 'data/raw',
            'embedding_dir': 'data/embeddings',
            'batch_size': 32,
            'num_workers': 4,
            'support_size': 128,
            'query_size': 72,
            'min_samples': 200,
            'max_train_files': None,
            'max_val_files': None
        },
        'training': {
            'epochs': 100,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'min_lr': 1e-6,
            'grad_clip_norm': 1.0,
            'seed': 42,
            'save_interval': 10,
            'loss': {
                'margin': 0.1,
                'n_pairs': 1000,
                'ranking_weight': 1.0,
                'use_mse': False,
                'mse_weight': 0.1,
                'aux_weight': 0.01
            }
        },
        'logging': {
            'log_interval': 100,
            'eval_interval': 1,
            'save_interval': 10
        },
        'paths': {
            'output_dir': 'outputs',
            'checkpoint_dir': 'outputs/checkpoints',
            'log_dir': 'outputs/logs'
        }
    }


def create_config_template(output_path: str):
    """
    Create a configuration template file.

    Args:
        output_path (str): Path to save the template
    """
    default_config = get_default_config()
    save_config(default_config, output_path)
    print(f"Configuration template saved to: {output_path}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.

    Args:
        base_config (Dict[str, Any]): Base configuration
        override_config (Dict[str, Any]): Override configuration

    Returns:
        Dict[str, Any]: Merged configuration
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def update_config_paths(config: Dict[str, Any], base_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Update relative paths in configuration to be relative to base_path.

    Args:
        config (Dict[str, Any]): Configuration dictionary
        base_path (str, optional): Base path for relative paths

    Returns:
        Dict[str, Any]: Updated configuration
    """
    if base_path is None:
        base_path = os.getcwd()

    base_path = Path(base_path)

    # Update data paths
    if 'data' in config:
        for key in ['data_dir', 'embedding_dir']:
            if key in config['data']:
                path = Path(config['data'][key])
                if not path.is_absolute():
                    config['data'][key] = str(base_path / path)

    # Update output paths
    if 'paths' in config:
        for key, path_value in config['paths'].items():
            path = Path(path_value)
            if not path.is_absolute():
                config['paths'][key] = str(base_path / path)

    return config