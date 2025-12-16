"""
Logging utilities for protein functionality prediction.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_file: Optional[str] = None,
    level: str = 'INFO',
    format_string: Optional[str] = None
):
    """
    Setup logging configuration.

    Args:
        log_file (str, optional): Path to log file
        level (str): Logging level
        format_string (str, optional): Custom format string
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Setup specific loggers
    setup_library_loggers()


def setup_library_loggers():
    """Setup logging for common libraries."""
    # Reduce noise from transformers
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)

    # Reduce noise from matplotlib
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    # Reduce noise from PIL
    logging.getLogger('PIL').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name (str): Logger name

    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for terminal output."""

    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"

        # Format the message
        return super().format(record)


def setup_colored_logging(
    log_file: Optional[str] = None,
    level: str = 'INFO'
):
    """
    Setup logging with colored console output.

    Args:
        log_file (str, optional): Path to log file
        level (str): Logging level
    """
    # Format string
    format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Colored console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = ColoredFormatter(format_string)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Plain file handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Setup library loggers
    setup_library_loggers()


class TrainingLogger:
    """Logger specifically for training metrics and progress."""

    def __init__(self, log_file: Optional[str] = None):
        self.logger = get_logger('training')

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Create a separate handler for training metrics
            handler = logging.FileHandler(log_path)
            handler.setLevel(logging.INFO)

            # Simple format for metrics
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            handler.setFormatter(formatter)

            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict):
        """Log metrics for an epoch."""
        self.logger.info(f"Epoch {epoch}")
        self.logger.info(f"Train: {self._format_metrics(train_metrics)}")
        self.logger.info(f"Val: {self._format_metrics(val_metrics)}")

    def log_metrics(self, prefix: str, metrics: dict):
        """Log a set of metrics."""
        self.logger.info(f"{prefix}: {self._format_metrics(metrics)}")

    def _format_metrics(self, metrics: dict) -> str:
        """Format metrics dictionary as string."""
        formatted = []
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted.append(f"{key}={value:.6f}")
            else:
                formatted.append(f"{key}={value}")
        return " | ".join(formatted)


def log_system_info():
    """Log system information for debugging."""
    import torch
    import platform

    logger = get_logger('system')

    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        logger.info(f"CUDA available: True")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name} ({props.total_memory // 1024**2} MB)")
    else:
        logger.info("CUDA available: False")

    logger.info("==========================")