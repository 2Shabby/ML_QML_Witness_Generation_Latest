"""
Utility functions for ML-QML Witness Generation.

This module provides common utilities used across the project:
- Random seed management for reproducibility
- JSON serialization helpers
- Logging setup utilities
"""

import random
import logging
from typing import Any, Optional

import numpy as np
from sklearn.model_selection import train_test_split

# Try to import torch for complete seed setting
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    This ensures consistent behavior for:
    - Python's random module
    - NumPy's random number generator
    - PyTorch (if available)

    Args:
        seed: Random seed value
        deterministic: If True and torch is available, enables deterministic
                      algorithms (may impact performance)
    """
    random.seed(seed)
    np.random.seed(seed)

    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def convert_to_json_serializable(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python native types for JSON serialization.

    Handles:
    - numpy arrays -> lists
    - numpy integers -> int
    - numpy floats -> float
    - numpy booleans -> bool
    - Nested dicts and lists

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    logger_name: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        format_string: Custom format string. If None, uses default format.
        logger_name: Name for the logger. If None, configures root logger.

    Returns:
        Configured logger instance
    """
    from ..config import DEFAULT_LOG_FORMAT

    if format_string is None:
        format_string = DEFAULT_LOG_FORMAT

    if logger_name is None:
        # Configure root logger
        logging.basicConfig(level=level, format=format_string)
        return logging.getLogger()
    else:
        # Configure named logger
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)

        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(level)
            handler.setFormatter(logging.Formatter(format_string))
            logger.addHandler(handler)

        return logger


def get_split_seed(base_seed: Optional[int], offset: int = 1000) -> Optional[int]:
    """
    Get a derived seed for data splitting to avoid correlation with model seed.

    This pattern is used consistently across SVM and Transformer learners
    to ensure reproducible but independent data splits.

    Args:
        base_seed: The base random seed (can be None)
        offset: Offset to add to the base seed

    Returns:
        Derived seed for data splitting, or None if base_seed is None
    """
    return base_seed + offset if base_seed is not None else None


def stratified_split_indices(
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """Return reproducible train/test indices for reuse across models."""
    labels = np.asarray(labels)
    indices = np.arange(len(labels))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=get_split_seed(random_state),
        stratify=labels,
    )
    return {"train": train_indices, "test": test_indices}


__all__ = [
    'set_seed',
    'convert_to_json_serializable',
    'setup_logging',
    'get_split_seed',
    'stratified_split_indices',
    'TORCH_AVAILABLE',
]
