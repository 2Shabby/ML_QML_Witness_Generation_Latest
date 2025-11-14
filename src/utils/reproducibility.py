"""
Reproducibility Utilities

Functions to ensure reproducible results across runs.
Sets seeds for Python, NumPy, and TensorFlow.
"""

import random
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_GLOBAL_SEED = None
_RANDOM_STATE = None


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    global _GLOBAL_SEED, _RANDOM_STATE

    _GLOBAL_SEED = seed
    _RANDOM_STATE = np.random.RandomState(seed)

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # TensorFlow (if available)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        logger.debug(f"Set TensorFlow random seed: {seed}")
    except ImportError:
        pass

    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug(f"Set PyTorch random seed: {seed}")
    except ImportError:
        pass

    logger.info(f"Global random seed set to: {seed}")


def get_random_state() -> Optional[np.random.RandomState]:
    """
    Get the global RandomState object.

    Returns:
        NumPy RandomState or None if seed not set
    """
    return _RANDOM_STATE


def get_seed() -> Optional[int]:
    """
    Get the current global seed.

    Returns:
        Current seed value or None if not set
    """
    return _GLOBAL_SEED
