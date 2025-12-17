"""Minimal utility functions for reproducibility."""

import random
import numpy as np


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)


__all__ = ['set_seed']
