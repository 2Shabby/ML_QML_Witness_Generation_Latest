"""
Utility functions for ML-QML Witness Generation.

- Random seed management for reproducibility
- Deterministic split-seed derivation
- Reusable stratified train/test split indices
"""

import random
from typing import Optional

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


def get_split_seed(base_seed: Optional[int], offset: int = 1000) -> Optional[int]:
    """
    Get a derived seed for data splitting to avoid correlation with model seed.

    This pattern is used consistently across the learners to ensure
    reproducible but independent data splits.

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
    'get_split_seed',
    'stratified_split_indices',
    'TORCH_AVAILABLE',
]
