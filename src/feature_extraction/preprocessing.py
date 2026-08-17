"""Feature transformations for amplitude-encoding control experiments."""

from typing import Dict

import numpy as np


def l2_normalize_features(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize each sample and return its original L2 norm separately."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, received {features.shape}")

    norms = np.linalg.norm(features, axis=1)
    if np.any(norms == 0):
        raise ValueError("Zero-norm feature vectors cannot be L2-normalized")
    return features / norms[:, None], norms


def create_amplitude_encoding_controls(
    features: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Create raw, normalized, and normalized-plus-norm classical inputs."""
    features = np.asarray(features, dtype=np.float64)
    normalized, norms = l2_normalize_features(features)
    return {
        "raw": features.copy(),
        "l2_normalized": normalized,
        "l2_plus_norm": np.column_stack((normalized, norms)),
    }
