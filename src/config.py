"""
Centralized Configuration for ML-QML Witness Generation.

Only the settings actually consumed by the live code paths live here:
the transformer architecture/training defaults and the log format.
The MLP, SVM, and both QML learners keep their hyperparameters as
constructor defaults; ``scripts/run_clean_dataset_experiments.py``
records the exact settings in its JSON artifact.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TransformerConfig:
    """
    Configuration for transformer witness learner.

    These are minimal settings optimized for the 36D binary classification task.
    The architecture is intentionally small since the feature space is limited.
    """
    # Architecture
    d_model: int = 16        # Hidden dimension (reduced from 64)
    n_heads: int = 2         # Attention heads (reduced from 4)
    n_layers: int = 1        # Transformer layers (reduced from 2)
    d_ff: int = 32           # Feed-forward dimension (reduced from 128)
    dropout: float = 0.1     # Dropout probability

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    n_epochs: int = 100
    patience: int = 15       # Early stopping patience

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'd_ff': self.d_ff,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'patience': self.patience,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TransformerConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


DEFAULT_TRANSFORMER_CONFIG = TransformerConfig()


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


__all__ = [
    'TransformerConfig',
    'DEFAULT_TRANSFORMER_CONFIG',
    'DEFAULT_LOG_FORMAT',
]
