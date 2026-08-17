"""
Centralized Configuration for ML-QML Witness Generation

This module provides a single source of truth for all configuration values
used across the project, including experiment parameters, model hyperparameters,
and default settings.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


# =============================================================================
# EXPERIMENT DEFAULTS
# =============================================================================

# Dataset generation
DEFAULT_N_SAMPLES = 5000
DEFAULT_NOISE_RANGE: Tuple[float, float] = (0.0, 0.5)

# Cross-validation
DEFAULT_CV_FOLDS = 5
DEFAULT_SEEDS: List[int] = [42, 123, 456, 789, 1011]
DEFAULT_SEED = 42


# =============================================================================
# SVM CONFIGURATION
# =============================================================================

@dataclass
class SVMConfig:
    """Configuration for SVM witness learner."""
    C: float = 1.0
    kernel: str = 'linear'


DEFAULT_SVM_CONFIG = SVMConfig()


# =============================================================================
# TRANSFORMER CONFIGURATION
# =============================================================================

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
# MLP CONFIGURATION
# =============================================================================

@dataclass
class MLPConfig:
    """Configuration for MLP discriminator classifier."""
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.3
    learning_rate: float = 1e-3
    batch_size: int = 64
    n_epochs: int = 100
    patience: int = 15

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'patience': self.patience,
        }


DEFAULT_MLP_CONFIG = MLPConfig()


# =============================================================================
# POVM CONFIGURATION
# =============================================================================

@dataclass
class POVMConfig:
    """Configuration for variational POVM learner."""
    n_qubits: int = 3
    n_layers: int = 4
    learning_rate: float = 1e-3
    batch_size: int = 64
    n_epochs: int = 200
    patience: int = 20

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'patience': self.patience,
        }


DEFAULT_POVM_CONFIG = POVMConfig()


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

from pathlib import Path

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results'


__all__ = [
    # Experiment defaults
    'DEFAULT_N_SAMPLES',
    'DEFAULT_NOISE_RANGE',
    'DEFAULT_CV_FOLDS',
    'DEFAULT_SEEDS',
    'DEFAULT_SEED',
    # SVM
    'SVMConfig',
    'DEFAULT_SVM_CONFIG',
    # Transformer
    'TransformerConfig',
    'DEFAULT_TRANSFORMER_CONFIG',
    # MLP
    'MLPConfig',
    'DEFAULT_MLP_CONFIG',
    # POVM
    'POVMConfig',
    'DEFAULT_POVM_CONFIG',
    # Logging
    'DEFAULT_LOG_FORMAT',
    # Paths
    'PROJECT_ROOT',
    'RESULTS_DIR',
]
