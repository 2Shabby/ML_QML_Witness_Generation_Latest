"""Machine learning models for witness learning."""

from .svm_witness import SVMWitnessLearner

# Conditionally import transformer models (requires PyTorch)
try:
    from .transformer_witness import (
        TransformerWitnessLearner,
        TransformerClassifier,
        HybridTransformerWitness
    )
    _TRANSFORMER_AVAILABLE = True
except ImportError:
    _TRANSFORMER_AVAILABLE = False
    TransformerWitnessLearner = None
    TransformerClassifier = None
    HybridTransformerWitness = None

# Conditionally import MLP classifier (requires PyTorch)
try:
    from .mlp_classifier import MLPClassifierLearner, MLPDiscriminator
    _MLP_AVAILABLE = True
except ImportError:
    _MLP_AVAILABLE = False
    MLPClassifierLearner = None
    MLPDiscriminator = None

# Conditionally import variational POVM (requires PyTorch)
try:
    from .variational_povm import (
        VariationalPOVMLearner,
        VariationalPOVMClassifier,
        ParameterizedUnitary,
    )
    _POVM_AVAILABLE = True
except ImportError:
    _POVM_AVAILABLE = False
    VariationalPOVMLearner = None
    VariationalPOVMClassifier = None
    ParameterizedUnitary = None

__all__ = [
    'SVMWitnessLearner',
]

# Add transformer models to __all__ if available
if _TRANSFORMER_AVAILABLE:
    __all__.extend([
        'TransformerWitnessLearner',
        'TransformerClassifier',
        'HybridTransformerWitness',
    ])

# Add MLP models to __all__ if available
if _MLP_AVAILABLE:
    __all__.extend([
        'MLPClassifierLearner',
        'MLPDiscriminator',
    ])

# Add POVM models to __all__ if available
if _POVM_AVAILABLE:
    __all__.extend([
        'VariationalPOVMLearner',
        'VariationalPOVMClassifier',
        'ParameterizedUnitary',
    ])
