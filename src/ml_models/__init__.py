"""Machine learning models for witness learning."""

from .svm_witness import SVMWitnessLearner
from .witness_utils import (
    build_witness_operator,
    get_sparse_witness,
    evaluate_witness_on_states,
    get_witness_coefficients_dict,
    get_top_coefficients,
    categorize_pauli_terms,
    compute_term_importance,
)

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

__all__ = [
    'SVMWitnessLearner',
    # Witness utilities
    'build_witness_operator',
    'get_sparse_witness',
    'evaluate_witness_on_states',
    'get_witness_coefficients_dict',
    'get_top_coefficients',
    'categorize_pauli_terms',
    'compute_term_importance',
]

# Add transformer models to __all__ if available
if _TRANSFORMER_AVAILABLE:
    __all__.extend([
        'TransformerWitnessLearner',
        'TransformerClassifier',
        'HybridTransformerWitness',
    ])
