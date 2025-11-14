"""Machine learning models for witness learning."""

from .svm_witness import SVMWitnessLearner

# Try to import MLP (requires TensorFlow)
try:
    from .mlp_witness import MLPWitnessLearner
    __all__ = ['SVMWitnessLearner', 'MLPWitnessLearner']
except ImportError:
    __all__ = ['SVMWitnessLearner']
