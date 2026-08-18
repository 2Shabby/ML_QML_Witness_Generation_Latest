"""Machine learning models for witness learning."""

from .svm_witness import SVMWitnessLearner
from .transformer_witness import (
    TransformerWitnessLearner,
    TransformerClassifier,
    HybridTransformerWitness,
)
from .mlp_classifier import MLPClassifierLearner, MLPDiscriminator
from .amplitude_qml import (
    AmplitudeEncodedQMLClassifier,
    AmplitudeQMLClassifierLearner,
)
from .direct_state_qml import (
    DirectStateQMLClassifier,
    DirectStateQMLClassifierLearner,
)

__all__ = [
    'SVMWitnessLearner',
    'TransformerWitnessLearner',
    'TransformerClassifier',
    'HybridTransformerWitness',
    'MLPClassifierLearner',
    'MLPDiscriminator',
    'AmplitudeEncodedQMLClassifier',
    'AmplitudeQMLClassifierLearner',
    'DirectStateQMLClassifier',
    'DirectStateQMLClassifierLearner',
]
