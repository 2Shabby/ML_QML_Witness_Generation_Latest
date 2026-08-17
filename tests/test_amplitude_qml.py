"""Tests for the six-qubit amplitude-encoded QML classifier."""

import numpy as np
import pytest
import torch

from src.ml_models.amplitude_qml import (
    AmplitudeEncodedQMLClassifier,
    AmplitudeQMLClassifierLearner,
)


def test_batched_forward_and_gradients():
    model = AmplitudeEncodedQMLClassifier(n_layers=1)
    features = torch.randn(4, 36, dtype=torch.float64)

    logits = model(features)
    logits.sum().backward()

    assert logits.shape == (4, 2)
    assert model.weights.grad is not None
    assert torch.isfinite(model.weights.grad).all()


def test_rejects_insufficient_qubits():
    with pytest.raises(ValueError, match="encode at most"):
        AmplitudeEncodedQMLClassifier(n_features=65, n_qubits=6)


def test_rejects_zero_norm_features():
    model = AmplitudeEncodedQMLClassifier(n_layers=1)
    with pytest.raises(ValueError, match="Zero-norm"):
        model(torch.zeros(1, 36, dtype=torch.float64))


def test_learner_training_and_predictions():
    rng = np.random.default_rng(42)
    features = rng.normal(size=(20, 36))
    labels = (features[:, 0] > 0).astype(int)
    learner = AmplitudeQMLClassifierLearner(
        n_layers=1,
        n_epochs=2,
        batch_size=8,
        random_state=42,
        device="cpu",
    )

    metrics = learner.train(features, labels, test_size=0.2, verbose=False)
    predictions = learner.predict(features[:3])
    probabilities = learner.predict_proba(features[:3])

    assert "test_accuracy" in metrics
    assert predictions.shape == (3,)
    assert probabilities.shape == (3, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert learner.split_indices is not None
def test_save_and_load(tmp_path):
    rng = np.random.default_rng(7)
    features = rng.normal(size=(12, 36))
    labels = np.array([0, 1] * 6)
    learner = AmplitudeQMLClassifierLearner(
        n_layers=1,
        n_epochs=1,
        batch_size=6,
        random_state=7,
        device="cpu",
    )
    learner.train(features, labels, verbose=False)
    path = tmp_path / "amplitude_qml.pt"
    learner.save(path)

    restored = AmplitudeQMLClassifierLearner(n_layers=1, device="cpu")
    restored.load(path)

    np.testing.assert_array_equal(
        learner.predict(features[:4]),
        restored.predict(features[:4]),
    )
