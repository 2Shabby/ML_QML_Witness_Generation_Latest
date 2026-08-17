"""Tests for the PennyLane direct density-matrix classifier."""

import numpy as np
import pytest
import torch

from src.ml_models.direct_state_qml import (
    DirectStateQMLClassifier,
    DirectStateQMLClassifierLearner,
)


def computational_basis_states(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Create alternating |000> and |111> density matrices."""
    states = np.zeros((n_samples, 8, 8), dtype=np.complex128)
    labels = np.arange(n_samples) % 2
    states[np.arange(n_samples), labels * 7, labels * 7] = 1.0
    return states, labels


def test_batched_forward_and_gradients():
    model = DirectStateQMLClassifier(n_layers=1)
    states, _ = computational_basis_states(4)

    logits = model(torch.as_tensor(states))
    logits.sum().backward()

    assert logits.shape == (4, 2)
    assert model.weights.grad is not None
    assert torch.isfinite(model.weights.grad).all()


def test_rejects_invalid_density_matrix():
    model = DirectStateQMLClassifier(n_layers=1)
    invalid = torch.eye(8, dtype=torch.complex128) * 2

    with pytest.raises(ValueError, match="unit trace"):
        model(invalid)


def test_learner_training_and_predictions():
    states, labels = computational_basis_states(12)
    learner = DirectStateQMLClassifierLearner(
        n_layers=1,
        n_epochs=1,
        batch_size=4,
        random_state=42,
        device="cpu",
    )

    metrics = learner.train(states, labels, test_size=0.25, verbose=False)
    predictions = learner.predict(states[:3])
    probabilities = learner.predict_proba(states[:3])

    assert "test_accuracy" in metrics
    assert predictions.shape == (3,)
    assert probabilities.shape == (3, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert learner.split_indices is not None
