"""
Tests for MLP Discriminator Classifier

Tests the MLPClassifier model for quantum state classification.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Skip all tests if torch not available
torch_available = True
try:
    import torch
    from src.ml_models.mlp_classifier import (
        MLPClassifierLearner,
        MLPDiscriminator
    )
except ImportError:
    torch_available = False

from src.feature_extraction.pauli_features import create_sparse_measurement_set

pytestmark = pytest.mark.skipif(
    not torch_available,
    reason="PyTorch not installed"
)


class TestMLPDiscriminator:
    """Test MLPDiscriminator neural network."""

    def test_output_shape(self):
        """Test network produces correct output shape."""
        model = MLPDiscriminator(input_dim=36, hidden_dims=[64, 32])
        x = torch.randn(10, 36)
        out = model(x)
        assert out.shape == (10, 2)

    def test_default_hidden_dims(self):
        """Test default hidden layer configuration."""
        model = MLPDiscriminator(input_dim=36)
        assert model.hidden_dims == [128, 64, 32]
        x = torch.randn(5, 36)
        out = model(x)
        assert out.shape == (5, 2)

    def test_custom_hidden_dims(self):
        """Test custom hidden layer configuration."""
        model = MLPDiscriminator(input_dim=36, hidden_dims=[128, 64, 32, 16])
        x = torch.randn(5, 36)
        out = model(x)
        assert out.shape == (5, 2)

    def test_get_features(self):
        """Test intermediate feature extraction."""
        model = MLPDiscriminator(input_dim=36, hidden_dims=[64, 32])
        x = torch.randn(10, 36)
        features = model.get_features(x)
        # After last hidden layer, dimension should be 32
        assert features.shape == (10, 32)

    def test_single_sample(self):
        """Test a singleton training batch supports forward and backward passes."""
        model = MLPDiscriminator(input_dim=36, hidden_dims=[64, 32])
        x = torch.randn(1, 36)
        out = model(x)
        out.sum().backward()

        assert out.shape == (1, 2)
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )

class TestMLPClassifierLearner:
    """Test MLPClassifierLearner wrapper class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample classification data."""
        np.random.seed(42)
        n_samples = 200
        n_features = 36

        # Create linearly separable data with some noise
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.5 > 0).astype(int)

        return X, y

    def test_initialization(self):
        """Test learner initialization."""
        learner = MLPClassifierLearner(n_features=36)
        assert learner.n_features == 36
        assert learner.is_trained is False
        assert learner.hidden_dims == [128, 64, 32]

    def test_initialization_custom_params(self):
        """Test learner initialization with custom parameters."""
        learner = MLPClassifierLearner(
            n_features=36,
            hidden_dims=[64, 32],
            dropout=0.5,
            learning_rate=0.01,
            batch_size=32,
            n_epochs=50,
            patience=10
        )
        assert learner.n_features == 36
        assert learner.hidden_dims == [64, 32]
        assert learner.dropout == 0.5
        assert learner.learning_rate == 0.01
        assert learner.batch_size == 32
        assert learner.n_epochs == 50
        assert learner.patience == 10

    def test_train(self, sample_data):
        """Test training with automatic split."""
        X, y = sample_data
        learner = MLPClassifierLearner(
            n_features=36,
            n_epochs=20,
            random_state=42
        )

        metrics = learner.train(X, y, test_size=0.2, verbose=False)

        assert learner.is_trained is True
        assert 'train_accuracy' in metrics
        assert 'test_accuracy' in metrics
        assert metrics['train_accuracy'] > 0.5

    def test_fit(self, sample_data):
        """Test training with pre-split data."""
        X, y = sample_data
        X_train, X_test = X[:160], X[160:]
        y_train, y_test = y[:160], y[160:]

        learner = MLPClassifierLearner(n_features=36, n_epochs=20, random_state=42)
        metrics = learner.fit(X_train, y_train, X_test, y_test, verbose=False)

        assert learner.is_trained is True
        assert 'test_accuracy' in metrics

    def test_fit_without_validation(self, sample_data):
        """Test training without validation data."""
        X, y = sample_data
        learner = MLPClassifierLearner(n_features=36, n_epochs=20, random_state=42)
        metrics = learner.fit(X, y, verbose=False)

        assert learner.is_trained is True
        assert 'train_accuracy' in metrics
        assert 'test_accuracy' not in metrics

    def test_predict(self, sample_data):
        """Test prediction."""
        X, y = sample_data
        learner = MLPClassifierLearner(n_features=36, n_epochs=20, random_state=42)
        learner.train(X, y, verbose=False)

        predictions = learner.predict(X[:10])

        assert predictions.shape == (10,)
        assert set(predictions).issubset({0, 1})

    def test_predict_proba(self, sample_data):
        """Test probability prediction."""
        X, y = sample_data
        learner = MLPClassifierLearner(n_features=36, n_epochs=20, random_state=42)
        learner.train(X, y, verbose=False)

        probs = learner.predict_proba(X[:10])

        assert probs.shape == (10, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_decision_function(self, sample_data):
        """Test decision function."""
        X, y = sample_data
        learner = MLPClassifierLearner(n_features=36, n_epochs=20, random_state=42)
        learner.train(X, y, verbose=False)

        decisions = learner.decision_function(X[:10])

        assert decisions.shape == (10,)

    def test_witness_returns_none(self, sample_data):
        """Test that witness methods return None (pure classifier)."""
        X, y = sample_data
        learner = MLPClassifierLearner(n_features=36, n_epochs=10, random_state=42)
        learner.train(X, y, verbose=False)

        assert learner.get_witness_operator() is None
        assert learner.get_sparse_witness() is None
        assert learner.get_measurement_cost() == -1

    def test_save_load(self, sample_data, tmp_path):
        """Test model serialization."""
        X, y = sample_data
        learner = MLPClassifierLearner(n_features=36, n_epochs=20, random_state=42)
        learner.train(X, y, verbose=False)

        # Save
        save_path = tmp_path / "model.pt"
        learner.save(str(save_path))

        # Load into new learner
        learner2 = MLPClassifierLearner(n_features=36)
        learner2.load(str(save_path))

        # Compare predictions
        pred1 = learner.predict(X[:10])
        pred2 = learner2.predict(X[:10])

        assert np.array_equal(pred1, pred2)

    def test_early_stopping(self, sample_data):
        """Test that early stopping works."""
        X, y = sample_data
        learner = MLPClassifierLearner(
            n_features=36,
            n_epochs=1000,  # High max epochs
            patience=5,    # Low patience
            random_state=42
        )

        learner.train(X, y, verbose=False)

        # Should have stopped before 1000 epochs
        assert len(learner.training_history) < 1000

    def test_metrics_include_n_parameters(self, sample_data):
        """Test that metrics include parameter count."""
        X, y = sample_data
        learner = MLPClassifierLearner(n_features=36, n_epochs=10, random_state=42)
        metrics = learner.train(X, y, verbose=False)

        assert 'n_parameters' in metrics
        assert metrics['n_parameters'] > 0

    def test_predict_before_train_raises_error(self):
        """Test that prediction before training raises error."""
        learner = MLPClassifierLearner(n_features=36)
        X = np.random.randn(10, 36)

        with pytest.raises(RuntimeError, match="Model must be trained"):
            learner.predict(X)

    def test_predict_proba_before_train_raises_error(self):
        """Test that predict_proba before training raises error."""
        learner = MLPClassifierLearner(n_features=36)
        X = np.random.randn(10, 36)

        with pytest.raises(RuntimeError, match="Model must be trained"):
            learner.predict_proba(X)

    def test_decision_function_before_train_raises_error(self):
        """Test that decision_function before training raises error."""
        learner = MLPClassifierLearner(n_features=36)
        X = np.random.randn(10, 36)

        with pytest.raises(RuntimeError, match="Model must be trained"):
            learner.decision_function(X)


class TestMLPWithQuantumData:
    """Integration tests with quantum state features."""

    def test_with_pauli_features(self):
        """Test with actual Pauli feature dimensions."""
        from src.feature_extraction.pauli_features import extract_features_batch
        from src.quantum_states.state_generation import generate_distillability_dataset

        # Generate small dataset
        states, labels = generate_distillability_dataset(
            n_samples=100,
            noise_range=(0.0, 0.5),
            seed=42
        )

        # Extract features
        basis = create_sparse_measurement_set(3, 'two_body')
        features = extract_features_batch(states, basis, verbose=False)

        # Train classifier
        learner = MLPClassifierLearner(
            n_features=len(basis),
            n_epochs=30,
            random_state=42
        )

        metrics = learner.train(features, np.array(labels), verbose=False)

        # Should achieve reasonable accuracy
        assert metrics['test_accuracy'] > 0.55

    def test_predictions_consistent_with_probabilities(self):
        """Test that predictions are consistent with highest probability class."""
        from src.quantum_states.state_generation import generate_distillability_dataset
        from src.feature_extraction.pauli_features import extract_features_batch

        states, labels = generate_distillability_dataset(
            n_samples=100,
            noise_range=(0.0, 0.5),
            seed=42
        )

        basis = create_sparse_measurement_set(3, 'two_body')
        features = extract_features_batch(states, basis, verbose=False)

        learner = MLPClassifierLearner(
            n_features=len(basis),
            n_epochs=20,
            random_state=42
        )
        learner.train(features, np.array(labels), verbose=False)

        # Get predictions and probabilities
        predictions = learner.predict(features[:20])
        probs = learner.predict_proba(features[:20])

        # Predictions should match argmax of probabilities
        expected_predictions = probs.argmax(axis=1)
        assert np.array_equal(predictions, expected_predictions)
