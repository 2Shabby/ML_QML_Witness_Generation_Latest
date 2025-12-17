"""
Tests for Transformer Witness Learner

Tests both TransformerClassifier and HybridTransformerWitness models.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Skip all tests if torch is not available
torch_available = True
try:
    import torch
    from src.ml_models.transformer_witness import (
        TransformerWitnessLearner,
        TransformerClassifier,
        HybridTransformerWitness,
        PositionalEncoding,
        TransformerBlock
    )
except ImportError:
    torch_available = False

from src.feature_extraction.pauli_features import create_sparse_measurement_set

pytestmark = pytest.mark.skipif(
    not torch_available,
    reason="PyTorch not installed"
)


class TestTransformerComponents:
    """Test individual transformer components."""

    def test_positional_encoding_shape(self):
        """Test positional encoding output shape."""
        d_model = 64
        seq_len = 36
        batch_size = 8

        pe = PositionalEncoding(d_model=d_model, max_len=100)
        x = torch.randn(batch_size, seq_len, d_model)

        output = pe(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_transformer_block_output(self):
        """Test transformer block output shape and attention weights."""
        d_model = 64
        n_heads = 4
        d_ff = 128
        seq_len = 37  # 36 features + 1 CLS token
        batch_size = 8

        block = TransformerBlock(d_model, n_heads, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)

        output, attention = block(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attention.shape == (batch_size, n_heads, seq_len, seq_len)


class TestTransformerClassifier:
    """Test TransformerClassifier model."""

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        n_features = 36
        batch_size = 8
        d_model = 64

        model = TransformerClassifier(
            n_features=n_features,
            d_model=d_model,
            n_heads=4,
            n_layers=2
        )

        x = torch.randn(batch_size, n_features)
        logits = model(x)

        assert logits.shape == (batch_size, 2)

    def test_attention_weights_available(self):
        """Test that attention weights are stored after forward pass."""
        model = TransformerClassifier(n_features=36, n_layers=2)
        x = torch.randn(4, 36)

        _ = model(x)
        attention = model.get_attention_weights()

        assert attention is not None
        assert len(attention) == 2  # n_layers attention matrices


class TestHybridTransformerWitness:
    """Test HybridTransformerWitness model."""

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        n_features = 36
        batch_size = 8

        model = HybridTransformerWitness(
            n_features=n_features,
            d_model=64,
            n_heads=4,
            n_layers=2
        )

        x = torch.randn(batch_size, n_features)
        logits = model(x)

        assert logits.shape == (batch_size, 2)

    def test_witness_coefficients(self):
        """Test that witness coefficients are generated."""
        n_features = 36
        batch_size = 8

        model = HybridTransformerWitness(n_features=n_features)
        x = torch.randn(batch_size, n_features)

        _ = model(x)

        assert model.witness_coefficients is not None
        assert model.witness_coefficients.shape == (batch_size, n_features)

    def test_mean_coefficients(self):
        """Test mean coefficient extraction."""
        n_features = 36
        batch_size = 16

        model = HybridTransformerWitness(n_features=n_features)
        x = torch.randn(batch_size, n_features)

        mean_coeffs = model.get_mean_coefficients(x)

        assert mean_coeffs.shape == (n_features,)
        assert isinstance(mean_coeffs, np.ndarray)


class TestTransformerWitnessLearner:
    """Test TransformerWitnessLearner wrapper class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        n_samples = 200
        n_features = 36

        # Create synthetic data
        X = np.random.randn(n_samples, n_features)
        # Labels based on simple rule for testing
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        basis = create_sparse_measurement_set(3, 'two_body')

        return X, y, basis

    def test_classifier_mode_training(self, sample_data):
        """Test training in classifier mode."""
        X, y, basis = sample_data

        learner = TransformerWitnessLearner(
            pauli_basis=basis,
            mode='classifier',
            d_model=32,
            n_heads=2,
            n_layers=1,
            n_epochs=10,
            patience=5,
            random_state=42
        )

        metrics = learner.train(X, y, test_size=0.2, verbose=False)

        assert 'test_accuracy' in metrics
        assert 'n_parameters' in metrics
        assert learner.is_trained

    def test_hybrid_mode_training(self, sample_data):
        """Test training in hybrid mode."""
        X, y, basis = sample_data

        learner = TransformerWitnessLearner(
            pauli_basis=basis,
            mode='hybrid',
            d_model=32,
            n_heads=2,
            n_layers=1,
            n_epochs=10,
            patience=5,
            random_state=42
        )

        metrics = learner.train(X, y, test_size=0.2, verbose=False)

        assert 'test_accuracy' in metrics
        assert learner.is_trained

    def test_prediction(self, sample_data):
        """Test prediction after training."""
        X, y, basis = sample_data

        learner = TransformerWitnessLearner(
            pauli_basis=basis,
            mode='classifier',
            d_model=32,
            n_heads=2,
            n_layers=1,
            n_epochs=5,
            random_state=42
        )

        learner.train(X, y, test_size=0.2, verbose=False)

        predictions = learner.predict(X[:10])

        assert predictions.shape == (10,)
        assert set(predictions).issubset({0, 1})

    def test_predict_proba(self, sample_data):
        """Test probability prediction."""
        X, y, basis = sample_data

        learner = TransformerWitnessLearner(
            pauli_basis=basis,
            mode='classifier',
            d_model=32,
            n_epochs=5,
            random_state=42
        )

        learner.train(X, y, test_size=0.2, verbose=False)

        probs = learner.predict_proba(X[:10])

        assert probs.shape == (10, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_witness_extraction_hybrid(self, sample_data):
        """Test witness extraction in hybrid mode."""
        X, y, basis = sample_data

        learner = TransformerWitnessLearner(
            pauli_basis=basis,
            mode='hybrid',
            d_model=32,
            n_heads=2,
            n_layers=1,
            n_epochs=10,
            random_state=42
        )

        learner.train(X, y, test_size=0.2, verbose=False)

        witness = learner.get_witness_operator()

        assert witness is not None
        # Witness should have some non-zero terms
        assert len(witness.to_list()) > 0

    def test_witness_extraction_fails_classifier(self, sample_data):
        """Test that witness extraction fails in classifier mode."""
        X, y, basis = sample_data

        learner = TransformerWitnessLearner(
            pauli_basis=basis,
            mode='classifier',
            d_model=32,
            n_epochs=5,
            random_state=42
        )

        learner.train(X, y, test_size=0.2, verbose=False)

        with pytest.raises(ValueError, match="hybrid mode"):
            learner.get_witness_operator()

    def test_sparse_witness(self, sample_data):
        """Test sparse witness extraction."""
        X, y, basis = sample_data

        learner = TransformerWitnessLearner(
            pauli_basis=basis,
            mode='hybrid',
            d_model=32,
            n_epochs=10,
            random_state=42
        )

        learner.train(X, y, test_size=0.2, verbose=False)

        sparse_witness = learner.get_sparse_witness(threshold=0.1)

        # Sparse witness should have fewer or equal terms
        full_witness = learner.get_witness_operator()
        assert len(sparse_witness.to_list()) <= len(full_witness.to_list())

    def test_decision_function(self, sample_data):
        """Test decision function computation."""
        X, y, basis = sample_data

        learner = TransformerWitnessLearner(
            pauli_basis=basis,
            mode='hybrid',
            d_model=32,
            n_epochs=5,
            random_state=42
        )

        learner.train(X, y, test_size=0.2, verbose=False)

        decisions = learner.decision_function(X[:10])

        assert decisions.shape == (10,)

    def test_training_history(self, sample_data):
        """Test that training history is recorded."""
        X, y, basis = sample_data

        learner = TransformerWitnessLearner(
            pauli_basis=basis,
            mode='classifier',
            d_model=32,
            n_epochs=10,
            random_state=42
        )

        learner.train(X, y, test_size=0.2, verbose=False)

        assert len(learner.training_history) > 0
        assert 'epoch' in learner.training_history[0]
        assert 'train_loss' in learner.training_history[0]
        assert 'val_acc' in learner.training_history[0]


class TestTransformerIntegration:
    """Integration tests with quantum state data."""

    def test_with_quantum_features(self):
        """Test with actual quantum state features."""
        from src.quantum_states.state_generation import generate_distillability_dataset
        from src.feature_extraction.pauli_features import extract_features_batch

        # Generate small dataset
        states, labels = generate_distillability_dataset(n_samples=100, seed=42)
        labels = np.array(labels)

        basis = create_sparse_measurement_set(3, 'two_body')
        features = extract_features_batch(states, basis, verbose=False)

        # Train hybrid transformer
        learner = TransformerWitnessLearner(
            pauli_basis=basis,
            mode='hybrid',
            d_model=32,
            n_heads=2,
            n_layers=1,
            n_epochs=20,
            batch_size=16,
            random_state=42
        )

        metrics = learner.train(features, labels, test_size=0.2, verbose=False)

        # Should achieve reasonable accuracy (above random)
        assert metrics['test_accuracy'] > 0.4

        # Should be able to extract witness
        witness = learner.get_witness_operator()
        assert len(witness.to_list()) > 0

    def test_comparable_to_svm(self):
        """Test that transformer is comparable to SVM on quantum data."""
        from src.quantum_states.state_generation import generate_distillability_dataset
        from src.feature_extraction.pauli_features import extract_features_batch
        from src.ml_models.svm_witness import SVMWitnessLearner

        # Generate dataset
        states, labels = generate_distillability_dataset(n_samples=200, seed=42)
        labels = np.array(labels)

        basis = create_sparse_measurement_set(3, 'two_body')
        features = extract_features_batch(states, basis, verbose=False)

        # Train SVM
        svm_learner = SVMWitnessLearner(
            pauli_basis=basis,
            C=1.0,
            kernel='linear',
            random_state=42
        )
        svm_metrics = svm_learner.train(features, labels, test_size=0.2, verbose=False)

        # Train Transformer
        tf_learner = TransformerWitnessLearner(
            pauli_basis=basis,
            mode='hybrid',
            d_model=64,
            n_heads=4,
            n_layers=2,
            n_epochs=50,
            batch_size=32,
            random_state=42
        )
        tf_metrics = tf_learner.train(features, labels, test_size=0.2, verbose=False)

        # Transformer should be within reasonable range of SVM
        # (might be worse due to small dataset, but shouldn't be dramatically worse)
        accuracy_gap = svm_metrics['test_accuracy'] - tf_metrics['test_accuracy']
        assert accuracy_gap < 0.2, f"Transformer too far below SVM: gap={accuracy_gap:.4f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
