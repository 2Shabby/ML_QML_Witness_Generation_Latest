"""
Tests for Variational POVM Learner

Tests the parameterized unitary construction, POVM validity,
and end-to-end training pipeline.
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
    from src.ml_models.variational_povm import (
        ParameterizedUnitary,
        VariationalPOVMClassifier,
        VariationalPOVMLearner,
    )
except ImportError:
    torch_available = False

pytestmark = pytest.mark.skipif(
    not torch_available,
    reason="PyTorch not installed"
)


class TestParameterizedUnitary:
    """Test the parameterized unitary construction."""

    def test_unitary_shape(self):
        """Constructed matrix should be 8x8 for 3 qubits."""
        pu = ParameterizedUnitary(n_qubits=3, n_layers=2)
        U = pu.build_unitary()
        assert U.shape == (8, 8)

    def test_unitarity(self):
        """U^dag U should equal identity."""
        pu = ParameterizedUnitary(n_qubits=3, n_layers=4)
        U = pu.build_unitary()
        U_dag = U.conj().T
        product = (U_dag @ U).detach().numpy()
        np.testing.assert_allclose(
            product, np.eye(8), atol=1e-5,
            err_msg="U^dag U != I: constructed matrix is not unitary"
        )

    def test_unitarity_reverse(self):
        """U U^dag should also equal identity."""
        pu = ParameterizedUnitary(n_qubits=3, n_layers=4)
        U = pu.build_unitary()
        U_dag = U.conj().T
        product = (U @ U_dag).detach().numpy()
        np.testing.assert_allclose(
            product, np.eye(8), atol=1e-5,
            err_msg="U U^dag != I"
        )

    def test_deterministic_with_seed(self):
        """Same seed should produce same unitary."""
        torch.manual_seed(42)
        pu1 = ParameterizedUnitary(n_qubits=3, n_layers=2)
        U1 = pu1.build_unitary().detach().numpy()

        torch.manual_seed(42)
        pu2 = ParameterizedUnitary(n_qubits=3, n_layers=2)
        U2 = pu2.build_unitary().detach().numpy()

        np.testing.assert_allclose(U1, U2, atol=1e-7)

    def test_different_n_layers(self):
        """Different layer counts should all produce valid unitaries."""
        for n_layers in [1, 2, 4, 8]:
            pu = ParameterizedUnitary(n_qubits=3, n_layers=n_layers)
            U = pu.build_unitary()
            product = (U.conj().T @ U).detach().numpy()
            np.testing.assert_allclose(
                product, np.eye(8), atol=1e-5,
                err_msg=f"Not unitary with n_layers={n_layers}"
            )

    def test_gradients_flow(self):
        """Gradients should flow through the unitary construction."""
        pu = ParameterizedUnitary(n_qubits=3, n_layers=2)
        U = pu.build_unitary()
        # Use a scalar loss
        loss = U.abs().sum()
        loss.backward()
        assert pu.angles.grad is not None
        assert not torch.all(pu.angles.grad == 0)

    def test_probability_output(self):
        """Forward pass should produce valid probabilities."""
        pu = ParameterizedUnitary(n_qubits=3, n_layers=2)

        # Create a batch of density matrices (identity / 8 = maximally mixed)
        batch_size = 4
        dim = 8
        rho = torch.eye(dim, dtype=torch.complex64).unsqueeze(0) / dim
        rho_batch = rho.expand(batch_size, -1, -1)

        probs = pu(rho_batch)

        assert probs.shape == (batch_size, dim)
        # Probabilities should sum to 1
        np.testing.assert_allclose(
            probs.detach().sum(dim=1).numpy(),
            np.ones(batch_size),
            atol=1e-5
        )
        # All probabilities should be non-negative
        assert torch.all(probs >= 0)

    def test_pure_state_probabilities(self):
        """For a pure state |0>, probabilities should be valid."""
        pu = ParameterizedUnitary(n_qubits=3, n_layers=2)

        # |000> state
        dim = 8
        rho = torch.zeros(1, dim, dim, dtype=torch.complex64)
        rho[0, 0, 0] = 1.0

        probs = pu(rho)
        assert probs.shape == (1, dim)
        np.testing.assert_allclose(probs.detach().sum().item(), 1.0, atol=1e-5)


class TestVariationalPOVMClassifier:
    """Test the end-to-end classifier model."""

    def test_forward_shape(self):
        """Forward pass should produce (batch, 2) logits."""
        model = VariationalPOVMClassifier(n_qubits=3, n_layers=2)
        batch_size = 8
        dim = 8
        rho = torch.eye(dim, dtype=torch.complex64).unsqueeze(0) / dim
        rho_batch = rho.expand(batch_size, -1, -1)

        logits = model(rho_batch)
        assert logits.shape == (batch_size, 2)

    def test_parameter_count(self):
        """Check parameter count is reasonable."""
        model = VariationalPOVMClassifier(n_qubits=3, n_layers=4)
        n_params = sum(p.numel() for p in model.parameters())
        # 4 layers * 3 qubits * 2 angles = 24 (unitary)
        # + 8*2 + 2 = 18 (linear classifier)
        assert n_params == 24 + 18


class TestVariationalPOVMLearner:
    """Test the learner wrapper."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic density matrix dataset."""
        np.random.seed(42)
        n_samples = 200
        dim = 8

        rho_list = []
        labels = []
        for i in range(n_samples):
            # Random density matrix via partial trace of random pure state
            psi = np.random.randn(dim) + 1j * np.random.randn(dim)
            psi /= np.linalg.norm(psi)
            rho = np.outer(psi, psi.conj())
            rho_list.append(rho)
            # Label based on whether first diagonal element is above threshold
            labels.append(1 if rho[0, 0].real > 1.0 / dim else 0)

        return np.array(rho_list), np.array(labels)

    def test_train_returns_metrics(self, synthetic_data):
        """Training should return a metrics dictionary."""
        X_rho, y = synthetic_data

        learner = VariationalPOVMLearner(
            n_qubits=3,
            n_layers=2,
            n_epochs=10,
            patience=5,
            random_state=42,
        )

        metrics = learner.train(X_rho, y, test_size=0.2, verbose=False)

        assert 'test_accuracy' in metrics
        assert 'test_precision' in metrics
        assert 'test_recall' in metrics
        assert 'n_parameters' in metrics
        assert 'measurement_settings' in metrics
        assert metrics['measurement_settings'] == 1
        assert learner.is_trained

    def test_predict(self, synthetic_data):
        """Predictions should be 0/1 with correct shape."""
        X_rho, y = synthetic_data

        learner = VariationalPOVMLearner(
            n_qubits=3, n_layers=2, n_epochs=5, random_state=42
        )
        learner.train(X_rho, y, verbose=False)

        preds = learner.predict(X_rho[:10])
        assert preds.shape == (10,)
        assert set(preds).issubset({0, 1})

    def test_predict_proba(self, synthetic_data):
        """Probability predictions should be valid."""
        X_rho, y = synthetic_data

        learner = VariationalPOVMLearner(
            n_qubits=3, n_layers=2, n_epochs=5, random_state=42
        )
        learner.train(X_rho, y, verbose=False)

        probs = learner.predict_proba(X_rho[:10])
        assert probs.shape == (10, 2)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_povm_elements(self, synthetic_data):
        """Extracted POVM elements should be valid."""
        X_rho, y = synthetic_data

        learner = VariationalPOVMLearner(
            n_qubits=3, n_layers=2, n_epochs=5, random_state=42
        )
        learner.train(X_rho, y, verbose=False)

        elements = learner.get_povm_elements()
        assert len(elements) == 8
        assert elements[0].shape == (8, 8)

        # Completeness: sum M_k = I
        total = sum(elements)
        np.testing.assert_allclose(total, np.eye(8), atol=1e-5)

        # Positivity: all eigenvalues >= 0
        for M_k in elements:
            eigvals = np.linalg.eigvalsh(M_k)
            assert np.all(eigvals > -1e-10), f"Negative eigenvalue: {eigvals.min()}"

    def test_verify_povm(self, synthetic_data):
        """POVM verification should pass."""
        X_rho, y = synthetic_data

        learner = VariationalPOVMLearner(
            n_qubits=3, n_layers=2, n_epochs=5, random_state=42
        )
        learner.train(X_rho, y, verbose=False)

        verification = learner.verify_povm()
        assert verification['is_valid']
        assert verification['completeness_error'] < 1e-5

    def test_measurement_cost(self, synthetic_data):
        """Measurement cost should be 1."""
        X_rho, y = synthetic_data

        learner = VariationalPOVMLearner(
            n_qubits=3, n_layers=2, n_epochs=5, random_state=42
        )
        learner.train(X_rho, y, verbose=False)

        assert learner.get_measurement_cost() == 1

    def test_training_history(self, synthetic_data):
        """Training history should be recorded."""
        X_rho, y = synthetic_data

        learner = VariationalPOVMLearner(
            n_qubits=3, n_layers=2, n_epochs=10, random_state=42
        )
        learner.train(X_rho, y, verbose=False)

        assert len(learner.training_history) > 0
        assert 'epoch' in learner.training_history[0]
        assert 'train_loss' in learner.training_history[0]
        assert 'val_acc' in learner.training_history[0]

    def test_predict_before_train_raises(self):
        """Predicting before training should raise."""
        learner = VariationalPOVMLearner(n_qubits=3, n_layers=2)
        rho = np.random.randn(5, 8, 8) + 1j * np.random.randn(5, 8, 8)

        with pytest.raises(RuntimeError, match="trained"):
            learner.predict(rho)


class TestPOVMIntegration:
    """Integration tests with actual quantum state data."""

    def test_with_quantum_states(self):
        """Test with actual generated quantum states."""
        from src.quantum_states.state_generation import generate_distillability_dataset

        states, labels = generate_distillability_dataset(n_samples=100, seed=42)
        labels = np.array(labels)

        # Convert DensityMatrix objects to numpy array
        rho_array = np.array([np.array(state) for state in states])

        learner = VariationalPOVMLearner(
            n_qubits=3,
            n_layers=2,
            n_epochs=20,
            batch_size=16,
            random_state=42,
        )

        metrics = learner.train(rho_array, labels, test_size=0.2, verbose=False)

        # Should achieve above-random accuracy on this small dataset
        assert metrics['test_accuracy'] > 0.4

        # POVM should be valid
        verification = learner.verify_povm()
        assert verification['is_valid']
