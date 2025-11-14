"""
Unit tests for quantum state generation.
"""

import pytest
import numpy as np
from qiskit.quantum_info import DensityMatrix
import sys
sys.path.insert(0, '/home/user/ML_QML_Witness_Generation')

from src.quantum_states.state_generation import (
    generate_random_density_matrix,
    generate_separable_state,
    generate_bell_state,
    generate_werner_state,
    generate_entangled_state,
    generate_dataset,
    partial_transpose,
    check_ppt_criterion
)


class TestStateGeneration:
    """Tests for quantum state generation."""

    def test_random_density_matrix(self):
        """Test random density matrix generation."""
        rho = generate_random_density_matrix(2, seed=42)

        # Check it's a valid density matrix
        assert rho.is_valid()
        assert np.allclose(np.trace(rho.data), 1.0)
        assert np.allclose(rho.data, rho.data.conj().T)  # Hermitian

        # Check positive semidefinite
        eigenvalues = np.linalg.eigvalsh(rho.data)
        assert np.all(eigenvalues >= -1e-10)

    def test_separable_state(self):
        """Test separable state generation."""
        rho = generate_separable_state(2, seed=42)

        assert rho.is_valid()
        assert np.allclose(np.trace(rho.data), 1.0)

    def test_bell_states(self):
        """Test Bell state generation."""
        for i in range(4):
            bell = generate_bell_state(i)

            # Check valid
            assert bell.is_valid()
            assert np.allclose(np.trace(bell.data), 1.0)

            # Bell states should be pure (rank 1)
            eigenvalues = np.linalg.eigvalsh(bell.data)
            rank = np.sum(eigenvalues > 1e-10)
            assert rank == 1

    def test_werner_state(self):
        """Test Werner state generation."""
        rho = generate_werner_state(2, p=0.5, seed=42)

        assert rho.is_valid()
        assert np.allclose(np.trace(rho.data), 1.0)

    def test_entangled_state_types(self):
        """Test different entangled state types."""
        for ent_type in ['ghz', 'w', 'random']:
            rho = generate_entangled_state(2, entanglement_type=ent_type, seed=42)

            assert rho.is_valid()
            assert np.allclose(np.trace(rho.data), 1.0)

    def test_dataset_generation(self):
        """Test dataset generation."""
        states, labels = generate_dataset(
            n_qubits=2,
            n_samples=50,
            entangled_fraction=0.5,
            seed=42
        )

        assert len(states) == 50
        assert len(labels) == 50
        assert np.sum(labels == 1) + np.sum(labels == 0) == 50
        # Check roughly correct fraction
        assert 20 <= np.sum(labels == 1) <= 30

    def test_partial_transpose(self):
        """Test partial transpose operation."""
        # Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        bell = generate_bell_state(0)
        rho = bell.data

        # Partial transpose
        rho_pt = partial_transpose(rho, (2, 2), system=1)

        # Should be Hermitian
        assert np.allclose(rho_pt, rho_pt.conj().T)

        # Should have unit trace
        assert np.isclose(np.trace(rho_pt), 1.0)

    def test_ppt_criterion(self):
        """Test PPT criterion for entanglement."""
        # Bell states should be NPT (fail PPT)
        bell = generate_bell_state(0)
        is_ppt = check_ppt_criterion(bell, dims=(2, 2))

        # Bell state is entangled, so should be NPT
        assert not is_ppt

        # Separable states should be PPT
        sep = generate_separable_state(2, seed=42)
        is_ppt = check_ppt_criterion(sep, dims=(2, 2))

        # Separable should pass PPT
        assert is_ppt


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
