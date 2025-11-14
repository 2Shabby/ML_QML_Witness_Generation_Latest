"""
Unit tests for quantum state generation module.
"""

import pytest
import numpy as np
from qiskit.quantum_info import DensityMatrix

from src.quantum_states.state_generation import (
    generate_random_density_matrix,
    generate_separable_state,
    generate_entangled_state,
    generate_bell_state,
    generate_werner_state,
    generate_dataset,
    check_ppt_criterion,
    partial_transpose
)


class TestStateGeneration:
    """Test state generation functions."""

    def test_random_density_matrix(self):
        """Test random density matrix generation."""
        n_qubits = 2
        rho = generate_random_density_matrix(n_qubits, seed=42)

        # Check it's a valid density matrix
        assert isinstance(rho, DensityMatrix)
        assert rho.dim == 2 ** n_qubits

        # Check Hermiticity
        assert np.allclose(rho.data, rho.data.conj().T)

        # Check trace = 1
        assert np.isclose(np.trace(rho.data), 1.0)

        # Check positive semidefinite
        eigenvalues = np.linalg.eigvalsh(rho.data)
        assert np.all(eigenvalues >= -1e-10)

    def test_separable_state(self):
        """Test separable state generation."""
        n_qubits = 2
        rho = generate_separable_state(n_qubits, seed=42)

        # Check it's a valid density matrix
        assert isinstance(rho, DensityMatrix)
        assert rho.dim == 2 ** n_qubits

        # For 2 qubits, PPT criterion is necessary and sufficient
        # So separable states should be PPT
        is_ppt = check_ppt_criterion(rho, dims=[2, 2])
        assert is_ppt, "Separable state should satisfy PPT criterion"

    def test_bell_states(self):
        """Test Bell state generation."""
        for bell_type in range(4):
            rho = generate_bell_state(bell_type)

            # Check it's pure (rank 1)
            eigenvalues = np.linalg.eigvalsh(rho.data)
            rank = np.sum(eigenvalues > 1e-10)
            assert rank == 1, f"Bell state should be pure, got rank {rank}"

            # Check it's entangled (not PPT)
            is_ppt = check_ppt_criterion(rho, dims=[2, 2])
            assert not is_ppt, f"Bell state {bell_type} should be entangled"

    def test_werner_state(self):
        """Test Werner state generation."""
        n_qubits = 2

        # Pure singlet (p=1) should be entangled
        rho_pure = generate_werner_state(n_qubits, p=1.0)
        is_ppt_pure = check_ppt_criterion(rho_pure, dims=[2, 2])
        assert not is_ppt_pure, "Pure Werner state should be entangled"

        # Maximally mixed (p=0) should be separable
        rho_mixed = generate_werner_state(n_qubits, p=0.0)
        is_ppt_mixed = check_ppt_criterion(rho_mixed, dims=[2, 2])
        assert is_ppt_mixed, "Maximally mixed state should be separable"

    def test_entangled_state_types(self):
        """Test different entangled state types."""
        n_qubits = 2

        for ent_type in ['random', 'bell', 'ghz']:
            if ent_type == 'bell' and n_qubits != 2:
                continue

            rho = generate_entangled_state(n_qubits, entanglement_type=ent_type, seed=42)

            assert isinstance(rho, DensityMatrix)
            assert rho.dim == 2 ** n_qubits

    def test_dataset_generation(self):
        """Test dataset generation."""
        n_qubits = 2
        n_samples = 50
        entangled_fraction = 0.6

        states, labels = generate_dataset(
            n_qubits=n_qubits,
            n_samples=n_samples,
            entangled_fraction=entangled_fraction,
            seed=42
        )

        # Check sizes
        assert len(states) == n_samples
        assert len(labels) == n_samples

        # Check label distribution
        n_entangled = np.sum(labels == 1)
        n_separable = np.sum(labels == 0)
        assert n_entangled == int(n_samples * entangled_fraction)
        assert n_separable == n_samples - n_entangled

        # Check all states are valid
        for state in states:
            assert isinstance(state, DensityMatrix)
            assert state.dim == 2 ** n_qubits

    def test_partial_transpose(self):
        """Test partial transpose computation."""
        # Test on a known entangled state (Bell state)
        rho_bell = generate_bell_state(bell_type=0)

        # Partial transpose should have negative eigenvalue
        rho_pt = partial_transpose(rho_bell.data, dims=[2, 2], axis=1)
        eigenvalues = np.linalg.eigvalsh(rho_pt)

        # Bell state should have one negative eigenvalue after PT
        assert np.any(eigenvalues < -1e-10), "Bell state PT should have negative eigenvalue"

    def test_ppt_criterion(self):
        """Test PPT criterion function."""
        # Separable state should be PPT
        rho_sep = generate_separable_state(n_qubits=2, seed=42)
        assert check_ppt_criterion(rho_sep, dims=[2, 2])

        # Bell state should not be PPT
        rho_bell = generate_bell_state(bell_type=0)
        assert not check_ppt_criterion(rho_bell, dims=[2, 2])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
