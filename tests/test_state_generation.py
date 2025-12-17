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
    partial_transpose,
    check_npt_any_bipartition,
    generate_noisy_cluster_state,
    generate_3qubit_product_state,
    generate_distillability_dataset,
    _permute_qubits
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


class TestNPTOracleAndDistillability:
    """Test NPT oracle and 3-qubit distillability functions."""

    def test_permute_qubits_identity(self):
        """Test that identity permutation leaves state unchanged."""
        ghz = generate_entangled_state(3, 'ghz', noise_level=0.0)
        rho_data = np.asarray(ghz.data)

        # Identity permutation
        rho_permuted = _permute_qubits(rho_data, [0, 1, 2])

        assert np.allclose(rho_data, rho_permuted), "Identity permutation should not change state"

    def test_permute_qubits_swap(self):
        """Test that permutation preserves trace and Hermiticity."""
        ghz = generate_entangled_state(3, 'ghz', noise_level=0.1, seed=42)
        rho_data = np.asarray(ghz.data)

        # Swap qubits 0 and 1
        rho_permuted = _permute_qubits(rho_data, [1, 0, 2])

        # Check trace preserved
        assert np.isclose(np.trace(rho_permuted), 1.0), "Trace should be 1"

        # Check Hermiticity preserved
        assert np.allclose(rho_permuted, rho_permuted.conj().T), "Should be Hermitian"

        # Check positive semidefinite
        eigenvalues = np.linalg.eigvalsh(rho_permuted)
        assert np.all(eigenvalues >= -1e-10), "Should be positive semidefinite"

    def test_npt_pure_ghz_distillable(self):
        """Pure GHZ state should be distillable (NPT)."""
        ghz = generate_entangled_state(3, 'ghz', noise_level=0.0)

        is_distillable = check_npt_any_bipartition(ghz)

        assert is_distillable, "Pure 3-qubit GHZ state should be distillable (NPT)"

    def test_npt_pure_w_distillable(self):
        """Pure W state should be distillable (NPT)."""
        w_state = generate_entangled_state(3, 'w', noise_level=0.0)

        is_distillable = check_npt_any_bipartition(w_state)

        assert is_distillable, "Pure 3-qubit W state should be distillable (NPT)"

    def test_npt_pure_cluster_distillable(self):
        """Pure cluster state should be distillable (NPT)."""
        cluster = generate_noisy_cluster_state(n_qubits=3, noise_level=0.0)

        is_distillable = check_npt_any_bipartition(cluster)

        assert is_distillable, "Pure 3-qubit cluster state should be distillable (NPT)"

    def test_npt_product_state_not_distillable(self):
        """Product states should NOT be distillable (PPT across all bipartitions)."""
        for seed in [42, 123, 456]:
            product = generate_3qubit_product_state(seed=seed)

            is_distillable = check_npt_any_bipartition(product)

            assert not is_distillable, f"Product state (seed={seed}) should NOT be distillable"

    def test_npt_noisy_ghz_threshold(self):
        """Noisy GHZ states should transition from distillable to non-distillable."""
        distillable_count = 0
        non_distillable_count = 0

        # Test various noise levels
        for noise in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            ghz = generate_entangled_state(3, 'ghz', noise_level=noise, seed=42)
            if check_npt_any_bipartition(ghz):
                distillable_count += 1
            else:
                non_distillable_count += 1

        # Low noise should be distillable, high noise should not
        assert distillable_count > 0, "Some low-noise GHZ states should be distillable"
        assert non_distillable_count > 0, "Some high-noise GHZ states should NOT be distillable"

    def test_cluster_state_valid_density_matrix(self):
        """Cluster state should be a valid density matrix."""
        for noise in [0.0, 0.2, 0.5]:
            cluster = generate_noisy_cluster_state(n_qubits=3, noise_level=noise)

            # Check dimension
            assert cluster.dim == 8, f"3-qubit state should have dim 8, got {cluster.dim}"

            # Check trace = 1
            assert np.isclose(np.trace(cluster.data), 1.0), "Trace should be 1"

            # Check Hermiticity
            assert np.allclose(cluster.data, cluster.data.conj().T), "Should be Hermitian"

            # Check positive semidefinite
            eigenvalues = np.linalg.eigvalsh(cluster.data)
            assert np.all(eigenvalues >= -1e-10), "Should be positive semidefinite"

    def test_cluster_state_pure_when_no_noise(self):
        """Cluster state without noise should be pure (rank 1)."""
        cluster = generate_noisy_cluster_state(n_qubits=3, noise_level=0.0)

        eigenvalues = np.linalg.eigvalsh(cluster.data)
        rank = np.sum(eigenvalues > 1e-10)

        assert rank == 1, f"Pure cluster state should have rank 1, got {rank}"

    def test_product_state_valid_density_matrix(self):
        """Product state should be a valid density matrix."""
        for seed in [42, 123, 456]:
            product = generate_3qubit_product_state(seed=seed)

            # Check dimension
            assert product.dim == 8, f"3-qubit state should have dim 8"

            # Check trace = 1
            assert np.isclose(np.trace(product.data), 1.0), "Trace should be 1"

            # Check Hermiticity
            assert np.allclose(product.data, product.data.conj().T), "Should be Hermitian"

            # Check pure (rank 1)
            eigenvalues = np.linalg.eigvalsh(product.data)
            rank = np.sum(eigenvalues > 1e-10)
            assert rank == 1, f"Product state should be pure (rank 1), got rank {rank}"

    def test_distillability_dataset_sizes(self):
        """Test distillability dataset generation produces correct sizes."""
        n_samples = 100
        states, labels = generate_distillability_dataset(n_samples=n_samples, seed=42)

        assert len(states) == n_samples, f"Expected {n_samples} states, got {len(states)}"
        assert len(labels) == n_samples, f"Expected {n_samples} labels, got {len(labels)}"

        # Check all states are 3-qubit
        for state in states:
            assert state.dim == 8, "All states should be 3-qubit (dim=8)"

    def test_distillability_dataset_label_balance(self):
        """Test that dataset has reasonable class balance."""
        n_samples = 200
        states, labels = generate_distillability_dataset(
            n_samples=n_samples,
            noise_range=(0.0, 0.3),  # Lower noise → more distillable
            seed=42
        )

        n_distillable = np.sum(labels)
        n_non_distillable = len(labels) - n_distillable

        # Should have both classes represented
        assert n_distillable > 0, "Should have some distillable states"
        assert n_non_distillable > 0, "Should have some non-distillable states"

        # Product states (1/5 of dataset) are always non-distillable
        assert n_non_distillable >= n_samples // 5, "At least product states should be non-distillable"

    def test_distillability_dataset_reproducibility(self):
        """Test that dataset generation is reproducible with same seed."""
        states1, labels1 = generate_distillability_dataset(n_samples=50, seed=42)
        states2, labels2 = generate_distillability_dataset(n_samples=50, seed=42)

        # Labels should be identical
        assert np.array_equal(labels1, labels2), "Same seed should produce same labels"

        # States should be identical
        for s1, s2 in zip(states1, states2):
            assert np.allclose(s1.data, s2.data), "Same seed should produce same states"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
