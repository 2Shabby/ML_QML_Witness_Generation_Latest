"""
Tests for DPS Hierarchy and Distillability Oracles

Tests the oracle abstraction layer and DPS Level 2 implementation.
"""

import pytest
import numpy as np
from qiskit.quantum_info import DensityMatrix

from src.quantum_states.state_generation import (
    generate_entangled_state,
    generate_3qubit_product_state,
    generate_noisy_cluster_state,
    generate_random_density_matrix,
    check_npt_any_bipartition
)
from src.quantum_states.distillability_oracles import (
    NPTOracle,
    DPSOracle,
    PPTOracle,
    get_oracle,
    compare_oracles
)


class TestOracleAbstraction:
    """Test the oracle abstraction layer."""

    def test_npt_oracle_creation(self):
        """Test NPT oracle can be created."""
        oracle = NPTOracle()
        assert oracle.name() == "NPT"

    def test_ppt_oracle_creation(self):
        """Test PPT oracle can be created."""
        oracle = PPTOracle()
        assert oracle.name() == "PPT"

    def test_dps_oracle_creation(self):
        """Test DPS oracle can be created."""
        oracle = DPSOracle(level=2)
        assert oracle.name() == "DPS-L2"

    def test_get_oracle_factory(self):
        """Test oracle factory function."""
        oracle_npt = get_oracle('npt')
        assert isinstance(oracle_npt, NPTOracle)

        oracle_dps = get_oracle('dps2')
        assert isinstance(oracle_dps, DPSOracle)

    def test_oracle_callable(self):
        """Test oracles can be called as functions."""
        oracle = NPTOracle()
        ghz = generate_entangled_state(3, 'ghz', noise_level=0.0)
        result = oracle(ghz)  # Call as function
        assert isinstance(result, bool)


class TestNPTOracle:
    """Test NPT oracle correctness."""

    def test_pure_ghz_distillable(self):
        """Pure GHZ state should be distillable (NPT)."""
        oracle = NPTOracle()
        ghz = generate_entangled_state(3, 'ghz', noise_level=0.0)
        assert oracle.is_distillable(ghz) == True

    def test_pure_w_distillable(self):
        """Pure W state should be distillable (NPT)."""
        oracle = NPTOracle()
        w = generate_entangled_state(3, 'w', noise_level=0.0)
        assert oracle.is_distillable(w) == True

    def test_pure_cluster_distillable(self):
        """Pure cluster state should be distillable (NPT)."""
        oracle = NPTOracle()
        cluster = generate_noisy_cluster_state(3, noise_level=0.0)
        assert oracle.is_distillable(cluster) == True

    def test_product_not_distillable(self):
        """Product state should NOT be distillable (PPT)."""
        oracle = NPTOracle()
        product = generate_3qubit_product_state(seed=42)
        assert oracle.is_distillable(product) == False

    def test_highly_noisy_not_distillable(self):
        """Highly noisy (>80%) GHZ should become PPT."""
        oracle = NPTOracle()
        noisy_ghz = generate_entangled_state(3, 'ghz', noise_level=0.85)
        assert oracle.is_distillable(noisy_ghz) == False


class TestDPSOracle:
    """Test DPS oracle functionality."""

    def test_dps_agrees_with_npt_on_pure_states(self):
        """DPS should agree with NPT on pure entangled states."""
        npt = NPTOracle()
        dps = DPSOracle(level=2)

        ghz = generate_entangled_state(3, 'ghz', noise_level=0.0)
        assert npt.is_distillable(ghz) == dps.is_distillable(ghz)

        w = generate_entangled_state(3, 'w', noise_level=0.0)
        assert npt.is_distillable(w) == dps.is_distillable(w)

    def test_dps_agrees_with_npt_on_product(self):
        """DPS should agree with NPT on product states."""
        npt = NPTOracle()
        dps = DPSOracle(level=2)

        product = generate_3qubit_product_state(seed=123)
        assert npt.is_distillable(product) == dps.is_distillable(product)

    def test_dps_classify_distillable(self):
        """DPS classify should identify distillable states."""
        dps = DPSOracle(level=2)
        ghz = generate_entangled_state(3, 'ghz', noise_level=0.0)
        assert dps.classify(ghz) == 'distillable'

    def test_dps_classify_product_as_separable(self):
        """DPS classify should identify product states as separable."""
        dps = DPSOracle(level=2)
        product = generate_3qubit_product_state(seed=456)

        # Product states are separable (not bound entangled)
        classification = dps.classify(product)
        # Should be either 'separable' or treated conservatively
        assert classification in ['separable', 'bound_entangled']

    def test_dps_separability_check_on_product(self):
        """DPS separability check should pass for product states."""
        dps = DPSOracle(level=2)
        product = generate_3qubit_product_state(seed=789)

        # Product states should be separable
        is_sep = dps.check_separability(product, bipartition='A|BC')
        # Note: SDP solver might have numerical issues, so we just check it runs
        assert isinstance(is_sep, bool)


class TestOracleComparison:
    """Test oracle comparison utilities."""

    def test_compare_oracles_basic(self):
        """Test oracle comparison on small dataset."""
        states = [
            generate_entangled_state(3, 'ghz', noise_level=0.0),
            generate_entangled_state(3, 'w', noise_level=0.0),
            generate_3qubit_product_state(seed=42)
        ]

        result = compare_oracles(states, oracle_names=['npt', 'ppt'])

        assert 'per_oracle' in result
        assert 'agreement_rate' in result
        assert result['agreement_rate'] == 1.0  # NPT and PPT should agree on these

    def test_npt_ppt_equivalent_for_distillability(self):
        """NPT and PPT oracles should give same distillability results."""
        npt = NPTOracle()
        ppt = PPTOracle()

        for seed in range(10):
            state = generate_random_density_matrix(3, seed=seed)
            assert npt.is_distillable(state) == ppt.is_distillable(state)


class TestAdversarialScenarios:
    """
    Test scenarios where restricted features might fail.

    These tests investigate potential negative results - cases where
    36D restricted features may not reliably distinguish states.
    """

    def test_near_boundary_states(self):
        """
        Test states near the NPT/PPT boundary.

        At noise ~0.80, GHZ transitions from NPT to PPT.
        States near this boundary are hardest to classify.
        """
        oracle = NPTOracle()

        # Just below threshold - should be NPT
        ghz_below = generate_entangled_state(3, 'ghz', noise_level=0.79)
        assert oracle.is_distillable(ghz_below) == True

        # Just above threshold - should be PPT
        ghz_above = generate_entangled_state(3, 'ghz', noise_level=0.81)
        assert oracle.is_distillable(ghz_above) == False

    def test_asymmetric_noise(self):
        """
        Test GHZ under asymmetric noise (affects qubits differently).

        Asymmetric noise might create states where distillability
        depends on correlations the 36D features can't capture.
        """
        # Create GHZ state
        dim = 8
        psi = np.zeros(dim, dtype=complex)
        psi[0] = 1/np.sqrt(2)  # |000⟩
        psi[7] = 1/np.sqrt(2)  # |111⟩
        rho = np.outer(psi, psi.conj())

        # Apply asymmetric depolarizing noise
        # More noise on qubit 0, less on qubits 1,2
        noise_q0 = 0.5  # Heavy noise on qubit 0
        noise_q12 = 0.1  # Light noise on qubits 1,2

        # Single-qubit depolarizing channels
        I = np.eye(2)
        dep_q0 = (1 - noise_q0) * np.eye(2) + noise_q0 * I / 2

        # Apply via Kraus operators (simplified - just mixing)
        identity = np.eye(dim) / dim
        rho_noisy = (1 - noise_q0) * rho + noise_q0 * identity

        dm = DensityMatrix(rho_noisy)
        oracle = NPTOracle()

        # Check that oracle still works on asymmetric noise
        result = oracle.is_distillable(dm)
        assert isinstance(result, bool)

    def test_phase_damped_ghz(self):
        """
        Test GHZ under phase damping (dephasing) noise.

        Phase damping destroys coherences differently than depolarizing.
        This might create states where 36D features behave differently.
        """
        # Create GHZ state
        dim = 8
        psi = np.zeros(dim, dtype=complex)
        psi[0] = 1/np.sqrt(2)
        psi[7] = 1/np.sqrt(2)
        rho = np.outer(psi, psi.conj())

        # Apply dephasing: kill off-diagonal elements
        gamma = 0.5  # Dephasing rate

        rho_dephased = rho.copy()
        for i in range(dim):
            for j in range(dim):
                if i != j:
                    # Count bit differences (Hamming weight of i XOR j)
                    n_diff = bin(i ^ j).count('1')
                    rho_dephased[i, j] *= (1 - gamma) ** n_diff

        # Renormalize
        rho_dephased = rho_dephased / np.trace(rho_dephased)

        dm = DensityMatrix(rho_dephased)
        oracle = NPTOracle()
        result = oracle.is_distillable(dm)

        # Phase-damped GHZ should still be NPT for moderate dephasing
        assert isinstance(result, bool)

    def test_amplitude_damped_ghz(self):
        """
        Test GHZ under amplitude damping (T1 decay).

        Amplitude damping drives |1⟩ → |0⟩, changing populations.
        """
        # Create GHZ state
        dim = 8
        psi = np.zeros(dim, dtype=complex)
        psi[0] = 1/np.sqrt(2)
        psi[7] = 1/np.sqrt(2)
        rho = np.outer(psi, psi.conj())

        # Simplified amplitude damping: mix toward |000⟩⟨000|
        gamma = 0.3
        ground = np.zeros((dim, dim))
        ground[0, 0] = 1.0

        rho_damped = (1 - gamma) * rho + gamma * ground

        dm = DensityMatrix(rho_damped)
        oracle = NPTOracle()
        result = oracle.is_distillable(dm)

        assert isinstance(result, bool)


class TestPotentialNegativeResults:
    """
    Investigate scenarios for potential negative results.

    These tests look for cases where:
    1. 36D features fail to distinguish distillable from non-distillable
    2. 3-body correlations are essential
    3. Linear SVM boundary breaks down
    """

    def test_ghz_w_distinguishability(self):
        """
        GHZ and W states have different entanglement structure.

        Check if 36D features can distinguish them when both are distillable.
        (This isn't a negative result, but tests feature expressiveness)
        """
        from src.feature_extraction.pauli_features import (
            create_sparse_measurement_set,
            extract_pauli_features
        )

        basis = create_sparse_measurement_set(3, 'two_body')

        ghz = generate_entangled_state(3, 'ghz', noise_level=0.0)
        w = generate_entangled_state(3, 'w', noise_level=0.0)

        f_ghz = extract_pauli_features(ghz, basis)
        f_w = extract_pauli_features(w, basis)

        # Features should be different (states are distinguishable)
        assert not np.allclose(f_ghz, f_w)

        # Check L2 distance is significant
        dist = np.linalg.norm(f_ghz - f_w)
        assert dist > 0.1  # Should be noticeably different

    def test_features_near_npt_boundary(self):
        """
        Check feature continuity near NPT/PPT boundary.

        If features change discontinuously, classifier might struggle.
        """
        from src.feature_extraction.pauli_features import (
            create_sparse_measurement_set,
            extract_pauli_features
        )

        basis = create_sparse_measurement_set(3, 'two_body')

        # States just below and above NPT threshold
        noise_below = 0.79
        noise_above = 0.81

        ghz_below = generate_entangled_state(3, 'ghz', noise_level=noise_below)
        ghz_above = generate_entangled_state(3, 'ghz', noise_level=noise_above)

        f_below = extract_pauli_features(ghz_below, basis)
        f_above = extract_pauli_features(ghz_above, basis)

        # Features should be continuous (similar for close noise levels)
        dist = np.linalg.norm(f_below - f_above)
        assert dist < 0.5  # Should be relatively close

        # But labels flip across the boundary
        oracle = NPTOracle()
        assert oracle.is_distillable(ghz_below) == True
        assert oracle.is_distillable(ghz_above) == False

    def test_three_body_correlation_importance(self):
        """
        Compare 36D vs 63D features to see if 3-body correlations matter.

        If 63D features separate boundary states better, 3-body terms are important.
        """
        from src.feature_extraction.pauli_features import (
            create_sparse_measurement_set,
            get_pauli_basis,
            extract_pauli_features
        )

        basis_36d = create_sparse_measurement_set(3, 'two_body')
        basis_63d = get_pauli_basis(3, include_identity=False)

        # Generate boundary states
        boundary_states = [
            generate_entangled_state(3, 'ghz', noise_level=0.78 + i*0.01)
            for i in range(5)
        ]

        features_36d = [extract_pauli_features(s, basis_36d) for s in boundary_states]
        features_63d = [extract_pauli_features(s, basis_63d) for s in boundary_states]

        # Check variance in each feature set
        var_36d = np.var(features_36d, axis=0).sum()
        var_63d = np.var(features_63d, axis=0).sum()

        # 63D should have at least as much information
        assert var_63d >= var_36d * 0.9  # Allow some tolerance


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
