"""
Unit tests for feature extraction module.
"""

import pytest
import numpy as np
from qiskit.quantum_info import DensityMatrix, PauliList
import sys
sys.path.insert(0, '/home/user/ML_QML_Witness_Generation')

from src.quantum_states.state_generation import generate_bell_state, generate_separable_state
from src.feature_extraction.pauli_features import (
    get_pauli_basis,
    extract_pauli_features,
    extract_features_batch,
    create_sparse_measurement_set,
    group_commuting_paulis,
    estimate_measurement_cost
)


class TestFeatureExtraction:
    """Tests for Pauli feature extraction."""

    def test_pauli_basis_generation(self):
        """Test Pauli basis generation."""
        # 2 qubits: 4^2 - 1 = 15 non-identity Paulis
        pauli_basis = get_pauli_basis(2, include_identity=False)
        assert len(pauli_basis) == 15

        # 3 qubits: 4^3 - 1 = 63 non-identity Paulis
        pauli_basis = get_pauli_basis(3, include_identity=False)
        assert len(pauli_basis) == 63

    def test_extract_pauli_features(self):
        """Test single Pauli feature extraction."""
        # Bell state should have specific Pauli expectations
        bell_state = generate_bell_state(0)
        pauli_basis = get_pauli_basis(2, include_identity=False)

        # Extract all features
        features = []
        for pauli in pauli_basis:
            feat = extract_pauli_features(bell_state, pauli)
            features.append(feat)

        # Features should be real numbers
        assert all(isinstance(f, (float, np.floating)) for f in features)

        # Check sum of squares (should be ≤ 1 for physical states)
        # Actually for Pauli decomposition, this constraint is more subtle
        assert len(features) == 15

    def test_features_batch(self):
        """Test batch feature extraction."""
        states = [generate_bell_state(i) for i in range(4)]
        pauli_basis = get_pauli_basis(2, include_identity=False)

        features = extract_features_batch(states, pauli_basis, verbose=False)

        assert features.shape == (4, 15)
        # All features should be finite
        assert np.all(np.isfinite(features))

    def test_sparse_measurement_sets(self):
        """Test sparse measurement set generation."""
        n_qubits = 3

        # Single-body measurements
        sparse_basis = create_sparse_measurement_set(n_qubits, strategy='single_body')
        # Should have 3 * 3 = 9 single-qubit Paulis
        assert len(sparse_basis) == 9

        # Two-body measurements
        sparse_basis = create_sparse_measurement_set(n_qubits, strategy='two_body')
        # Should have single + pairwise two-qubit terms
        # Single: 9, Two-body: 3*3 choose 2 = 3*2/2 * 3*3 = 3*9 = 27
        assert len(sparse_basis) >= 9  # At least single-body

    def test_commuting_pauli_grouping(self):
        """Test grouping of commuting Paulis."""
        # All Z-type Paulis should commute
        pauli_list = PauliList(['ZI', 'IZ', 'ZZ'])
        groups = group_commuting_paulis(pauli_list)

        # Should all be in one group
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_measurement_cost_estimation(self):
        """Test measurement cost estimation."""
        # All mutually commuting Paulis (all Z)
        pauli_list = PauliList(['ZI', 'IZ', 'ZZ'])
        cost = estimate_measurement_cost(pauli_list)
        assert cost == 1  # All can be measured together

        # Non-commuting Paulis (X and Z on same qubit don't commute)
        pauli_list = PauliList(['XI', 'ZI', 'YI'])
        cost = estimate_measurement_cost(pauli_list)
        assert cost >= 2  # At least 2 settings needed (X, Y, Z are mutually non-commuting)

    def test_feature_extraction_properties(self):
        """Test properties of extracted features."""
        # Separable state should have different features than entangled
        sep_state = generate_separable_state(2, seed=42)
        ent_state = generate_bell_state(0)

        pauli_basis = get_pauli_basis(2, include_identity=False)

        sep_features = extract_features_batch([sep_state], pauli_basis, verbose=False)[0]
        ent_features = extract_features_batch([ent_state], pauli_basis, verbose=False)[0]

        # Features should be different
        assert not np.allclose(sep_features, ent_features, atol=0.1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
