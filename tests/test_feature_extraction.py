"""
Unit tests for feature extraction module.
"""

import numpy as np
from qiskit.quantum_info import PauliList

from src.feature_extraction.pauli_features import (
    get_pauli_basis,
    create_sparse_measurement_set,
    group_commuting_paulis,
    estimate_measurement_cost,
)


class TestFeatureExtraction:
    """Test feature extraction functions."""

    def test_pauli_basis_generation(self):
        """Test Pauli basis generation."""
        n_qubits = 2

        # Without identity
        basis = get_pauli_basis(n_qubits, include_identity=False)
        assert len(basis) == 4**n_qubits - 1  # 15 for 2 qubits

        # With identity
        basis_with_id = get_pauli_basis(n_qubits, include_identity=True)
        assert len(basis_with_id) == 4**n_qubits  # 16 for 2 qubits

    def test_sparse_measurement_sets(self):
        """Test sparse measurement set creation."""
        n_qubits = 3

        # Local measurements only
        local_set = create_sparse_measurement_set(n_qubits, strategy='local')
        assert len(local_set) == 3 * n_qubits  # 3 Paulis per qubit

        # Two-body correlations
        two_body_set = create_sparse_measurement_set(n_qubits, strategy='two_body')
        # Should have 3*n + 3*n*(n-1)/2 * 3^2 terms
        expected_single = 3 * n_qubits
        expected_two_body = 9 * (n_qubits * (n_qubits - 1) // 2)
        assert len(two_body_set) == expected_single + expected_two_body

        # Random measurements
        n_measurements = 10
        random_set = create_sparse_measurement_set(
            n_qubits, strategy='random', n_measurements=n_measurements
        )
        assert len(random_set) == n_measurements

    def test_commuting_pauli_grouping(self):
        """Test grouping of commuting Paulis."""
        pauli_list = PauliList(['XX', 'XI', 'IX', 'ZZ', 'ZI', 'IZ'])

        groups = group_commuting_paulis(pauli_list)

        # Check all Paulis are assigned
        all_indices = set()
        for group in groups:
            all_indices.update(group)
        assert all_indices == set(range(len(pauli_list)))

        # Check mutual commutativity within groups
        for group in groups:
            for i in group:
                for j in group:
                    assert pauli_list[i].commutes(pauli_list[j])

    def test_measurement_cost_estimation(self):
        """Test measurement cost estimation."""
        # All mutually commuting Paulis (all Z)
        pauli_list = PauliList(['ZI', 'IZ', 'ZZ'])
        cost = estimate_measurement_cost(pauli_list)
        assert cost == 1  # All can be measured together

        # XX, YY, ZZ also all commute (can be measured together)
        pauli_list = PauliList(['XX', 'YY', 'ZZ'])
        cost = estimate_measurement_cost(pauli_list)
        assert cost == 1  # All can be measured together

        # Truly non-commuting Paulis on same qubit
        pauli_list = PauliList(['XI', 'YI', 'ZI'])
        cost = estimate_measurement_cost(pauli_list)
        assert cost == 3  # All three are mutually non-commuting, need 3 settings

        # Mixed case: some commute, some don't
        pauli_list = PauliList(['XI', 'ZI', 'IX'])
        cost = estimate_measurement_cost(pauli_list)
        assert cost == 2  # XI-ZI don't commute, but IX commutes with ZI
