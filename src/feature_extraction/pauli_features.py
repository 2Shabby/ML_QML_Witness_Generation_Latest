"""
Pauli feature extraction from quantum states.

This module implements feature extraction based on Pauli operator expectations,
following the framework described in the ML/QML witness generation document.
"""

import numpy as np
from typing import List, Tuple, Optional
from qiskit.quantum_info import Pauli, PauliList, DensityMatrix
import logging

logger = logging.getLogger(__name__)


def get_pauli_basis(n_qubits: int, include_identity: bool = False) -> PauliList:
    """
    Generate the complete Pauli basis for n qubits.

    Args:
        n_qubits: Number of qubits
        include_identity: Whether to include the identity operator

    Returns:
        PauliList containing all Pauli operators
    """
    # Generate all 4^n Pauli strings
    pauli_labels = []

    # Create all combinations of I, X, Y, Z
    basis_ops = ['I', 'X', 'Y', 'Z']

    def generate_paulis(n):
        if n == 0:
            return ['']
        smaller = generate_paulis(n - 1)
        result = []
        for op in basis_ops:
            for s in smaller:
                result.append(op + s)
        return result

    pauli_labels = generate_paulis(n_qubits)

    # Remove identity if not requested
    if not include_identity:
        pauli_labels = [p for p in pauli_labels if p != 'I' * n_qubits]

    return PauliList(pauli_labels)


def extract_pauli_features(rho: DensityMatrix, pauli: Pauli) -> float:
    """
    Extract a single Pauli feature: Tr(ρ P).

    Args:
        rho: Density matrix
        pauli: Pauli operator

    Returns:
        Expectation value Tr(ρ P)
    """
    pauli_matrix = pauli.to_matrix()
    rho_data = rho.data

    # Compute Tr(ρ P)
    expectation = np.trace(rho_data @ pauli_matrix)

    # Should be real for valid density matrices and Hermitian operators
    return np.real(expectation)


def extract_features_batch(states: List[DensityMatrix],
                           pauli_basis: PauliList,
                           verbose: bool = False) -> np.ndarray:
    """
    Extract Pauli features for a batch of states.

    Args:
        states: List of density matrices
        pauli_basis: PauliList of operators to measure
        verbose: Whether to print progress

    Returns:
        Feature matrix of shape (n_states, n_paulis)
    """
    n_states = len(states)
    n_features = len(pauli_basis)

    features = np.zeros((n_states, n_features))

    if verbose:
        logger.info(f"Extracting {n_features} features from {n_states} states...")

    for i, state in enumerate(states):
        for j, pauli in enumerate(pauli_basis):
            features[i, j] = extract_pauli_features(state, pauli)

        if verbose and (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{n_states} states")

    return features


def paulis_commute(p1: Pauli, p2: Pauli) -> bool:
    """
    Check if two Pauli operators commute.

    Two Pauli strings commute if and only if they anticommute in an even
    number of positions (including 0).

    Args:
        p1: First Pauli operator
        p2: Second Pauli operator

    Returns:
        True if they commute, False otherwise
    """
    # Get the Pauli strings
    s1 = p1.to_label()
    s2 = p2.to_label()

    if len(s1) != len(s2):
        raise ValueError("Pauli operators must have same length")

    # Count anticommuting positions
    anticommute_count = 0

    for c1, c2 in zip(s1, s2):
        # I commutes with everything
        if c1 == 'I' or c2 == 'I':
            continue

        # Same operators commute
        if c1 == c2:
            continue

        # Different non-identity operators anticommute
        anticommute_count += 1

    # Commute if even number of anticommutations
    return anticommute_count % 2 == 0


def group_commuting_paulis(pauli_list: PauliList) -> List[List[int]]:
    """
    Group Pauli operators into mutually commuting sets.

    Uses a greedy algorithm to partition operators into measurement settings.

    Args:
        pauli_list: List of Pauli operators

    Returns:
        List of groups, where each group is a list of indices
    """
    n_paulis = len(pauli_list)
    groups = []
    assigned = [False] * n_paulis

    for i in range(n_paulis):
        if assigned[i]:
            continue

        # Start new group with this Pauli
        current_group = [i]
        assigned[i] = True

        # Try to add more Paulis that commute with all in current group
        for j in range(i + 1, n_paulis):
            if assigned[j]:
                continue

            # Check if j commutes with all in current group
            commutes_with_all = True
            for k in current_group:
                if not paulis_commute(pauli_list[k], pauli_list[j]):
                    commutes_with_all = False
                    break

            if commutes_with_all:
                current_group.append(j)
                assigned[j] = True

        groups.append(current_group)

    return groups


def estimate_measurement_cost(pauli_list: PauliList) -> int:
    """
    Estimate the number of measurement settings needed.

    The cost is the number of mutually commuting groups needed to cover
    all operators.

    Args:
        pauli_list: List of Pauli operators

    Returns:
        Number of measurement settings (groups) needed
    """
    groups = group_commuting_paulis(pauli_list)
    return len(groups)


def create_sparse_measurement_set(n_qubits: int, strategy: str = 'single_body') -> PauliList:
    """
    Create a sparse set of Pauli measurements for incomplete tomography.

    Args:
        n_qubits: Number of qubits
        strategy: Strategy for selecting measurements
            - 'single_body': Only single-qubit Paulis (X_i, Y_i, Z_i)
            - 'two_body': Single and two-qubit Paulis
            - 'random': Random subset

    Returns:
        PauliList of selected operators
    """
    if strategy == 'single_body':
        # Only single-qubit Paulis
        pauli_labels = []
        for i in range(n_qubits):
            for op in ['X', 'Y', 'Z']:
                label = ['I'] * n_qubits
                label[i] = op
                pauli_labels.append(''.join(label))

    elif strategy == 'two_body':
        # Single and two-qubit Paulis
        pauli_labels = []

        # Single-qubit terms
        for i in range(n_qubits):
            for op in ['X', 'Y', 'Z']:
                label = ['I'] * n_qubits
                label[i] = op
                pauli_labels.append(''.join(label))

        # Two-qubit terms (only adjacent or all pairs)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                for op1 in ['X', 'Y', 'Z']:
                    for op2 in ['X', 'Y', 'Z']:
                        label = ['I'] * n_qubits
                        label[i] = op1
                        label[j] = op2
                        pauli_labels.append(''.join(label))

    elif strategy == 'random':
        # Random subset of full basis
        full_basis = get_pauli_basis(n_qubits, include_identity=False)
        # Select ~sqrt(N) random operators
        n_select = min(int(np.sqrt(len(full_basis))), len(full_basis))
        indices = np.random.choice(len(full_basis), n_select, replace=False)
        pauli_labels = [full_basis[i].to_label() for i in indices]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return PauliList(pauli_labels)
