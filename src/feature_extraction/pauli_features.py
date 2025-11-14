"""
Pauli Feature Extraction Module

Implements the Generalized Bloch Vector representation described in Section 3.1
of the framework document.

For n-qubits: ρ = Σₖ rₖ Pₖ where {Pₖ} is the Pauli basis
Feature vector: x_ρ = (r₁, r₂, ..., r_{d²-1}) where rₖ = Tr(ρ Pₖ)
"""

import numpy as np
from qiskit.quantum_info import DensityMatrix, Pauli, PauliList
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def get_pauli_basis(n_qubits: int, include_identity: bool = False) -> PauliList:
    """
    Generate the complete Pauli basis for n qubits.

    For n qubits, there are 4ⁿ Pauli strings (including identity).
    The basis without identity has 4ⁿ - 1 elements.

    Args:
        n_qubits: Number of qubits
        include_identity: Whether to include identity operator

    Returns:
        PauliList containing all Pauli basis operators
    """
    # Generate all Pauli strings
    pauli_labels = []

    for i in range(4 ** n_qubits):
        label = ""
        num = i
        for _ in range(n_qubits):
            pauli_idx = num % 4
            pauli_chars = ['I', 'X', 'Y', 'Z']
            label = pauli_chars[pauli_idx] + label
            num //= 4

        # Skip identity if not requested
        if not include_identity and label == 'I' * n_qubits:
            continue

        pauli_labels.append(label)

    return PauliList(pauli_labels)


def extract_pauli_features(
    rho: DensityMatrix,
    pauli_basis: Optional[PauliList] = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Extract Pauli feature vector from a density matrix.

    Following Section 3.1.2: x_ρ = (Tr(ρ P₁), Tr(ρ P₂), ..., Tr(ρ P_{N}))

    Args:
        rho: Density matrix to extract features from
        pauli_basis: Pauli basis to use (if None, generated automatically)
        normalize: Whether to normalize by √d for proper Bloch vector

    Returns:
        Feature vector as numpy array
    """
    # Get number of qubits from density matrix dimension
    dim = rho.dim
    n_qubits = int(np.log2(dim))

    if pauli_basis is None:
        pauli_basis = get_pauli_basis(n_qubits, include_identity=False)

    features = []

    for pauli in pauli_basis:
        # Compute expectation value: Tr(ρ P)
        expectation = np.trace(rho.data @ pauli.to_matrix()).real

        # Normalize if requested (Pauli operators have trace=0 except I)
        if normalize and pauli.to_label() != 'I' * n_qubits:
            expectation /= np.sqrt(dim)

        features.append(expectation)

    return np.array(features, dtype=np.float64)


def extract_features_batch(
    states: List[DensityMatrix],
    pauli_basis: Optional[PauliList] = None,
    normalize: bool = True,
    verbose: bool = True
) -> np.ndarray:
    """
    Extract Pauli features from a batch of states.

    Following Section 10.6 data pipeline approach.

    Args:
        states: List of density matrices
        pauli_basis: Pauli basis to use (computed once, reused for all)
        normalize: Whether to normalize features
        verbose: Whether to log progress

    Returns:
        Feature matrix of shape (n_samples, n_features)
    """
    if len(states) == 0:
        raise ValueError("states list is empty")

    # Get basis from first state if not provided
    if pauli_basis is None:
        dim = states[0].dim
        n_qubits = int(np.log2(dim))
        pauli_basis = get_pauli_basis(n_qubits, include_identity=False)

    if verbose:
        logger.info(f"Extracting features from {len(states)} states...")
        logger.info(f"Feature dimension: {len(pauli_basis)}")

    features_list = []

    for i, state in enumerate(states):
        if verbose and (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(states)} states")

        features = extract_pauli_features(state, pauli_basis, normalize)
        features_list.append(features)

    feature_matrix = np.array(features_list)

    if verbose:
        logger.info(f"Feature extraction complete. Shape: {feature_matrix.shape}")

    return feature_matrix


def create_sparse_measurement_set(
    n_qubits: int,
    strategy: str = 'local',
    n_measurements: Optional[int] = None
) -> PauliList:
    """
    Create a sparse set of measurements for incomplete tomography.

    Following Section 3.1.3 and Failure Mode 4 approach.

    Args:
        n_qubits: Number of qubits
        strategy: Strategy for selecting measurements:
            - 'local': Only single-qubit Paulis
            - 'two_body': All two-qubit correlations
            - 'random': Random subset of Pauli strings
        n_measurements: Number of measurements (if None, use strategy default)

    Returns:
        PauliList of selected measurements
    """
    if strategy == 'local':
        # Only measure single-qubit observables: Zᵢ, Xᵢ, Yᵢ
        pauli_labels = []
        for i in range(n_qubits):
            for pauli_char in ['X', 'Y', 'Z']:
                label = 'I' * i + pauli_char + 'I' * (n_qubits - i - 1)
                pauli_labels.append(label)
        return PauliList(pauli_labels)

    elif strategy == 'two_body':
        # Include all single-qubit and two-qubit correlations
        pauli_labels = []

        # Single-qubit terms
        for i in range(n_qubits):
            for pauli_char in ['X', 'Y', 'Z']:
                label = 'I' * i + pauli_char + 'I' * (n_qubits - i - 1)
                pauli_labels.append(label)

        # Two-qubit terms
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                for p1 in ['X', 'Y', 'Z']:
                    for p2 in ['X', 'Y', 'Z']:
                        label = ['I'] * n_qubits
                        label[i] = p1
                        label[j] = p2
                        pauli_labels.append(''.join(label))

        return PauliList(pauli_labels)

    elif strategy == 'random':
        # Random subset
        full_basis = get_pauli_basis(n_qubits, include_identity=False)

        if n_measurements is None:
            n_measurements = min(3 * n_qubits, len(full_basis))

        # Randomly select measurements
        indices = np.random.choice(len(full_basis), size=n_measurements, replace=False)
        return PauliList([full_basis[i] for i in indices])

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def group_commuting_paulis(pauli_list: PauliList) -> List[List[int]]:
    """
    Group Pauli operators into mutually commuting sets for co-measurement.

    Following Section 3.4.1 for measurement decomposition.

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

        # Start new group with pauli i
        group = [i]
        assigned[i] = True

        # Find all Paulis that commute with all members of current group
        for j in range(i + 1, n_paulis):
            if assigned[j]:
                continue

            # Check if j commutes with all members of group
            commutes_with_all = True
            for k in group:
                if not pauli_list[j].commutes(pauli_list[k]):
                    commutes_with_all = False
                    break

            if commutes_with_all:
                group.append(j)
                assigned[j] = True

        groups.append(group)

    return groups


def estimate_measurement_cost(pauli_list: PauliList) -> int:
    """
    Estimate the number of measurement settings required.

    Following Section 7.3 efficiency metrics.

    Args:
        pauli_list: List of Pauli operators in witness

    Returns:
        Number of measurement settings (M)
    """
    groups = group_commuting_paulis(pauli_list)
    return len(groups)
