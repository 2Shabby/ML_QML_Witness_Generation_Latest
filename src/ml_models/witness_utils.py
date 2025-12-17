"""
Witness Utilities

Common utilities for witness operator extraction and manipulation,
shared between SVM and Transformer witness learners.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from qiskit.quantum_info import PauliList, SparsePauliOp
import logging

logger = logging.getLogger(__name__)


def build_witness_operator(
    pauli_basis: PauliList,
    coefficients: np.ndarray,
    threshold: float = 1e-10
) -> Optional[SparsePauliOp]:
    """
    Build a witness operator from Pauli basis and coefficients.

    W = Σₖ wₖ Pₖ where Pₖ are Pauli basis operators and wₖ are coefficients.

    Args:
        pauli_basis: PauliList of basis operators
        coefficients: Array of coefficients (same length as pauli_basis)
        threshold: Minimum absolute coefficient value to include

    Returns:
        SparsePauliOp representing the witness, or None if all coefficients are zero
    """
    if len(pauli_basis) != len(coefficients):
        raise ValueError(
            f"Mismatch: {len(pauli_basis)} Paulis but {len(coefficients)} coefficients"
        )

    pauli_strings = []
    nonzero_coeffs = []

    for pauli, weight in zip(pauli_basis, coefficients):
        if abs(weight) > threshold:
            pauli_strings.append(str(pauli))
            nonzero_coeffs.append(float(weight))

    if len(pauli_strings) == 0:
        logger.warning("All witness coefficients are below threshold!")
        return None

    logger.info(f"Witness operator: {len(pauli_strings)} non-zero terms")

    return SparsePauliOp(pauli_strings, coeffs=nonzero_coeffs)


def get_sparse_witness(
    witness_operator: SparsePauliOp,
    threshold: float = 0.01
) -> SparsePauliOp:
    """
    Get a sparse version of a witness by thresholding small coefficients.

    This is useful for reducing measurement cost by focusing on the
    most important Pauli terms.

    Args:
        witness_operator: Full witness operator
        threshold: Coefficients with |w_k| < threshold are set to zero

    Returns:
        Sparse witness operator with only significant terms
    """
    pauli_strings = []
    coefficients = []

    for pauli, coeff in witness_operator.to_list():
        if abs(coeff) >= threshold:
            pauli_strings.append(pauli)
            coefficients.append(coeff)

    n_original = len(witness_operator.to_list())
    n_sparse = len(pauli_strings)

    logger.info(f"Sparse witness: {n_sparse}/{n_original} terms (threshold={threshold})")

    if len(pauli_strings) == 0:
        logger.warning(
            f"No coefficients exceed threshold {threshold}. "
            f"Returning full witness operator."
        )
        return witness_operator

    return SparsePauliOp(pauli_strings, coeffs=coefficients)


def evaluate_witness_on_states(
    witness_operator: SparsePauliOp,
    states: List[np.ndarray],
    bias: float = 0.0
) -> np.ndarray:
    """
    Evaluate witness expectation values on a list of density matrices.

    Computes Tr(W·ρ) for each state ρ.

    Args:
        witness_operator: The witness operator W
        states: List of density matrices
        bias: Optional bias term to add

    Returns:
        Array of witness values (one per state)
    """
    witness_matrix = witness_operator.to_matrix()
    witness_values = np.array([
        np.trace(rho @ witness_matrix).real + bias
        for rho in states
    ])
    return witness_values


def get_witness_coefficients_dict(witness_operator: SparsePauliOp) -> Dict[str, float]:
    """
    Extract witness coefficients as a dictionary mapping Pauli strings to values.

    Args:
        witness_operator: The witness operator

    Returns:
        Dictionary {pauli_string: coefficient}
    """
    return {str(pauli): float(np.real(coeff)) for pauli, coeff in witness_operator.to_list()}


def get_top_coefficients(
    witness_operator: SparsePauliOp,
    n: int = 10
) -> List[Tuple[str, float]]:
    """
    Get the top N Pauli terms by absolute coefficient magnitude.

    Args:
        witness_operator: The witness operator
        n: Number of top terms to return

    Returns:
        List of (pauli_string, coefficient) tuples sorted by |coefficient|
    """
    coeffs_dict = get_witness_coefficients_dict(witness_operator)
    sorted_coeffs = sorted(
        [(k, abs(v)) for k, v in coeffs_dict.items()],
        key=lambda x: x[1],
        reverse=True
    )
    return sorted_coeffs[:n]


def categorize_pauli_terms(witness_operator: SparsePauliOp) -> Dict[str, List[Tuple[str, float]]]:
    """
    Categorize witness terms by Pauli weight (1-body vs 2-body vs higher).

    Args:
        witness_operator: The witness operator

    Returns:
        Dictionary with keys 'one_body', 'two_body', 'higher' containing
        lists of (pauli_string, coefficient) tuples
    """
    categories = {'one_body': [], 'two_body': [], 'higher': []}

    for pauli, coeff in witness_operator.to_list():
        pauli_str = str(pauli)
        n_non_identity = sum(1 for c in pauli_str if c != 'I')
        real_coeff = float(np.real(coeff))

        if n_non_identity == 1:
            categories['one_body'].append((pauli_str, real_coeff))
        elif n_non_identity == 2:
            categories['two_body'].append((pauli_str, real_coeff))
        else:
            categories['higher'].append((pauli_str, real_coeff))

    return categories


def compute_term_importance(witness_operator: SparsePauliOp) -> Dict[str, float]:
    """
    Compute the relative importance of different term categories.

    Returns the fraction of total absolute coefficient weight in each category.

    Args:
        witness_operator: The witness operator

    Returns:
        Dictionary with importance fractions for each category
    """
    categories = categorize_pauli_terms(witness_operator)

    one_body_total = sum(abs(c) for _, c in categories['one_body'])
    two_body_total = sum(abs(c) for _, c in categories['two_body'])
    higher_total = sum(abs(c) for _, c in categories['higher'])
    total = one_body_total + two_body_total + higher_total

    if total == 0:
        return {'one_body': 0.0, 'two_body': 0.0, 'higher': 0.0}

    return {
        'one_body': one_body_total / total,
        'two_body': two_body_total / total,
        'higher': higher_total / total
    }


__all__ = [
    'build_witness_operator',
    'get_sparse_witness',
    'evaluate_witness_on_states',
    'get_witness_coefficients_dict',
    'get_top_coefficients',
    'categorize_pauli_terms',
    'compute_term_importance',
]
