"""
Partial-transpose and NPT machinery for 3-qubit states.

The dataset builders live in :mod:`src.quantum_states.balanced_dataset`;
this module keeps only the label-oracle primitives they (and the tests)
depend on: partial transpose, qubit permutation, and the
NPT-across-any-bipartition check used as the distillability label.
"""

import numpy as np
from qiskit.quantum_info import DensityMatrix
from typing import List


def _permute_qubits(rho: np.ndarray, perm: List[int]) -> np.ndarray:
    """
    Permute qubit ordering in density matrix.

    Args:
        rho: Density matrix as numpy array (d x d where d = 2^n_qubits)
        perm: Permutation of qubit indices, e.g., [1, 0, 2] swaps qubits 0 and 1

    Returns:
        Permuted density matrix
    """
    n_qubits = len(perm)
    # Reshape to tensor with separate index for each qubit (row and column)
    # rho[i0,i1,...,in-1; j0,j1,...,jn-1] -> rho_tensor[i0,i1,...,in-1,j0,j1,...,jn-1]
    shape = [2] * (2 * n_qubits)
    rho_tensor = rho.reshape(shape)

    # Permute both row indices (first n_qubits dims) and column indices (last n_qubits dims)
    perm_full = perm + [p + n_qubits for p in perm]
    rho_permuted = np.transpose(rho_tensor, perm_full)

    return rho_permuted.reshape(2**n_qubits, 2**n_qubits)


def check_npt_any_bipartition(rho: DensityMatrix) -> bool:
    """
    Check if state is NPT (negative partial transpose) across ANY bipartition.

    For 3 qubits, checks bipartitions:
      - A|BC (qubit 0 vs qubits 1,2): dims=[2,4]
      - B|AC (qubit 1 vs qubits 0,2): requires permutation, then dims=[2,4]
      - C|AB (qubit 2 vs qubits 0,1): dims=[4,2]

    A state that is NPT across any bipartition is distillable (can be used
    to extract pure entanglement via LOCC).

    Args:
        rho: 3-qubit density matrix

    Returns:
        True if NPT across ANY bipartition (proxy for distillable)
        False if PPT across ALL bipartitions (may be bound entangled or separable)
    """
    rho_data = np.asarray(rho.data)

    # Tolerance for numerical eigenvalue checks
    tol = -1e-10

    # Bipartition A|BC: partial transpose on subsystem A (dims=[2,4])
    # Subsystem A has dimension 2, subsystem BC has dimension 4
    rho_pt_A = partial_transpose(rho_data, dims=[2, 4], axis=0)
    if np.min(np.linalg.eigvalsh(rho_pt_A)) < tol:
        return True

    # Bipartition B|AC: need to permute qubits (0,1,2) -> (1,0,2)
    # Then subsystem B (now first) has dim 2, subsystem AC has dim 4
    rho_permuted_B = _permute_qubits(rho_data, [1, 0, 2])
    rho_pt_B = partial_transpose(rho_permuted_B, dims=[2, 4], axis=0)
    if np.min(np.linalg.eigvalsh(rho_pt_B)) < tol:
        return True

    # Bipartition C|AB: partial transpose on subsystem C (dims=[4,2])
    # Subsystem AB has dimension 4, subsystem C has dimension 2
    rho_pt_C = partial_transpose(rho_data, dims=[4, 2], axis=1)
    if np.min(np.linalg.eigvalsh(rho_pt_C)) < tol:
        return True

    return False  # PPT across all bipartitions


def partial_transpose(rho: np.ndarray, dims: List[int], axis: int = 1) -> np.ndarray:
    """
    Compute the partial transpose of a density matrix.

    Args:
        rho: Density matrix as numpy array
        dims: Dimensions of subsystems [dim_A, dim_B]
        axis: Which subsystem to transpose (0 or 1)

    Returns:
        Partially transposed density matrix
    """
    dim_A, dim_B = dims

    if axis == 0:
        # Partial transpose on first subsystem
        rho_reshaped = rho.reshape(dim_A, dim_B, dim_A, dim_B)
        rho_pt = rho_reshaped.transpose(2, 1, 0, 3)
    else:
        # Partial transpose on second subsystem
        rho_reshaped = rho.reshape(dim_A, dim_B, dim_A, dim_B)
        rho_pt = rho_reshaped.transpose(0, 3, 2, 1)

    return rho_pt.reshape(dim_A * dim_B, dim_A * dim_B)
