"""
Quantum state generation for witness testing.

This module provides functions to generate various quantum states including
entangled and separable states for testing witness operators.
"""

import numpy as np
from typing import Tuple, List
from qiskit.quantum_info import DensityMatrix, partial_trace
import logging

logger = logging.getLogger(__name__)


def generate_random_density_matrix(n_qubits: int, rank: int = None, seed: int = None) -> DensityMatrix:
    """
    Generate a random density matrix.

    Args:
        n_qubits: Number of qubits
        rank: Rank of the density matrix (default: full rank)
        seed: Random seed for reproducibility

    Returns:
        Random density matrix
    """
    if seed is not None:
        np.random.seed(seed)

    dim = 2 ** n_qubits
    if rank is None:
        rank = dim

    # Generate random matrix via Ginibre ensemble
    A = np.random.randn(dim, rank) + 1j * np.random.randn(dim, rank)
    rho = A @ A.conj().T
    rho = rho / np.trace(rho)  # Normalize

    return DensityMatrix(rho)


def generate_separable_state(n_qubits: int, seed: int = None) -> DensityMatrix:
    """
    Generate a separable (non-entangled) state.

    Args:
        n_qubits: Number of qubits
        seed: Random seed

    Returns:
        Separable density matrix
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate pure product state
    state_vector = np.ones(2 ** n_qubits, dtype=complex)
    for i in range(n_qubits):
        # Random single-qubit state
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        single_qubit = np.array([np.cos(theta/2), np.exp(1j * phi) * np.sin(theta/2)])

        # Build product state
        if i == 0:
            state_vector = single_qubit
        else:
            state_vector = np.kron(state_vector, single_qubit)

    rho = np.outer(state_vector, state_vector.conj())
    return DensityMatrix(rho)


def generate_bell_state(index: int = 0) -> DensityMatrix:
    """
    Generate one of the four Bell states.

    Args:
        index: Bell state index (0-3)
            0: |Φ+⟩ = (|00⟩ + |11⟩)/√2
            1: |Φ-⟩ = (|00⟩ - |11⟩)/√2
            2: |Ψ+⟩ = (|01⟩ + |10⟩)/√2
            3: |Ψ-⟩ = (|01⟩ - |10⟩)/√2

    Returns:
        Bell state density matrix
    """
    bell_states = [
        np.array([1, 0, 0, 1]) / np.sqrt(2),   # |Φ+⟩
        np.array([1, 0, 0, -1]) / np.sqrt(2),  # |Φ-⟩
        np.array([0, 1, 1, 0]) / np.sqrt(2),   # |Ψ+⟩
        np.array([0, 1, -1, 0]) / np.sqrt(2),  # |Ψ-⟩
    ]

    psi = bell_states[index % 4]
    rho = np.outer(psi, psi.conj())
    return DensityMatrix(rho)


def generate_werner_state(n_qubits: int, p: float, seed: int = None) -> DensityMatrix:
    """
    Generate a Werner state: ρ = p |Ψ⟩⟨Ψ| + (1-p) I/d

    Args:
        n_qubits: Number of qubits
        p: Mixing parameter (0 to 1)
        seed: Random seed

    Returns:
        Werner state density matrix
    """
    dim = 2 ** n_qubits

    # Use maximally entangled state for n_qubits=2 (Bell state)
    if n_qubits == 2:
        psi_ent = generate_bell_state(0).data
    else:
        # Generalized maximally entangled state
        psi = np.zeros(dim, dtype=complex)
        for i in range(int(np.sqrt(dim))):
            psi[i * int(np.sqrt(dim)) + i] = 1.0
        psi = psi / np.linalg.norm(psi)
        psi_ent = np.outer(psi, psi.conj())

    # Mix with maximally mixed state
    rho = p * psi_ent + (1 - p) * np.eye(dim) / dim
    return DensityMatrix(rho)


def generate_entangled_state(n_qubits: int, entanglement_type: str = 'ghz',
                             noise: float = 0.0, seed: int = None) -> DensityMatrix:
    """
    Generate various types of entangled states.

    Args:
        n_qubits: Number of qubits
        entanglement_type: Type of entanglement ('ghz', 'w', 'random')
        noise: Noise level (0 to 1)
        seed: Random seed

    Returns:
        Entangled density matrix
    """
    if seed is not None:
        np.random.seed(seed)

    dim = 2 ** n_qubits

    if entanglement_type == 'ghz':
        # GHZ state: (|00...0⟩ + |11...1⟩)/√2
        psi = np.zeros(dim, dtype=complex)
        psi[0] = 1 / np.sqrt(2)
        psi[-1] = 1 / np.sqrt(2)

    elif entanglement_type == 'w':
        # W state: superposition of all basis states with one 1
        psi = np.zeros(dim, dtype=complex)
        for i in range(n_qubits):
            psi[2 ** i] = 1 / np.sqrt(n_qubits)

    elif entanglement_type == 'random':
        # Random entangled state via Haar measure
        # Generate via QR decomposition
        A = np.random.randn(dim, 1) + 1j * np.random.randn(dim, 1)
        psi = A[:, 0] / np.linalg.norm(A[:, 0])
    else:
        raise ValueError(f"Unknown entanglement type: {entanglement_type}")

    # Create pure state density matrix
    rho = np.outer(psi, psi.conj())

    # Add noise if specified
    if noise > 0:
        rho = (1 - noise) * rho + noise * np.eye(dim) / dim

    return DensityMatrix(rho)


def partial_transpose(rho: np.ndarray, dims: Tuple[int, int], system: int = 1) -> np.ndarray:
    """
    Compute partial transpose of a density matrix.

    Args:
        rho: Density matrix
        dims: Dimensions of subsystems (d1, d2)
        system: Which system to transpose (0 or 1)

    Returns:
        Partially transposed density matrix
    """
    d1, d2 = dims
    rho_pt = np.zeros_like(rho)

    if system == 1:
        # Transpose second system
        for i1 in range(d1):
            for j1 in range(d1):
                for i2 in range(d2):
                    for j2 in range(d2):
                        rho_pt[i1 * d2 + i2, j1 * d2 + j2] = rho[i1 * d2 + j2, j1 * d2 + i2]
    else:
        # Transpose first system
        for i1 in range(d1):
            for j1 in range(d1):
                for i2 in range(d2):
                    for j2 in range(d2):
                        rho_pt[i1 * d2 + i2, j1 * d2 + j2] = rho[j1 * d2 + i2, i1 * d2 + j2]

    return rho_pt


def check_ppt_criterion(rho: DensityMatrix, dims: Tuple[int, int] = None) -> bool:
    """
    Check the Peres-Horodecki (PPT) criterion for entanglement.

    A state is separable if its partial transpose is positive semidefinite.

    Args:
        rho: Density matrix
        dims: Dimensions of subsystems (default: equal bipartition)

    Returns:
        True if PPT (might be separable), False if NPT (definitely entangled)
    """
    rho_data = rho.data
    dim = rho_data.shape[0]

    if dims is None:
        # Assume equal bipartition
        d = int(np.sqrt(dim))
        dims = (d, d)

    rho_pt = partial_transpose(rho_data, dims, system=1)

    # Check if positive semidefinite
    eigenvalues = np.linalg.eigvalsh(rho_pt)
    return np.all(eigenvalues >= -1e-10)  # Small tolerance for numerical errors


def generate_dataset(n_qubits: int, n_samples: int,
                     entangled_fraction: float = 0.5,
                     noise_range: Tuple[float, float] = (0.0, 0.0),
                     seed: int = None) -> Tuple[List[DensityMatrix], np.ndarray]:
    """
    Generate a dataset of quantum states for witness learning.

    Args:
        n_qubits: Number of qubits
        n_samples: Number of samples
        entangled_fraction: Fraction of entangled states
        noise_range: Range of noise levels (min, max)
        seed: Random seed

    Returns:
        Tuple of (states, labels) where labels are 1 for entangled, 0 for separable
    """
    if seed is not None:
        np.random.seed(seed)

    n_entangled = int(n_samples * entangled_fraction)
    n_separable = n_samples - n_entangled

    states = []
    labels = []

    logger.info(f"Generating {n_entangled} entangled and {n_separable} separable states...")

    # Generate entangled states
    for i in range(n_entangled):
        noise = np.random.uniform(*noise_range)
        ent_type = np.random.choice(['ghz', 'w', 'random'])
        state = generate_entangled_state(n_qubits, ent_type, noise, seed=seed + i if seed else None)
        states.append(state)
        labels.append(1)

    # Generate separable states
    for i in range(n_separable):
        state = generate_separable_state(n_qubits, seed=seed + n_entangled + i if seed else None)
        # Add noise to some separable states
        if noise_range[1] > 0:
            noise = np.random.uniform(*noise_range)
            dim = 2 ** n_qubits
            rho = state.data
            rho = (1 - noise) * rho + noise * np.eye(dim) / dim
            state = DensityMatrix(rho)
        states.append(state)
        labels.append(0)

    # Shuffle
    indices = np.random.permutation(n_samples)
    states = [states[i] for i in indices]
    labels = np.array([labels[i] for i in indices])

    return states, labels
