"""
Quantum State Generation Module

Implements state generation methods for creating training datasets
following Section 10.6 of the framework document.
"""

import numpy as np
from qiskit.quantum_info import (
    DensityMatrix,
    Statevector,
    random_statevector,
    random_density_matrix,
    partial_trace
)
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def generate_random_density_matrix(
    n_qubits: int,
    rank: Optional[int] = None,
    seed: Optional[int] = None
) -> DensityMatrix:
    """
    Generate a random density matrix for n qubits.

    Args:
        n_qubits: Number of qubits
        rank: Rank of the density matrix (default: full rank)
        seed: Random seed for reproducibility

    Returns:
        Random density matrix
    """
    dim = 2 ** n_qubits
    if seed is not None:
        np.random.seed(seed)

    return random_density_matrix(dim, rank=rank, seed=seed)


def generate_separable_state(
    n_qubits: int,
    n_components: int = 5,
    seed: Optional[int] = None
) -> DensityMatrix:
    """
    Generate a separable (non-entangled) state as a convex combination
    of product states: ρ_sep = Σᵢ pᵢ |ψ_A,i⟩⟨ψ_A,i| ⊗ |ψ_B,i⟩⟨ψ_B,i|

    Following Section 6.2 dataset generation approach.

    Args:
        n_qubits: Total number of qubits (must be even for bipartite)
        n_components: Number of product states in convex combination
        seed: Random seed for reproducibility

    Returns:
        Separable density matrix
    """
    if n_qubits % 2 != 0:
        raise ValueError("n_qubits must be even for bipartite separable states")

    if seed is not None:
        np.random.seed(seed)

    n_qubits_A = n_qubits // 2
    n_qubits_B = n_qubits // 2
    dim_A = 2 ** n_qubits_A
    dim_B = 2 ** n_qubits_B
    dim_total = dim_A * dim_B

    # Generate random probability distribution
    probs = np.random.dirichlet(np.ones(n_components))

    # Initialize separable state
    rho_sep = np.zeros((dim_total, dim_total), dtype=complex)

    for i in range(n_components):
        # Generate random product state
        psi_A = random_statevector(dim_A, seed=seed+i if seed else None)
        psi_B = random_statevector(dim_B, seed=seed+i+1000 if seed else None)

        # Create product state density matrix
        rho_A = DensityMatrix(psi_A)
        rho_B = DensityMatrix(psi_B)

        # Tensor product
        rho_product = np.kron(rho_A.data, rho_B.data)

        # Add to convex combination
        rho_sep += probs[i] * rho_product

    return DensityMatrix(rho_sep)


def generate_entangled_state(
    n_qubits: int,
    entanglement_type: str = 'random',
    noise_level: float = 0.0,
    seed: Optional[int] = None
) -> DensityMatrix:
    """
    Generate an entangled state.

    Args:
        n_qubits: Number of qubits
        entanglement_type: Type of entangled state ('random', 'bell', 'ghz', 'w')
        noise_level: Depolarizing noise level (0.0 = no noise, 1.0 = maximally mixed)
        seed: Random seed for reproducibility

    Returns:
        Entangled density matrix
    """
    if seed is not None:
        np.random.seed(seed)

    dim = 2 ** n_qubits

    if entanglement_type == 'random':
        # Generate random pure state (likely entangled for n>1)
        psi = random_statevector(dim, seed=seed)
        rho = DensityMatrix(psi)
    elif entanglement_type == 'bell' and n_qubits == 2:
        rho = generate_bell_state(bell_type=0)
    elif entanglement_type == 'ghz':
        # GHZ state: (|0...0⟩ + |1...1⟩)/√2
        psi = np.zeros(dim, dtype=complex)
        psi[0] = 1/np.sqrt(2)  # |0...0⟩
        psi[-1] = 1/np.sqrt(2)  # |1...1⟩
        rho = DensityMatrix(Statevector(psi))
    elif entanglement_type == 'w':
        # W state: (|10...0⟩ + |01...0⟩ + ... + |0...01⟩)/√n
        psi = np.zeros(dim, dtype=complex)
        for i in range(n_qubits):
            psi[2**i] = 1/np.sqrt(n_qubits)
        rho = DensityMatrix(Statevector(psi))
    else:
        raise ValueError(f"Unknown entanglement_type: {entanglement_type}")

    # Add depolarizing noise if requested
    if noise_level > 0:
        identity = np.eye(dim) / dim
        rho = DensityMatrix((1 - noise_level) * rho.data + noise_level * identity)

    return rho


def generate_bell_state(bell_type: int = 0) -> DensityMatrix:
    """
    Generate one of the four Bell states.

    Args:
        bell_type: Which Bell state (0-3)
            0: |Φ+⟩ = (|00⟩ + |11⟩)/√2
            1: |Φ-⟩ = (|00⟩ - |11⟩)/√2
            2: |Ψ+⟩ = (|01⟩ + |10⟩)/√2
            3: |Ψ-⟩ = (|01⟩ - |10⟩)/√2

    Returns:
        Bell state density matrix
    """
    psi = np.zeros(4, dtype=complex)

    if bell_type == 0:  # |Φ+⟩
        psi[0] = 1/np.sqrt(2)  # |00⟩
        psi[3] = 1/np.sqrt(2)  # |11⟩
    elif bell_type == 1:  # |Φ-⟩
        psi[0] = 1/np.sqrt(2)
        psi[3] = -1/np.sqrt(2)
    elif bell_type == 2:  # |Ψ+⟩
        psi[1] = 1/np.sqrt(2)  # |01⟩
        psi[2] = 1/np.sqrt(2)  # |10⟩
    elif bell_type == 3:  # |Ψ-⟩
        psi[1] = 1/np.sqrt(2)
        psi[2] = -1/np.sqrt(2)
    else:
        raise ValueError(f"bell_type must be 0-3, got {bell_type}")

    return DensityMatrix(Statevector(psi))


def generate_werner_state(n_qubits: int, p: float) -> DensityMatrix:
    """
    Generate a Werner state: ρ(p) = p|Ψ-⟩⟨Ψ-| + (1-p)I/d

    Args:
        n_qubits: Number of qubits (typically 2)
        p: Mixing parameter (0 ≤ p ≤ 1)

    Returns:
        Werner state density matrix
    """
    if not 0 <= p <= 1:
        raise ValueError(f"p must be in [0, 1], got {p}")

    dim = 2 ** n_qubits

    # Start with singlet state |Ψ-⟩
    psi_minus = generate_bell_state(bell_type=3)

    # Mix with maximally mixed state
    identity = np.eye(dim) / dim
    rho = p * psi_minus.data + (1 - p) * identity

    return DensityMatrix(rho)


def generate_dataset(
    n_qubits: int,
    n_samples: int,
    entangled_fraction: float = 0.5,
    noise_range: Tuple[float, float] = (0.0, 0.3),
    seed: Optional[int] = None
) -> Tuple[List[DensityMatrix], np.ndarray]:
    """
    Generate a labeled dataset of quantum states for training.

    Following Section 10.6 data pipeline approach.

    Args:
        n_qubits: Number of qubits per state
        n_samples: Total number of states to generate
        entangled_fraction: Fraction of entangled states (label=1)
        noise_range: Range of noise levels to apply
        seed: Random seed for reproducibility

    Returns:
        Tuple of (states, labels) where:
            states: List of DensityMatrix objects
            labels: Array of labels (0=separable, 1=entangled)
    """
    if seed is not None:
        np.random.seed(seed)

    n_entangled = int(n_samples * entangled_fraction)
    n_separable = n_samples - n_entangled

    states = []
    labels = []

    logger.info(f"Generating {n_separable} separable states...")
    # Generate separable states (label = 0)
    for i in range(n_separable):
        noise = np.random.uniform(*noise_range)
        state = generate_separable_state(
            n_qubits,
            n_components=5,
            seed=seed+i if seed else None
        )

        # Add noise
        if noise > 0:
            dim = 2 ** n_qubits
            identity = np.eye(dim) / dim
            state = DensityMatrix((1 - noise) * state.data + noise * identity)

        states.append(state)
        labels.append(0)

    logger.info(f"Generating {n_entangled} entangled states...")
    # Generate entangled states (label = 1)
    for i in range(n_entangled):
        noise = np.random.uniform(*noise_range)
        state = generate_entangled_state(
            n_qubits,
            entanglement_type='random',
            noise_level=noise,
            seed=seed+n_separable+i if seed else None
        )
        states.append(state)
        labels.append(1)

    # Shuffle the dataset
    indices = np.random.permutation(n_samples)
    states = [states[i] for i in indices]
    labels = np.array([labels[i] for i in indices])

    logger.info(f"Generated {n_samples} states ({n_separable} separable, {n_entangled} entangled)")

    return states, labels


def check_ppt_criterion(rho: DensityMatrix, dims: List[int]) -> bool:
    """
    Check if a state satisfies the PPT (Positive Partial Transpose) criterion.

    A state is PPT if its partial transpose has no negative eigenvalues.
    For 2x2 and 2x3 systems, PPT is necessary and sufficient for separability.

    Args:
        rho: Density matrix to check
        dims: Dimensions of subsystems [dim_A, dim_B]

    Returns:
        True if state is PPT (positive partial transpose)
    """
    # Compute partial transpose on second subsystem
    rho_pt = partial_transpose(rho.data, dims)

    # Check if all eigenvalues are non-negative
    eigenvalues = np.linalg.eigvalsh(rho_pt)

    # Use small tolerance for numerical errors
    return np.all(eigenvalues >= -1e-10)


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
