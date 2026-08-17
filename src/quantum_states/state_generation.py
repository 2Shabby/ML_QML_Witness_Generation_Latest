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
    random_density_matrix
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


def generate_noisy_cluster_state(
    n_qubits: int = 3,
    noise_level: float = 0.0,
    seed: Optional[int] = None
) -> DensityMatrix:
    """
    Generate a linear cluster state with depolarizing noise.

    The 3-qubit linear cluster state is:
        |cluster⟩ = CZ_{01} CZ_{12} |+⟩^⊗3

    where CZ is the controlled-Z gate. This state is a resource for
    measurement-based quantum computing.

    Args:
        n_qubits: Number of qubits (default 3)
        noise_level: Depolarizing noise level (0.0 = pure, 1.0 = maximally mixed)
        seed: Random seed (not used for pure state, but included for API consistency)

    Returns:
        Cluster state density matrix (possibly with depolarizing noise)
    """
    if seed is not None:
        np.random.seed(seed)

    dim = 2 ** n_qubits

    # Start with |+⟩^⊗n
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    psi = plus.copy()
    for _ in range(n_qubits - 1):
        psi = np.kron(psi, plus)

    # Apply CZ gates between consecutive qubits
    # CZ_{i,i+1} applies phase -1 to |11⟩ on qubits i and i+1
    for i in range(n_qubits - 1):
        # Build the CZ gate for qubits i and i+1
        # CZ|ab⟩ = (-1)^{a*b} |ab⟩
        for idx in range(dim):
            # Extract bits i and i+1
            bit_i = (idx >> (n_qubits - 1 - i)) & 1
            bit_i1 = (idx >> (n_qubits - 1 - (i + 1))) & 1
            if bit_i == 1 and bit_i1 == 1:
                psi[idx] *= -1

    # Create density matrix
    rho = np.outer(psi, psi.conj())

    # Add depolarizing noise if requested
    if noise_level > 0:
        identity = np.eye(dim) / dim
        rho = (1 - noise_level) * rho + noise_level * identity

    return DensityMatrix(rho)


def generate_3qubit_product_state(seed: Optional[int] = None) -> DensityMatrix:
    """
    Generate a random 3-qubit product (fully separable) state.

    The state is ρ = ρ_A ⊗ ρ_B ⊗ ρ_C where each single-qubit state
    is a random pure state on the Bloch sphere.

    Product states are guaranteed to be:
    - Separable (no entanglement)
    - PPT across all bipartitions
    - NOT distillable

    Args:
        seed: Random seed for reproducibility

    Returns:
        3-qubit product state density matrix
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate 3 random single-qubit pure states
    rho = None
    for _ in range(3):
        # Random point on Bloch sphere using uniform distribution
        theta = np.arccos(2 * np.random.rand() - 1)  # Uniform on sphere
        phi = 2 * np.pi * np.random.rand()

        # Pure state |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ}sin(θ/2)|1⟩
        psi_i = np.array([
            np.cos(theta / 2),
            np.exp(1j * phi) * np.sin(theta / 2)
        ], dtype=complex)
        rho_i = np.outer(psi_i, psi_i.conj())

        if rho is None:
            rho = rho_i
        else:
            rho = np.kron(rho, rho_i)

    return DensityMatrix(rho)


def generate_distillability_dataset(
    n_samples: int = 5000,
    noise_range: Tuple[float, float] = (0.0, 0.5),
    seed: Optional[int] = None
) -> Tuple[List[DensityMatrix], np.ndarray]:
    """
    Generate labeled dataset for 3-qubit distillability learning.

    Creates a diverse set of 3-qubit quantum states with labels indicating
    whether they are distillable (NPT across at least one bipartition).

    Labels:
        1 = Distillable (NPT across at least one bipartition)
        0 = Non-distillable (PPT across all bipartitions)

    State families included:
        - Noisy GHZ states (varying noise levels)
        - Noisy W states (varying noise levels)
        - Noisy cluster states (varying noise levels)
        - Random mixed states
        - Product states (guaranteed non-distillable)

    Args:
        n_samples: Total number of states to generate
        noise_range: Range of noise levels (min, max) for noisy states
        seed: Random seed for reproducibility

    Returns:
        Tuple of (states, labels) where:
            states: List of DensityMatrix objects (3-qubit)
            labels: numpy array of 0/1 labels (0=non-distillable, 1=distillable)
    """
    if seed is not None:
        np.random.seed(seed)

    states = []
    labels = []

    # Distribute samples across state families
    n_per_family = n_samples // 5
    remainder = n_samples % 5

    # Track seed offsets for reproducibility
    seed_offset = 0

    # 1. Noisy GHZ states
    for i in range(n_per_family):
        noise = np.random.uniform(*noise_range)
        state = generate_entangled_state(
            n_qubits=3,
            entanglement_type='ghz',
            noise_level=noise,
            seed=(seed + seed_offset + i) if seed else None
        )
        is_distillable = check_npt_any_bipartition(state)
        states.append(state)
        labels.append(1 if is_distillable else 0)
    seed_offset += n_per_family

    # 2. Noisy W states
    for i in range(n_per_family):
        noise = np.random.uniform(*noise_range)
        state = generate_entangled_state(
            n_qubits=3,
            entanglement_type='w',
            noise_level=noise,
            seed=(seed + seed_offset + i) if seed else None
        )
        is_distillable = check_npt_any_bipartition(state)
        states.append(state)
        labels.append(1 if is_distillable else 0)
    seed_offset += n_per_family

    # 3. Noisy cluster states
    for i in range(n_per_family):
        noise = np.random.uniform(*noise_range)
        state = generate_noisy_cluster_state(
            n_qubits=3,
            noise_level=noise,
            seed=(seed + seed_offset + i) if seed else None
        )
        is_distillable = check_npt_any_bipartition(state)
        states.append(state)
        labels.append(1 if is_distillable else 0)
    seed_offset += n_per_family

    # 4. Random mixed states
    for i in range(n_per_family):
        state = generate_random_density_matrix(
            n_qubits=3,
            seed=(seed + seed_offset + i) if seed else None
        )
        is_distillable = check_npt_any_bipartition(state)
        states.append(state)
        labels.append(1 if is_distillable else 0)
    seed_offset += n_per_family

    # 5. Product states (guaranteed non-distillable)
    for i in range(n_per_family + remainder):
        state = generate_3qubit_product_state(
            seed=(seed + seed_offset + i) if seed else None
        )
        # Product states are always PPT, hence non-distillable
        states.append(state)
        labels.append(0)

    # Shuffle the dataset
    indices = np.random.permutation(len(states))
    states = [states[i] for i in indices]
    labels = np.array([labels[i] for i in indices])

    logger.info(f"Generated distillability dataset: {len(states)} states")
    logger.info(f"  Distillable: {np.sum(labels)} ({100*np.mean(labels):.1f}%)")
    logger.info(f"  Non-distillable: {len(labels) - np.sum(labels)} ({100*(1-np.mean(labels)):.1f}%)")

    return states, labels


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
