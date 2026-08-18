"""
Unit tests for the partial-transpose / NPT label machinery.
"""

import numpy as np
from qiskit.quantum_info import DensityMatrix

from src.quantum_states.state_generation import (
    check_npt_any_bipartition,
    partial_transpose,
    _permute_qubits,
)


def _pure_state(psi: np.ndarray) -> DensityMatrix:
    return DensityMatrix(np.outer(psi, psi.conj()))


def _bell(bell_type: int = 0) -> DensityMatrix:
    # 0: |Phi+>  1: |Phi->  2: |Psi+>  3: |Psi->
    (i, s_i), (j, s_j) = [((0, +1), (3, +1)), ((0, +1), (3, -1)),
                          ((1, +1), (2, +1)), ((1, +1), (2, -1))][bell_type]
    psi = np.zeros(4, dtype=complex)
    psi[i] = s_i / np.sqrt(2)
    psi[j] = s_j / np.sqrt(2)
    return _pure_state(psi)


def _ghz(noise: float = 0.0) -> DensityMatrix:
    psi = np.zeros(8, dtype=complex)
    psi[0] = psi[-1] = 1 / np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    if noise > 0:
        rho = (1 - noise) * rho + noise * np.eye(8) / 8
    return DensityMatrix(rho)


def _w_state(noise: float = 0.0) -> DensityMatrix:
    psi = np.zeros(8, dtype=complex)
    for i in range(3):
        psi[2**i] = 1 / np.sqrt(3)
    rho = np.outer(psi, psi.conj())
    if noise > 0:
        rho = (1 - noise) * rho + noise * np.eye(8) / 8
    return DensityMatrix(rho)


def _cluster() -> DensityMatrix:
    psi = np.ones(8, dtype=complex) / np.sqrt(8)
    for idx in range(8):
        b0, b1, b2 = (idx >> 2) & 1, (idx >> 1) & 1, idx & 1
        if b0 & b1:
            psi[idx] *= -1
        if b1 & b2:
            psi[idx] *= -1  # |111> gets two flips -> +1
    return _pure_state(psi)


def _product_state(seed: int = 0) -> DensityMatrix:
    rng = np.random.default_rng(seed)
    rho = None
    for _ in range(3):
        theta = np.arccos(2 * rng.random() - 1)
        phi = 2 * np.pi * rng.random()
        psi_i = np.array([np.cos(theta / 2),
                          np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)
        rho_i = np.outer(psi_i, psi_i.conj())
        rho = rho_i if rho is None else np.kron(rho, rho_i)
    return DensityMatrix(rho)


class TestPartialTranspose:
    """Test partial transpose computation and qubit permutation."""

    def test_bell_pt_has_negative_eigenvalue(self):
        rho_pt = partial_transpose(np.asarray(_bell(0).data), dims=[2, 2], axis=1)
        assert np.any(np.linalg.eigvalsh(rho_pt) < -1e-10)

    def test_product_pt_stays_positive(self):
        rho = np.asarray(_product_state(42).data)
        for dims, axis in ([(2, 4), 0], [(2, 4), 1], [(4, 2), 0], [(4, 2), 1]):
            eigvals = np.linalg.eigvalsh(partial_transpose(rho, dims=dims, axis=axis))
            assert np.all(eigvals >= -1e-10)

    def test_permute_qubits_identity(self):
        rho = np.asarray(_ghz().data)
        assert np.allclose(rho, _permute_qubits(rho, [0, 1, 2]))

    def test_permute_qubits_swap_preserves_properties(self):
        rho = np.asarray(_ghz(0.1).data)
        rho_perm = _permute_qubits(rho, [1, 0, 2])
        assert np.isclose(np.trace(rho_perm), 1.0)
        assert np.allclose(rho_perm, rho_perm.conj().T)
        assert np.all(np.linalg.eigvalsh(rho_perm) >= -1e-10)


class TestNPTAnyBipartition:
    """Test the distillability label (NPT across any 1|23 bipartition)."""

    def test_pure_ghz_is_npt(self):
        assert check_npt_any_bipartition(_ghz())

    def test_pure_w_is_npt(self):
        assert check_npt_any_bipartition(_w_state())

    def test_pure_cluster_is_npt(self):
        assert check_npt_any_bipartition(_cluster())

    def test_product_states_are_ppt(self):
        for seed in (42, 123, 456):
            assert not check_npt_any_bipartition(_product_state(seed))

    def test_noisy_ghz_crosses_boundary(self):
        # Depolarized GHZ is NPT up to noise ~0.8 under this convention
        low = [check_npt_any_bipartition(_ghz(n)) for n in (0.0, 0.1, 0.2, 0.3, 0.75)]
        high = [check_npt_any_bipartition(_ghz(n)) for n in (0.8, 0.85, 0.9)]
        assert all(low)
        assert not any(high)
