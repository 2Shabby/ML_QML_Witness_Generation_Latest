"""
Unit tests for the confound-resistant balanced boundary-mixture generator.

Covers basic correctness only (validity, reproducibility, label balance,
oracle consistency, boundary bisection); anti-shortcut statistics are
covered by ``scripts/validate_dataset_confound.py``.
"""

import numpy as np
from qiskit.quantum_info import DensityMatrix

from src.quantum_states.balanced_dataset import (
    generate_balanced_distillability_dataset,
    min_pt_eigenvalue,
    find_boundary_q,
    _npt_generalized_ghz,
    _ppt_classical_diagonal,
)
from src.quantum_states.state_generation import check_npt_any_bipartition


def test_states_are_valid_density_matrices():
    states, labels, metadata = generate_balanced_distillability_dataset(
        n_samples=40, seed=0)
    assert len(states) == len(labels) == len(metadata) == 40
    for rho in states:
        m = np.asarray(rho.data, dtype=np.complex128)
        # Hermitian
        assert np.allclose(m, m.conj().T, atol=1e-10)
        # Trace 1
        assert abs(np.trace(m) - 1.0) < 1e-10
        # Positive semidefinite
        assert np.min(np.linalg.eigvalsh((m + m.conj().T) / 2.0)) > -1e-10
    # Metadata is complete
    for meta in metadata:
        for key in ("family", "q", "q_star", "sample_mode", "purity"):
            assert key in meta
        assert 0.0 < meta["q"] < 1.0
        assert 0.0 < meta["q_star"] < 1.0


def test_reproducibility_same_seed():
    s1, l1, m1 = generate_balanced_distillability_dataset(n_samples=40, seed=123)
    s2, l2, m2 = generate_balanced_distillability_dataset(n_samples=40, seed=123)
    assert np.array_equal(l1, l2)
    assert [x["family"] for x in m1] == [x["family"] for x in m2]
    assert np.allclose([x["q"] for x in m1], [x["q"] for x in m2])
    assert np.allclose([x["q_star"] for x in m1], [x["q_star"] for x in m2])
    assert [x["sample_mode"] for x in m1] == [x["sample_mode"] for x in m2]
    for r1, r2 in zip(s1, s2):
        assert np.allclose(np.asarray(r1.data), np.asarray(r2.data), atol=1e-12)


def test_both_labels_present_in_every_family():
    states, labels, metadata = generate_balanced_distillability_dataset(
        n_samples=400, seed=0)
    families = sorted({m["family"] for m in metadata})
    assert len(families) == 4
    sel_by_fam = {fam: [i for i, m in enumerate(metadata) if m["family"] == fam]
                  for fam in families}
    for fam, idx in sel_by_fam.items():
        yf = labels[np.array(idx)]
        assert 0 < yf.sum() < len(yf), (
            f"family {fam} has only one label "
            f"(n={len(yf)}, n_pos={int(yf.sum())})")


def test_labels_agree_with_npt_oracle():
    states, labels, _ = generate_balanced_distillability_dataset(
        n_samples=60, seed=7)
    for rho, lab in zip(states, labels):
        assert lab == (1 if check_npt_any_bipartition(rho) else 0)


def test_boundary_finder_gives_opposite_sides():
    rng = np.random.default_rng(42)
    rho_n, _ = _npt_generalized_ghz(rng)
    rho_p, _ = _ppt_classical_diagonal(rng)

    # Preconditions: endpoints have opposite labels, PPT anchor is interior.
    assert check_npt_any_bipartition(DensityMatrix(rho_n))
    assert not check_npt_any_bipartition(DensityMatrix(rho_p))
    assert min_pt_eigenvalue(rho_p) > 0.0

    q_star = find_boundary_q(rho_n, rho_p, tol=1e-5)
    assert 0.0 < q_star < 1.0

    # Convexity of the all-PPT set: below q* is PPT, above q* is NPT.
    for dq in (0.05, 0.10, 0.20):
        q_lo = max(0.005, q_star - dq)
        q_hi = min(0.995, q_star + dq)
        rho_lo = q_lo * rho_n + (1.0 - q_lo) * rho_p
        rho_hi = q_hi * rho_n + (1.0 - q_hi) * rho_p
        assert not check_npt_any_bipartition(DensityMatrix(rho_lo)), (
            f"q={q_lo} (q*-dq) should be PPT, q*={q_star}")
        assert check_npt_any_bipartition(DensityMatrix(rho_hi)), (
            f"q={q_hi} (q*+dq) should be NPT, q*={q_star}")
