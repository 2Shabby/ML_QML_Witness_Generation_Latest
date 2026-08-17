"""
Confound-resistant dataset generator for 3-qubit distillability learning.

Motivation
----------
The previous ``generate_distillability_dataset`` was badly confounded:
every negative sample was a product state, and the L2 norm of the 36D
Pauli feature vector alone achieved 100% classification.  All families
were single-parameter depolarizing paths, so the label was a threshold
on a monotone scalar (noise -> feature norm).

This module replaces that design with *boundary-crossing mixtures*:

    rho(q) = q * rho_NPT + (1 - q) * rho_PPT

where

* ``rho_NPT`` is an entangled anchor guaranteed NPT (generalized GHZ,
  generalized W, cluster, or Haar-random pure states, with varied
  amplitudes, phases, and local rotations);
* ``rho_PPT`` is a fully separable but *non-product* (classically
  correlated) anchor, verified to be interior to the all-PPT set with
  a strict positive partial-transpose eigenvalue margin;
* ``q`` is sampled around the actual PPT/NPT boundary ``q*`` of the
  pair, found by bisection on the NPT oracle.  Because the all-PPT set
  is convex, the set of ``q`` for which ``rho(q)`` is all-PPT is the
  interval ``[0, q*]`` and ``q*`` is well defined.

Note on the feature norm: the 36D feature vector is *linear* in ``q``,
but its L2 norm is NOT a weighted sum of the endpoint norms.  Whether
norm / purity / ``q`` still predict the label must therefore be
established empirically by the audit script
(``scripts/validate_dataset_confound.py``), not assumed.

Labels are always computed from the NPT oracle on the final state
(``check_npt_any_bipartition``); bisection is used only to *place*
samples near the boundary, never to assign labels.
"""

import numpy as np
from qiskit.quantum_info import DensityMatrix
from typing import List, Optional, Tuple

from .state_generation import (
    check_npt_any_bipartition,
    partial_transpose,
    _permute_qubits,
)

# Qubit order is (0, 1, 2); bipartitions A|BC, B|AC, C|AB.
_DIMS_3Q = 8


def min_pt_eigenvalue(rho: np.ndarray) -> float:
    """
    Minimum partial-transpose eigenvalue over all three 1|23 bipartitions.

    Returns
    -------
    float
        min eigenvalue of rho with subsystem A, B, and C transposed
        (the smallest over the three bipartitions).  Positive values mean
        the state is interior to the all-PPT set (PPT with a margin).
    """
    rho = np.asarray(rho)

    pt_a = partial_transpose(rho, dims=[2, 4], axis=0)
    pt_b = partial_transpose(_permute_qubits(rho, [1, 0, 2]), dims=[2, 4], axis=0)
    pt_c = partial_transpose(rho, dims=[4, 2], axis=1)

    return float(min(
        np.min(np.linalg.eigvalsh(pt_a)),
        np.min(np.linalg.eigvalsh(pt_b)),
        np.min(np.linalg.eigvalsh(pt_c)),
    ))


def _random_local_unitary(rng: np.random.Generator, dim: int = 2) -> np.ndarray:
    """Random SU(2) single-qubit unitary (Haar distributed)."""
    # Haar via two independent uniform angles (quaternion construction).
    u1 = rng.random()
    u2 = rng.random(2)
    theta = 2.0 * np.arcsin(np.sqrt(u1))
    phi = 2.0 * np.pi * u2[0]
    # n unit vector from u2 (simplified: fix theta_n uniform, phi_n uniform)
    cos_n = 2.0 * rng.random() - 1.0
    sin_n = np.sqrt(1.0 - cos_n ** 2)
    phi_n = 2.0 * np.pi * rng.random()
    n = np.array([sin_n * np.cos(phi_n), sin_n * np.sin(phi_n), cos_n])
    return np.cos(theta / 2) * np.eye(2) - 1j * np.sin(theta / 2) * (
        n[0] * np.array([[0, 1], [1, 0]])
        + n[1] * np.array([[0, -1j], [1j, 0]])
        + n[2] * np.array([[1, 0], [0, -1]])
    )


def _apply_local_unitaries(rho: np.ndarray, us: List[np.ndarray]) -> np.ndarray:
    """Apply local unitaries us[0..2] on qubits 0..2 (qubit 0 most significant)."""
    u_total = us[0]
    for ui in us[1:]:
        u_total = np.kron(u_total, ui)
    return u_total @ rho @ u_total.conj().T


# ---------------------------------------------------------------------------
# NPT anchors (guaranteed NPT across at least one bipartition)
# ---------------------------------------------------------------------------

def _npt_generalized_ghz(rng: np.random.Generator, apply_local_rotation: bool = True) -> Tuple[np.ndarray, dict]:
    """|psi> = cos(a)|000> + exp(i phi) sin(a)|111>, a, phi varied."""
    a = rng.uniform(0.15, 0.85) * (np.pi / 2)
    phi = rng.uniform(0.0, 2.0 * np.pi)
    psi = np.zeros(_DIMS_3Q, dtype=complex)
    psi[0] = np.cos(a)
    psi[-1] = np.exp(1j * phi) * np.sin(a)
    rho = np.outer(psi, psi.conj())
    params = {"alpha": float(a), "phase": float(phi), "local_rot": False}
    if apply_local_rotation and rng.random() < 0.5:
        us = [_random_local_unitary(rng) for _ in range(3)]
        rho = _apply_local_unitaries(rho, us)
        params["local_rot"] = True
    return rho, params


def _npt_generalized_w(rng: np.random.Generator, apply_local_rotation: bool = True) -> Tuple[np.ndarray, dict]:
    """W-like state with random amplitudes and relative phases on |100>,|010>,|001>."""
    # Sample positive amplitudes with no near-product corner.
    for _ in range(50):
        amp = np.abs(rng.normal(0.0, 1.0, 3))
        amp /= np.linalg.norm(amp)
        if np.all(amp >= 0.2):
            break
    else:
        amp = np.ones(3) / np.sqrt(3.0)
    p12 = rng.uniform(0.0, 2.0 * np.pi)
    p13 = rng.uniform(0.0, 2.0 * np.pi)
    psi = np.zeros(_DIMS_3Q, dtype=complex)
    psi[0b100] = amp[0]
    psi[0b010] = amp[1] * np.exp(1j * p12)
    psi[0b001] = amp[2] * np.exp(1j * p13)
    rho = np.outer(psi, psi.conj())
    params = {"amplitudes": [float(v) for v in amp],
              "phase_12": float(p12), "phase_13": float(p13),
              "local_rot": False}
    if apply_local_rotation and rng.random() < 0.5:
        us = [_random_local_unitary(rng) for _ in range(3)]
        rho = _apply_local_unitaries(rho, us)
        params["local_rot"] = True
    return rho, params


def _npt_cluster(rng: np.random.Generator, apply_local_rotation: bool = True) -> Tuple[np.ndarray, dict]:
    """3-qubit linear cluster state, optionally with random local rotations."""
    from .state_generation import generate_noisy_cluster_state
    rho = generate_noisy_cluster_state(n_qubits=3, noise_level=0.0).data
    params = {"local_rot": False}
    if apply_local_rotation and rng.random() < 0.5:
        us = [_random_local_unitary(rng) for _ in range(3)]
        rho = _apply_local_unitaries(rho, us)
        params["local_rot"] = True
    return rho, params


def _npt_random_pure(rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
    """Haar-random pure state; accepted only if NPT (rejected otherwise)."""
    for _ in range(50):
        psi = (rng.normal(size=_DIMS_3Q) + 1j * rng.normal(size=_DIMS_3Q))
        psi /= np.linalg.norm(psi)
        rho = np.outer(psi, psi.conj())
        if check_npt_any_bipartition(DensityMatrix(rho)):
            return rho, {"local_rot": False}
    raise RuntimeError("Could not draw an NPT random pure state")


# ---------------------------------------------------------------------------
# PPT anchors (fully separable, non-product, interior to the all-PPT set)
# ---------------------------------------------------------------------------

def _ppt_classical_diagonal(rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
    """
    Separable, classically correlated diagonal mixture
    p|000><000| + (1-p)|111><111|, mixed with I/8 at a small random eps.
    """
    p = rng.uniform(0.1, 0.9)
    rho = np.zeros((_DIMS_3Q, _DIMS_3Q), dtype=complex)
    rho[0, 0] = p
    rho[-1, -1] = 1.0 - p
    # Broad I/8 mixing range: keeps the anchor interior-PPT while making its
    # purity/norm overlap the NPT-anchor range (avoids a residual
    # "negatives = cleaner state" scalar confound).
    eps = rng.uniform(0.05, 0.5)
    rho = (1.0 - eps) * rho + eps * np.eye(_DIMS_3Q) / _DIMS_3Q
    return rho, {"p": float(p), "eps": float(eps)}


def _ppt_separable_mixture(rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
    """
    Separable mixture of 2-5 random product pure states, mixed with I/8 at a
    small random eps so the anchor is interior (not rank-deficient).
    Convex mixtures of product states are separable, hence PPT across every
    bipartition, but the state is generally non-product.
    """
    n_comp = int(rng.integers(2, 6))
    weights = rng.dirichlet(np.ones(n_comp))
    rho = np.zeros((_DIMS_3Q, _DIMS_3Q), dtype=complex)
    for w in weights:
        fac = np.array([[1.0]], dtype=complex)
        for _ in range(3):
            th = np.arccos(2.0 * rng.random() - 1.0)
            ph = 2.0 * np.pi * rng.random()
            psi_i = np.array([np.cos(th / 2), np.exp(1j * ph) * np.sin(th / 2)], dtype=complex)
            fac = np.kron(fac, np.outer(psi_i, psi_i.conj()))
        rho += w * fac
    eps = rng.uniform(0.1, 0.55)
    rho = (1.0 - eps) * rho + eps * np.eye(_DIMS_3Q) / _DIMS_3Q
    return rho, {"n_components": n_comp, "eps": float(eps)}


_PPT_BUILDERS = {
    "classical_diagonal": _ppt_classical_diagonal,
    "separable_mixture": _ppt_separable_mixture,
}


def _make_ppt_anchor(rng: np.random.Generator, kind: str, margin: float, max_tries: int = 30) -> Tuple[np.ndarray, dict]:
    """Sample a separable anchor until it has all-PT eigenvalue margin > `margin`."""
    for _ in range(max_tries):
        rho, params = _PPT_BUILDERS[kind](rng)
        if min_pt_eigenvalue(rho) > margin:
            params["type"] = kind
            return rho, params
    raise RuntimeError(f"PPT anchor of type {kind} never reached margin {margin} in {max_tries} tries")


# ---------------------------------------------------------------------------
# Boundary search and sampling
# ---------------------------------------------------------------------------

def find_boundary_q(rho_npt: np.ndarray, rho_ppt: np.ndarray, tol: float = 1e-4) -> float:
    """
    Bisection for q* such that rho(q) = q*rho_npt + (1-q)*rho_ppt is all-PPT
    for q < q* and NPT for q > q*.

    Valid because the all-PPT set is convex: the PPT segment of the line is
    the interval [0, q*].  Requires endpoints with opposite labels.
    """
    if check_npt_any_bipartition(DensityMatrix(rho_ppt)):
        raise ValueError("PPT anchor is NPT; bisection precondition violated")
    if not check_npt_any_bipartition(DensityMatrix(rho_npt)):
        raise ValueError("NPT anchor is PPT; bisection precondition violated")

    lo, hi = 0.0, 1.0  # lo: PPT, hi: NPT
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        rho = mid * rho_npt + (1.0 - mid) * rho_ppt
        if check_npt_any_bipartition(DensityMatrix(rho)):
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return float(0.5 * (lo + hi))


def _sample_q(rng: np.random.Generator, q_star: float, near_boundary_fraction: float) -> Tuple[float, str]:
    """Sample the mixing weight: near the boundary or uniform far away."""
    if rng.random() < near_boundary_fraction:
        side = 1.0 if rng.random() < 0.5 else -1.0
        d = rng.uniform(0.02, 0.15)
        q = q_star + side * d
        mode = "near"
    else:
        # Symmetric window around q*: keeps far samples roughly 50/50 while
        # reaching further from the boundary than the near band.
        side = 1.0 if rng.random() < 0.5 else -1.0
        d = rng.uniform(0.15, 0.45)
        q = q_star + side * d
        mode = "far"
    return float(np.clip(q, 0.005, 0.995)), mode


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------

# Each family pairs an NPT anchor class with one or more PPT anchor classes.
_FAMILIES = {
    "ghz_x_diag":   {"npt": _npt_generalized_ghz, "ppt": ["classical_diagonal"]},
    "w_x_sep":      {"npt": _npt_generalized_w, "ppt": ["separable_mixture"]},
    "cluster_x_sep": {"npt": _npt_cluster, "ppt": ["separable_mixture"]},
    "random_x_mix": {"npt": _npt_random_pure, "ppt": ["classical_diagonal", "separable_mixture"]},
}


def generate_balanced_distillability_dataset(
    n_samples: int = 4000,
    seed: Optional[int] = None,
    near_boundary_fraction: float = 0.6,
    ppt_margin: float = 1e-3,
    q_star_range: Tuple[float, float] = (0.05, 0.95),
    family_weights: Optional[List[float]] = None,
) -> Tuple[List[DensityMatrix], np.ndarray, List[dict]]:
    """
    Generate a confound-resistant labeled dataset of 3-qubit states.

    Each state is rho(q) = q * rho_NPT + (1 - q) * rho_PPT with q sampled
    around the pair's PPT/NPT boundary q* (found by bisection).  Labels are
    always recomputed from the NPT oracle on the final state.

    Parameters
    ----------
    n_samples : total number of states
    seed : RNG seed
    near_boundary_fraction : fraction of samples with q within [-0.15,+0.15]
        of q* (50/50 sides by construction); the rest use a symmetric
        window [-0.45,+0.45] around q*, also 50/50 sides
    ppt_margin : required minimum all-PT eigenvalue of PPT anchors
    q_star_range : pairs whose boundary lies outside this range are rejected
        (keeps the near-boundary band meaningful on both sides)

    Returns
    -------
    (states, labels, metadata) : DensityMatrix list, 0/1 label array, and
    per-state metadata (family, q, q_star, sample mode, anchor params).
    """
    rng = np.random.default_rng(seed)
    family_names = list(_FAMILIES.keys())
    if family_weights is None:
        family_weights = [1.0] * len(family_names)
    family_probs = np.asarray(family_weights) / sum(family_weights)

    states = []
    labels = []
    metadata = []

    max_pair_tries = 20
    for i in range(n_samples):
        family = family_names[int(rng.choice(len(family_names), p=family_probs))]
        spec = _FAMILIES[family]

        q_star = None
        for _ in range(max_pair_tries):
            rho_n, n_params = spec["npt"](rng)
            if not check_npt_any_bipartition(DensityMatrix(rho_n)):
                continue
            ppt_kind = str(rng.choice(spec["ppt"]))
            rho_p, p_params = _make_ppt_anchor(rng, ppt_kind, ppt_margin)
            try:
                qs = find_boundary_q(rho_n, rho_p)
            except ValueError:
                continue
            if not (q_star_range[0] <= qs <= q_star_range[1]):
                continue
            q_star = qs
            break
        if q_star is None:
            raise RuntimeError(
                f"Failed to find a valid anchor pair for family {family} "
                f"after {max_pair_tries} tries (sample {i})"
            )

        q, mode = _sample_q(rng, q_star, near_boundary_fraction)
        rho = q * rho_n + (1.0 - q) * rho_p
        rho = (rho + rho.conj().T) / 2.0  # kill numerical asymmetry

        label = 1 if check_npt_any_bipartition(DensityMatrix(rho)) else 0
        purity = float(np.real(np.trace(rho @ rho)))

        states.append(DensityMatrix(rho))
        labels.append(label)
        metadata.append({
            "family": family,
            "q": q,
            "q_star": q_star,
            "sample_mode": mode,
            "purity": purity,
            "npt_anchor": n_params,
            "ppt_anchor": p_params,
        })

    # Shuffle
    idx = rng.permutation(n_samples)
    states = [states[i] for i in idx]
    labels = np.asarray([labels[i] for i in idx], dtype=int)
    metadata = [metadata[i] for i in idx]

    return states, labels, metadata
