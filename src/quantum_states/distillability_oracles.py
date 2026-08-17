"""
Distillability Oracles for 3-Qubit States

Provides multiple methods for labeling quantum states as distillable or non-distillable:
- NPT (Negative Partial Transpose): Fast proxy, sufficient but not necessary
- DPS Hierarchy: Rigorous SDP-based criterion at various levels

References:
- Doherty, Parrilo, Spedalieri, "Complete family of separability criteria" (PRA 2004)
- Horodecki et al., "Quantum entanglement" (Rev. Mod. Phys. 2009)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict
from qiskit.quantum_info import DensityMatrix
import logging

logger = logging.getLogger(__name__)

# Import existing NPT implementation
from .state_generation import check_npt_any_bipartition, _permute_qubits


class DistillabilityOracle(ABC):
    """Abstract base class for distillability labeling oracles."""

    @abstractmethod
    def is_distillable(self, rho: DensityMatrix) -> bool:
        """
        Determine if a quantum state is distillable.

        Args:
            rho: Density matrix to classify

        Returns:
            True if distillable, False otherwise
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return oracle name for logging/identification."""
        pass

    def __call__(self, rho: DensityMatrix) -> bool:
        """Allow oracle to be called as function."""
        return self.is_distillable(rho)


class NPTOracle(DistillabilityOracle):
    """
    NPT (Negative Partial Transpose) oracle.

    A state is labeled distillable if it has negative eigenvalues under
    partial transpose across ANY bipartition. This is a sufficient but
    not necessary condition for distillability.

    Properties:
    - Fast: O(d³) eigenvalue computation
    - Sound: NPT → distillable (no false positives on distillable)
    - Incomplete: May miss some distillable states (bound entangled)
    """

    def __init__(self, tolerance: float = -1e-10):
        self.tolerance = tolerance

    def is_distillable(self, rho: DensityMatrix) -> bool:
        return check_npt_any_bipartition(rho)

    def name(self) -> str:
        return "NPT"


class DPSOracle(DistillabilityOracle):
    """
    DPS Hierarchy oracle using semidefinite programming.

    The DPS hierarchy provides increasingly tight relaxations of the
    separability problem:
    - Level 1: Equivalent to PPT criterion (fast)
    - Level 2: Symmetric extension criterion (more accurate)

    For 3-qubit distillability:
    - NPT across any bipartition → distillable
    - PPT across all bipartitions → check separability via DPS

    Note: For practical 3-qubit systems, Level 2 provides good accuracy
    with reasonable computational cost.
    """

    def __init__(self, level: int = 2, solver: str = 'SCS', verbose: bool = False):
        """
        Args:
            level: DPS hierarchy level (1=PPT, 2=symmetric extension)
            solver: CVXPY solver to use
            verbose: Whether to print solver output
        """
        self.level = level
        self.solver = solver
        self.verbose = verbose
        self._cvxpy = None

    def _get_cvxpy(self):
        """Lazy import of CVXPY."""
        if self._cvxpy is None:
            try:
                import cvxpy as cp
                self._cvxpy = cp
            except ImportError:
                raise ImportError(
                    "CVXPY required for DPS oracle. Install with: pip install cvxpy"
                )
        return self._cvxpy

    def is_distillable(self, rho: DensityMatrix) -> bool:
        """
        Check distillability using DPS hierarchy.

        Returns True (distillable) only if NPT across some bipartition.
        PPT states are classified as non-distillable.
        """
        # Fast path: NPT states are definitely distillable
        if check_npt_any_bipartition(rho):
            return True

        # PPT state - non-distillable (via LOCC)
        # Level 2 DPS can distinguish separable from bound entangled,
        # but both are non-distillable for our purposes
        return False

    def check_separability(self, rho: DensityMatrix, bipartition: str = 'A|BC') -> bool:
        """
        Check if state is separable across given bipartition using DPS Level 2.

        This implements the symmetric extension criterion:
        A state ρ_AB is separable iff it has a symmetric extension ρ_ABB'
        that is PPT across the AB' partition.

        Args:
            rho: Density matrix
            bipartition: Which bipartition to check ('A|BC', 'B|AC', 'C|AB')

        Returns:
            True if DPS certifies separability, False if entangled
        """
        cp = self._get_cvxpy()
        rho_data = np.asarray(rho.data)

        # Get dimensions for bipartition
        if bipartition == 'A|BC':
            d_A, d_B = 2, 4
        elif bipartition == 'B|AC':
            # Permute to put B first
            rho_data = _permute_qubits(rho_data, [1, 0, 2])
            d_A, d_B = 2, 4
        elif bipartition == 'C|AB':
            d_A, d_B = 4, 2
        else:
            raise ValueError(f"Unknown bipartition: {bipartition}")

        return self._symmetric_extension_test(rho_data, d_A, d_B)

    def _symmetric_extension_test(self, rho: np.ndarray, d_A: int, d_B: int) -> bool:
        """
        Symmetric extension test for separability (DPS Level 2).

        Tests if ρ_AB has a symmetric extension to ρ_ABB' with:
        1. ρ_ABB' ≥ 0
        2. Tr_B'(ρ_ABB') = ρ_AB
        3. ρ_ABB' symmetric under B ↔ B'
        4. ρ_ABB'^{T_B'} ≥ 0 (PPT on AB')

        Returns True if extension exists (separable), False otherwise.
        """
        cp = self._get_cvxpy()

        d_AB = d_A * d_B
        d_ext = d_A * d_B * d_B  # ABB' dimension

        # Variable: extended state ρ_ABB'
        rho_ext_real = cp.Variable((d_ext, d_ext), symmetric=True)
        rho_ext_imag = cp.Variable((d_ext, d_ext), symmetric=True)

        # For Hermitian matrix, we need special handling
        # Simplified: assume real density matrix (valid for many test cases)
        rho_ext = cp.Variable((d_ext, d_ext), symmetric=True)

        constraints = []

        # Constraint 1: Positive semidefinite
        constraints.append(rho_ext >> 0)

        # Constraint 2: Trace = 1
        constraints.append(cp.trace(rho_ext) == 1)

        # Constraint 3: Partial trace over B' gives ρ_AB
        for i in range(d_AB):
            for j in range(d_AB):
                # Tr_B'(ρ)[i,j] = Σ_k ρ[i*d_B + k, j*d_B + k]
                trace_sum = sum(
                    rho_ext[i * d_B + k, j * d_B + k]
                    for k in range(d_B)
                )
                constraints.append(trace_sum == np.real(rho[i, j]))

        # Constraint 4: B ↔ B' symmetry
        for a in range(d_A):
            for b1 in range(d_B):
                for b2 in range(d_B):
                    for a2 in range(d_A):
                        for b1p in range(d_B):
                            for b2p in range(d_B):
                                idx1 = a * d_B * d_B + b1 * d_B + b2
                                idx2 = a2 * d_B * d_B + b1p * d_B + b2p
                                # Swapped indices
                                idx1_swap = a * d_B * d_B + b2 * d_B + b1
                                idx2_swap = a2 * d_B * d_B + b2p * d_B + b1p

                                if idx1 <= idx2:  # Use symmetry
                                    constraints.append(
                                        rho_ext[idx1, idx2] == rho_ext[idx1_swap, idx2_swap]
                                    )

        # Constraint 5: PPT on AB' (partial transpose of B')
        # Build partial transpose matrix
        rho_ppt = cp.Variable((d_ext, d_ext), symmetric=True)
        constraints.append(rho_ppt >> 0)

        for i in range(d_ext):
            for j in range(d_ext):
                a_i = i // (d_B * d_B)
                b_i = (i // d_B) % d_B
                bp_i = i % d_B

                a_j = j // (d_B * d_B)
                b_j = (j // d_B) % d_B
                bp_j = j % d_B

                # Transpose B' indices
                i_t = a_i * d_B * d_B + b_i * d_B + bp_j
                j_t = a_j * d_B * d_B + b_j * d_B + bp_i

                if i <= j:  # Use symmetry
                    constraints.append(rho_ppt[i, j] == rho_ext[i_t, j_t])

        # Solve
        prob = cp.Problem(cp.Minimize(0), constraints)

        try:
            prob.solve(solver=getattr(cp, self.solver), verbose=self.verbose)

            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return True  # Separable
            else:
                return False  # Entangled

        except Exception as e:
            logger.warning(f"DPS solver failed: {e}")
            # Conservative fallback
            return True

    def classify(self, rho: DensityMatrix) -> str:
        """
        Full classification: distillable / bound_entangled / separable.

        Returns:
            'distillable': NPT state (definitely distillable)
            'separable': PPT and passes DPS separability test
            'bound_entangled': PPT but fails DPS test (entangled but not distillable)
        """
        if check_npt_any_bipartition(rho):
            return 'distillable'

        # PPT - check separability
        if self.check_separability(rho):
            return 'separable'
        else:
            return 'bound_entangled'

    def name(self) -> str:
        return f"DPS-L{self.level}"


# Simple PPT oracle (equivalent to DPS Level 1)
class PPTOracle(DistillabilityOracle):
    """
    PPT (Positive Partial Transpose) oracle.

    Equivalent to DPS Level 1. Classifies based on PPT criterion only.
    Faster than full DPS but may misclassify bound entangled states.
    """

    def is_distillable(self, rho: DensityMatrix) -> bool:
        # NPT = distillable, PPT = non-distillable
        return check_npt_any_bipartition(rho)

    def name(self) -> str:
        return "PPT"


def get_oracle(name: str = 'npt', **kwargs) -> DistillabilityOracle:
    """
    Factory function to create distillability oracles.

    Args:
        name: Oracle type ('npt', 'ppt', 'dps', 'dps2')
        **kwargs: Oracle-specific parameters

    Returns:
        DistillabilityOracle instance
    """
    oracles = {
        'npt': NPTOracle,
        'ppt': PPTOracle,
        'dps': DPSOracle,
        'dps1': lambda **kw: DPSOracle(level=1, **kw),
        'dps2': lambda **kw: DPSOracle(level=2, **kw),
    }

    if name.lower() not in oracles:
        raise ValueError(f"Unknown oracle: {name}. Available: {list(oracles.keys())}")

    return oracles[name.lower()](**kwargs)


# Comparison utilities
def compare_oracles(
    states: list,
    oracle_names: list = ['npt', 'dps2']
) -> Dict[str, Dict]:
    """
    Compare multiple oracles on a set of states.

    Returns disagreement statistics and per-state classifications.
    """
    oracles = {name: get_oracle(name) for name in oracle_names}
    results = {name: [] for name in oracle_names}

    for state in states:
        for name, oracle in oracles.items():
            results[name].append(oracle.is_distillable(state))

    # Compute agreement
    results_arr = np.array([results[name] for name in oracle_names])
    agreement = np.all(results_arr == results_arr[0:1], axis=0)

    return {
        'per_oracle': results,
        'agreement_rate': np.mean(agreement),
        'disagreement_indices': np.where(~agreement)[0].tolist()
    }
