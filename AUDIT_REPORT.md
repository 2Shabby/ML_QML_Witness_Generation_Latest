# NPT Oracle & 3-Qubit Pipeline Audit Report

**Date:** December 17, 2025
**Auditor:** Claude (Automated Audit Session)
**Branch:** claude/audit-npt-oracle-QYAkM

---

## Executive Summary

All 32 tests pass. The NPT distillability oracle is mathematically correct and the complete 3-qubit distillability witness learning pipeline is operational. Test accuracy of **87%** significantly exceeds the 55% threshold. No critical bugs were found.

---

## Audit Results Summary

| Criterion | Result | Notes |
|-----------|--------|-------|
| All 32 tests pass | ✅ PASS | 100% pass rate |
| NPT oracle logic correct | ✅ PASS | All bipartitions verified |
| Dataset class balance | ✅ PASS | 80/20 split is physically expected |
| Pipeline accuracy >55% | ✅ PASS | Achieved 87% |
| No obvious bugs | ✅ PASS | Edge cases verified |

---

## Detailed Findings

### 1. NPT Oracle Correctness (check_npt_any_bipartition)

**Status:** ✅ Verified Correct

- **Maximally mixed state:** Correctly identified as PPT (non-distillable)
- **Partial transpose:** Eigenvalue computation verified (Bell state gives -0.5)
- **Three bipartitions:** All correctly checked (A|BC, B|AC, C|AB)
- **Permutation helper:** `_permute_qubits()` preserves trace, Hermiticity, and positive semidefiniteness
- **Double permutation:** Returns to original state correctly
- **Threshold:** NPT threshold at noise=0.80 is physically correct for depolarized GHZ

### 2. Cluster State Generation (generate_noisy_cluster_state)

**Status:** ✅ Verified Correct

- **CZ gate application:** Matches manual calculation exactly (||diff||_F = 0)
- **Stabilizer eigenvalues:** State is +1 eigenstate of all 3 generators:
  - X₀Z₁I₂ → +1
  - Z₀X₁Z₂ → +1
  - I₀Z₁X₂ → +1
- **Purity:** Rank-1 pure state when noise=0
- **Entanglement:** Correctly NPT (distillable)

### 3. Dataset Balance Analysis

**Status:** ⚠️ Expected Behavior (Not a Bug)

| Noise Range | % Distillable | % Non-Distillable |
|-------------|---------------|-------------------|
| (0.0, 0.2)  | 80.0%         | 20.0%             |
| (0.0, 0.3)  | 80.0%         | 20.0%             |
| (0.0, 0.5)  | 80.0%         | 20.0%             |
| (0.0, 0.7)  | 80.0%         | 20.0%             |

**Explanation:**
- NPT threshold for GHZ/W/Cluster is ~80% noise (verified)
- Random mixed states are almost always NPT (~100% distillable)
- Only product states (1/5 of dataset = 20%) are guaranteed PPT
- This is physically correct behavior

### 4. Numerical Stability

**Status:** ✅ Verified Stable

- Eigenvalue threshold `-1e-10` is appropriate
- Consistent results across random seeds (tested: 0, 42, 100, 12345, 99999)
- Sharp transition at noise=0.80:
  - noise=0.7999: min_eigenvalue=-6.25e-05, NPT=True
  - noise=0.8000: min_eigenvalue=+5.55e-17, NPT=False
- No false positives for product states (10/10 correctly PPT)

### 5. Pipeline Integration

**Status:** ✅ Verified Working

| Metric | Value | Target |
|--------|-------|--------|
| Test Accuracy | 87.0% | >55% |
| Test Precision | 86.8% | - |
| Test Recall | 98.8% | - |
| Measurement Settings | 12 | <63 |

- Witness extraction produces valid `SparsePauliOp`
- 36D restricted features in valid range [-1, 1]
- Significant reduction from 63 full tomography settings to 12

### 6. Minor Issues Found

| Issue | Severity | Impact |
|-------|----------|--------|
| `__init__.py` missing new exports | Low | Non-blocking; direct imports work |
| pytest warning (test returns dict) | Cosmetic | No functional impact |

---

## Distillability Threshold Analysis

Tested noise levels where states transition from distillable (NPT) to non-distillable (PPT):

| State Family | Threshold Noise | Notes |
|--------------|-----------------|-------|
| GHZ | ~0.80 | Sharp transition |
| W | ~0.80 | Same as GHZ |
| Cluster | ~0.80 | Same as GHZ |
| Random Mixed | ~never PPT | Almost always NPT |
| Product | always PPT | By construction |

---

## Commands Run

```bash
# All tests
python3 -m pytest tests/ -v
# Result: 32 passed, 1 warning

# Integration pipeline
python3 -m pytest tests/test_integration.py::TestIntegration::test_3qubit_distillability_pipeline -v -s
# Result: PASSED

# Dataset balance check
python3 -c "
from src.quantum_states.state_generation import generate_distillability_dataset
import numpy as np
for noise_max in [0.2, 0.3, 0.5, 0.7]:
    _, labels = generate_distillability_dataset(1000, noise_range=(0.0, noise_max), seed=42)
    print(f'noise_range=(0.0, {noise_max}): {np.mean(labels)*100:.1f}% distillable')
"
```

---

## Recommendations

1. **Optional:** Update `src/quantum_states/__init__.py` to export new functions
2. **Consider:** Dataset rebalancing if more non-distillable states are needed
3. **Ready for:** Larger-scale training experiments (5000+ samples)
4. **Ready for:** Ablation studies comparing 36D restricted vs 63D full features

---

## Conclusion

**The NPT oracle implementation is verified correct and the 3-qubit distillability pipeline is production-ready.**

All success criteria have been met:
- ✅ All 32 tests pass
- ✅ NPT oracle logic confirmed correct
- ✅ Dataset produces reasonable class balance
- ✅ Integration pipeline achieves >55% accuracy (87%)
- ✅ No obvious bugs or edge cases identified

The codebase is ready for training experiments and ablation studies as outlined in GOAL.md.
