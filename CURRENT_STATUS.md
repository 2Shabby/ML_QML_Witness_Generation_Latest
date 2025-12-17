# CANONICAL: Current Codebase Status

**Document Status:** CANONICAL
**Version:** 3.0
**Last Updated:** December 17, 2025
**Aligned With:** GOAL.md v1.0

> This document describes the current state of the codebase relative to the canonical research goal: learning restricted witnesses for three-qubit distillability.

---

## Executive Summary

### MVP COMPLETE

The codebase now has **all critical components implemented** for the 3-qubit distillability witness learning pipeline as specified in GOAL.md.

| Metric | Before (v2.0) | After (v3.0) | Change |
|--------|---------------|--------------|--------|
| Source files | 5 | 5 | Same |
| Lines of code | ~950 | ~1,200 | +26% (new features) |
| Tests passing | 19 | 32 | +68% |
| GOAL.md alignment | Partial | **Complete** | MVP Ready |

### Alignment with GOAL.md

| GOAL Requirement | Codebase Status | Ready? |
|------------------|-----------------|--------|
| 3-qubit quantum states | `generate_entangled_state(3, 'ghz'/'w')` validated | ✅ |
| Cluster states | `generate_noisy_cluster_state()` | ✅ |
| Product states | `generate_3qubit_product_state()` | ✅ |
| Restricted features (36D, 1+2 body) | `create_sparse_measurement_set('two_body')` | ✅ |
| Linear SVM classifier | `SVMWitnessLearner` | ✅ |
| Witness extraction as operator | `get_witness_operator() → SparsePauliOp` | ✅ |
| Measurement cost estimation | `group_commuting_paulis()` | ✅ |
| **Distillability labeling (NPT)** | `check_npt_any_bipartition()` | ✅ NEW |
| **Distillability dataset** | `generate_distillability_dataset()` | ✅ NEW |
| Distillability labeling (DPS) | Not implemented | ❌ Future |

### Bottom Line

**The complete MVP pipeline is now operational:**
- 3-qubit states → 36D features → SVM → witness
- NPT oracle correctly identifies distillable states
- All 32 tests passing

**Ready for:** Training experiments, ablation studies, witness analysis

---

## What Was Implemented (This Session)

### New Functions in `state_generation.py`

| Function | Lines | Purpose |
|----------|-------|---------|
| `check_npt_any_bipartition(rho)` | ~45 | **CRITICAL:** NPT distillability oracle for 3-qubit states |
| `_permute_qubits(rho, perm)` | ~20 | Helper for B|AC bipartition check |
| `generate_noisy_cluster_state(n, noise)` | ~50 | Linear cluster state with depolarizing noise |
| `generate_3qubit_product_state(seed)` | ~30 | Random 3-qubit product (separable) state |
| `generate_distillability_dataset(n, noise_range)` | ~80 | **CRITICAL:** Labeled dataset for distillability |

**Total new code:** ~225 lines

### New Tests

| Test Class | Tests | Coverage |
|------------|-------|----------|
| `TestNPTOracleAndDistillability` | 13 | NPT oracle, state generators, dataset |
| `test_3qubit_distillability_pipeline` | 1 | End-to-end integration |

**Total new tests:** 14 tests (all passing)

---

## Current File Structure

```
ML_QML_Witness_Generation/
├── GOAL.md                              ✅ CANONICAL research objective
├── CURRENT_STATUS.md                    ✅ CANONICAL (this document, v3.0)
├── RESTRUCTURE_PLAN.md                  ✅ Debloating plan (reference)
├── requirements.txt                     ⚠️ Needs update (remove TensorFlow, etc.)
│
├── src/
│   ├── __init__.py
│   ├── quantum_states/
│   │   ├── __init__.py
│   │   └── state_generation.py          ✅ NPT oracle + all generators READY
│   ├── feature_extraction/
│   │   ├── __init__.py
│   │   └── pauli_features.py            ✅ Ready (36D restricted)
│   ├── ml_models/
│   │   ├── __init__.py                  ✅ Updated (SVM only)
│   │   └── svm_witness.py               ✅ Ready (witness extraction)
│   └── utils/
│       └── __init__.py                  ✅ Minimal (set_seed only)
│
└── tests/
    ├── __init__.py
    ├── test_state_generation.py         ✅ 21 tests (includes NPT oracle)
    ├── test_feature_extraction.py       ✅ 7 tests
    └── test_integration.py              ✅ 4 tests (includes 3-qubit pipeline)
```

---

## Module Status (Detailed)

### 1. Quantum State Generation

**File:** `src/quantum_states/state_generation.py` (~550 lines)
**GOAL Alignment:** ✅ COMPLETE

| Function | Works? | 3-Qubit? | GOAL Relevance |
|----------|--------|----------|----------------|
| `generate_random_density_matrix(n)` | ✅ | ✅ Tested | Random mixed states |
| `generate_entangled_state(n, 'ghz')` | ✅ | ✅ Tested | QEC resource |
| `generate_entangled_state(n, 'w')` | ✅ | ✅ Tested | QEC resource |
| `generate_werner_state(n, p)` | ✅ | ✅ Tested | Benchmark |
| `generate_noisy_cluster_state(n, noise)` | ✅ | ✅ Tested | **NEW: QEC resource** |
| `generate_3qubit_product_state(seed)` | ✅ | ✅ Tested | **NEW: Separable states** |
| `check_npt_any_bipartition(rho)` | ✅ | ✅ Tested | **NEW: Distillability oracle** |
| `generate_distillability_dataset(n, noise_range)` | ✅ | ✅ Tested | **NEW: Correct labels** |
| `partial_transpose(rho, dims)` | ✅ | ✅ Works | Core primitive |
| `_permute_qubits(rho, perm)` | ✅ | ✅ Works | Helper for B|AC bipartition |

**NPT Oracle Validation:**
- Pure GHZ → Distillable (NPT) ✅
- Pure W → Distillable (NPT) ✅
- Pure Cluster → Distillable (NPT) ✅
- Product states → NOT Distillable (PPT) ✅
- Noisy states → Threshold behavior confirmed ✅

---

### 2. Feature Extraction

**File:** `src/feature_extraction/pauli_features.py` (273 lines)
**GOAL Alignment:** ✅ READY (unchanged)

| Function | Status | GOAL Relevance |
|----------|--------|----------------|
| `get_pauli_basis(n_qubits)` | ✅ Ready | Full basis (63 for 3-qubit) |
| `extract_pauli_features(rho, basis)` | ✅ Ready | Feature vector extraction |
| `extract_features_batch(states, basis)` | ✅ Ready | Efficient batch processing |
| `create_sparse_measurement_set(n, 'two_body')` | ✅ Ready | **CRITICAL: 36D restricted basis** |
| `group_commuting_paulis(pauli_list)` | ✅ Ready | Measurement optimization |
| `estimate_measurement_cost(pauli_list)` | ✅ Ready | Experimental cost metric |

---

### 3. SVM Witness Learner

**File:** `src/ml_models/svm_witness.py` (327 lines)
**GOAL Alignment:** ✅ READY (unchanged)

| Method | Status | GOAL Relevance |
|--------|--------|----------------|
| `__init__(pauli_basis, C, kernel='linear')` | ✅ | Configure with restricted basis |
| `train(X, y, test_size)` | ✅ | Learn from labeled data |
| `predict(X)` | ✅ | Binary classification |
| `get_witness_operator()` | ✅ | **CRITICAL: W = Σ wₖPₖ as SparsePauliOp** |
| `get_sparse_witness(threshold)` | ✅ | Measurement-efficient witness |
| `get_measurement_cost()` | ✅ | Experimental feasibility metric |

---

## Test Results Summary

```
======================== 32 passed in 4.12s ========================

test_state_generation.py::TestStateGeneration (8 tests)
test_state_generation.py::TestNPTOracleAndDistillability (13 tests)  ← NEW
test_feature_extraction.py::TestFeatureExtraction (7 tests)
test_integration.py::TestIntegration (4 tests, includes 3-qubit pipeline)  ← NEW
```

---

## Quick Start Commands

```bash
# Setup
cd /home/user/ML_QML_Witness_Generation

# Run all tests
python3 -m pytest tests/ -v

# Quick 3-qubit distillability verification
python3 -c "
from src.quantum_states.state_generation import (
    generate_entangled_state,
    generate_3qubit_product_state,
    generate_noisy_cluster_state,
    check_npt_any_bipartition,
    generate_distillability_dataset
)

# Test NPT oracle on known states
ghz = generate_entangled_state(3, 'ghz', noise_level=0.0)
print(f'Pure GHZ distillable: {check_npt_any_bipartition(ghz)}')  # True

product = generate_3qubit_product_state(seed=42)
print(f'Product distillable: {check_npt_any_bipartition(product)}')  # False

cluster = generate_noisy_cluster_state(3, noise_level=0.0)
print(f'Pure cluster distillable: {check_npt_any_bipartition(cluster)}')  # True

# Generate small dataset
states, labels = generate_distillability_dataset(n_samples=100, seed=42)
print(f'Dataset: {len(states)} states, {sum(labels)} distillable')
"

# Run end-to-end pipeline test
python3 -m pytest tests/test_integration.py::TestIntegration::test_3qubit_distillability_pipeline -v -s
```

---

## Gap Analysis for GOAL.md

### Completed (This Session)

| Item | Status | Notes |
|------|--------|-------|
| NPT Distillability Oracle | ✅ Done | All 3 bipartitions checked |
| 3-Qubit Validation | ✅ Done | GHZ, W, cluster, product tested |
| Distillability Dataset | ✅ Done | 5 state families, correct labels |
| Cluster State Generator | ✅ Done | Linear cluster with noise |
| 3-Qubit Product States | ✅ Done | Random Bloch sphere sampling |
| End-to-End Pipeline Test | ✅ Done | 500 states, >55% accuracy |

### Remaining (Future Work)

| Gap | Priority | Effort | Description |
|-----|----------|--------|-------------|
| Update requirements.txt | P2 | 0.5h | Remove unused dependencies |
| DPS Hierarchy | P1 | 8-12h | Rigorous SDP labeling (optional) |
| L1 Sparse SVM | P2 | 2h | Measurement-optimal witnesses |
| Larger Dataset Experiments | P1 | 2-4h | 5000+ samples, ablation studies |
| Witness Analysis | P1 | 2-4h | Coefficient interpretation, visualizations |

---

## Risk Assessment (Updated)

| Risk | Likelihood | Impact | Status |
|------|------------|--------|--------|
| 3-qubit code has bugs | Low | High | ✅ Mitigated (13 new tests) |
| NPT proxy is too weak | Medium | Medium | Monitor; DPS backup available |
| Linear SVM insufficient | Medium | Low | This IS the research question |
| Boundary is nonlinear | Medium | None | Negative result is valuable |

---

## Summary

**The codebase is now MVP-complete for GOAL.md.**

- ✅ NPT distillability oracle implemented and tested
- ✅ All 3-qubit state generators (GHZ, W, cluster, product) working
- ✅ Distillability dataset generator with correct labels
- ✅ End-to-end pipeline tested: 3-qubit states → 36D features → SVM → witness
- ✅ 32 tests passing

**Next steps:** Training experiments, ablation studies, witness coefficient analysis

---

*This document is CANONICAL. Updates should maintain alignment with GOAL.md and include version increments.*
