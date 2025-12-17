# CANONICAL: Current Codebase Status

**Document Status:** CANONICAL
**Version:** 2.0
**Last Updated:** December 17, 2025
**Aligned With:** GOAL.md v1.0

> This document describes the current state of the codebase relative to the canonical research goal: learning restricted witnesses for three-qubit distillability.

---

## Executive Summary

### Post-Debloating Status

The codebase has been **debloated** to focus exclusively on GOAL.md. Dead weight code (MLP, over-engineered utilities, hierarchical configs) has been removed.

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Source files | 9 | 5 | -44% |
| Lines of code | ~1,700 | ~950 | -44% |
| Config files | 6 | 0 | -100% |
| Dependencies needed | 16 | 8 | -50% |

### Alignment with GOAL.md

| GOAL Requirement | Codebase Status | Ready? |
|------------------|-----------------|--------|
| 3-qubit quantum states | Code exists, needs validation | ⚠️ |
| Restricted features (36D, 1+2 body) | `create_sparse_measurement_set('two_body')` | ✅ |
| Linear SVM classifier | `SVMWitnessLearner` | ✅ |
| Witness extraction as operator | `get_witness_operator() → SparsePauliOp` | ✅ |
| Measurement cost estimation | `group_commuting_paulis()` | ✅ |
| Distillability labeling (NPT) | Not implemented | ❌ |
| Distillability labeling (DPS) | Not implemented | ❌ |
| QEC state families | Not implemented | ❌ |

### Bottom Line

**The core pipeline (features → SVM → witness) is ready.** The main gaps are:
1. NPT distillability oracle for labeling
2. 3-qubit validation
3. QEC-relevant state generators (cluster states, etc.)

**Estimated effort to MVP:** 10-14 hours (reduced from 14-20 due to debloating)

---

## What Was Removed (Debloating)

### Deleted Source Files

| File | Lines | Reason |
|------|-------|--------|
| `src/ml_models/mlp_witness.py` | 364 | GOAL specifies linear SVM; MLP is optional future work |
| `src/utils/checkpoint_manager.py` | 165 | Over-engineered; SVM trains in seconds |
| `src/utils/logger.py` | 103 | Over-engineered; standard logging sufficient |
| `src/utils/config_manager.py` | 185 | Hierarchical YAML configs unnecessary for focused project |
| `src/utils/reproducibility.py` | 78 | Set PyTorch/TensorFlow seeds unnecessarily |

### Deleted Config Files

| File | Reason |
|------|--------|
| `config/defaults.yaml` | Over-engineered config system |
| `config/experiment/bound_entanglement_3x3.yaml` | 3×3 qutrit, not 3-qubit |
| `config/experiment/incomplete_measurements.yaml` | 2-qubit focus |
| `config/model/kan.yaml` | KAN not implemented |
| `config/model/mlp.yaml` | MLP deleted |
| `config/model/svm.yaml` | Single hardcoded config sufficient |

### Deleted Test Files

| File | Reason |
|------|--------|
| `tests/test_mlp_witness.py` | Tests deleted MLP code |
| `tests/test_utils.py` | Tests deleted utilities |

---

## Current File Structure

```
ML_QML_Witness_Generation/
├── GOAL.md                              ✅ CANONICAL research objective
├── CURRENT_STATUS.md                    ✅ CANONICAL (this document)
├── RESTRUCTURE_PLAN.md                  ✅ Debloating plan (reference)
├── requirements.txt                     ⚠️ Needs update (remove TensorFlow, etc.)
│
├── src/
│   ├── __init__.py
│   ├── quantum_states/
│   │   ├── __init__.py
│   │   └── state_generation.py          ⚠️ Needs NPT oracle + 3-qubit validation
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
    ├── test_state_generation.py         ✅ Existing tests
    ├── test_feature_extraction.py       ✅ Existing tests
    └── test_integration.py              ✅ End-to-end pipeline tests
```

---

## Module Status (Detailed)

### 1. Quantum State Generation

**File:** `src/quantum_states/state_generation.py` (335 lines)
**GOAL Alignment:** Partial

| Function | Works? | 3-Qubit? | GOAL Relevance |
|----------|--------|----------|----------------|
| `generate_random_density_matrix(n)` | ✅ | Untested | Random mixed states |
| `generate_entangled_state(n, 'ghz')` | ✅ | Untested | QEC resource |
| `generate_entangled_state(n, 'w')` | ✅ | Untested | QEC resource |
| `generate_werner_state(n, p)` | ✅ | Untested | Benchmark |
| `generate_separable_state(n)` | ⚠️ | **Fails** (even n only) | Needs replacement |
| `generate_bell_state()` | ✅ | N/A (2-qubit) | Not needed for goal |
| `generate_dataset(n, samples)` | ⚠️ | Untested | **Wrong labels** (entangled, not distillable) |
| `check_ppt_criterion(rho, dims)` | ✅ | Works | Part of NPT proxy |
| `partial_transpose(rho, dims)` | ✅ | Works | Core primitive |

**What's Missing:**
- [ ] `check_npt_any_bipartition(rho)` - NPT across all 3 bipartitions
- [ ] `generate_noisy_cluster_state(n, noise)` - QEC resource
- [ ] `generate_3qubit_product_state()` - Proper separable for n=3
- [ ] `generate_distillability_dataset()` - Correct labeling

---

### 2. Feature Extraction

**File:** `src/feature_extraction/pauli_features.py` (273 lines)
**GOAL Alignment:** ✅ READY

| Function | Status | GOAL Relevance |
|----------|--------|----------------|
| `get_pauli_basis(n_qubits)` | ✅ Ready | Full basis (63 for 3-qubit) |
| `extract_pauli_features(rho, basis)` | ✅ Ready | Feature vector extraction |
| `extract_features_batch(states, basis)` | ✅ Ready | Efficient batch processing |
| `create_sparse_measurement_set(n, 'two_body')` | ✅ Ready | **CRITICAL: 36D restricted basis** |
| `group_commuting_paulis(pauli_list)` | ✅ Ready | Measurement optimization |
| `estimate_measurement_cost(pauli_list)` | ✅ Ready | Experimental cost metric |

**Verification:**
```python
from src.feature_extraction.pauli_features import create_sparse_measurement_set, get_pauli_basis

full_basis = get_pauli_basis(3)           # 63 operators
restricted = create_sparse_measurement_set(3, 'two_body')  # 36 operators

# 1-body: 3 qubits × 3 Paulis = 9
# 2-body: 3 pairs × 9 combinations = 27
# Total: 36
```

---

### 3. SVM Witness Learner

**File:** `src/ml_models/svm_witness.py` (327 lines)
**GOAL Alignment:** ✅ READY

| Method | Status | GOAL Relevance |
|--------|--------|----------------|
| `__init__(pauli_basis, C, kernel='linear')` | ✅ | Configure with restricted basis |
| `train(X, y, test_size)` | ✅ | Learn from labeled data |
| `predict(X)` | ✅ | Binary classification |
| `get_witness_operator()` | ✅ | **CRITICAL: W = Σ wₖPₖ as SparsePauliOp** |
| `get_sparse_witness(threshold)` | ✅ | Measurement-efficient witness |
| `get_measurement_cost()` | ✅ | Experimental feasibility metric |

**Key Properties:**
- Hyperplane extraction directly maps to Hermitian operator
- Output is Qiskit `SparsePauliOp` - lab-ready
- Coefficients are interpretable measurement weights

---

### 4. Utilities

**File:** `src/utils/__init__.py` (18 lines)
**GOAL Alignment:** ✅ MINIMAL

| Function | Purpose |
|----------|---------|
| `set_seed(seed)` | Set numpy and random seeds for reproducibility |

---

## Gap Analysis for GOAL.md

### Critical Gaps (Block MVP)

| Gap | Priority | Effort | Description |
|-----|----------|--------|-------------|
| **NPT Distillability Oracle** | P0 | 3-4h | Check NPT across all 3 bipartitions |
| **3-Qubit Validation** | P0 | 2h | Test existing code with n=3 |
| **Distillability Dataset** | P0 | 4-6h | Replace entanglement labels with distillability |

### Important Gaps (Strengthen Result)

| Gap | Priority | Effort | Description |
|-----|----------|--------|-------------|
| Cluster State Generator | P1 | 2h | Noisy 3-qubit cluster states |
| 3-Qubit Separable Generator | P1 | 1h | Product states for n=3 |
| DPS Hierarchy | P1 | 8-12h | Rigorous SDP labeling |

### Nice-to-Have

| Gap | Priority | Effort | Description |
|-----|----------|--------|-------------|
| Update requirements.txt | P2 | 0.5h | Remove unused dependencies |
| L1 Sparse SVM | P2 | 2h | Measurement-optimal witnesses |

---

## Quick Start Commands

```bash
# Setup
cd /home/user/ML_QML_Witness_Generation
source venv/bin/activate

# Run existing tests
pytest tests/ -v

# Quick 3-qubit verification
python -c "
from src.quantum_states.state_generation import generate_entangled_state
from src.feature_extraction.pauli_features import create_sparse_measurement_set, extract_pauli_features

# Generate 3-qubit GHZ state
ghz = generate_entangled_state(3, 'ghz', noise_level=0.1)
print(f'State dimension: {ghz.dim}')  # Should be 8

# Get restricted basis
restricted = create_sparse_measurement_set(3, strategy='two_body')
print(f'Restricted basis size: {len(restricted)}')  # Should be 36

# Extract features
features = extract_pauli_features(ghz, restricted)
print(f'Feature vector length: {len(features)}')  # Should be 36
"
```

---

## Dependency Status

### Required (Core)
```
qiskit>=1.0.0           ✅ Core quantum operations
scikit-learn>=1.3.0     ✅ SVM
numpy>=1.24.0           ✅ Numerics
scipy>=1.11.0           ✅ Linear algebra
pytest>=7.4.0           ✅ Testing
```

### Optional (Nice-to-Have)
```
matplotlib>=3.7.0       ⚠️ For visualization (optional)
tqdm>=4.65.0            ⚠️ Progress bars (optional)
joblib>=1.3.0           ⚠️ Model serialization (optional)
```

### To Remove from requirements.txt
```
tensorflow>=2.15.0      ❌ MLP deleted
qiskit-machine-learning ❌ VQC/QSVC not in scope
qiskit-algorithms       ❌ Not needed
cvxpylayers             ❌ Not needed
pandas                  ❌ numpy sufficient
seaborn                 ❌ matplotlib sufficient
hydra-core              ❌ Config system deleted
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| 3-qubit code has bugs | Low | High | Thorough validation |
| NPT proxy is too weak | Medium | Medium | Implement DPS as backup |
| Linear SVM insufficient | Medium | Low | This IS the research question |
| Boundary is nonlinear | Medium | None | Negative result is valuable |

---

## Implementation Priorities

### Immediate (Next Steps)

1. **Implement NPT Oracle** (`check_npt_any_bipartition`)
   - Check all 3 bipartitions: A|BC, B|AC, C|AB
   - Use existing `partial_transpose()` primitive
   - Return True if NPT across ANY cut

2. **Add 3-Qubit State Generators**
   - `generate_noisy_cluster_state(n_qubits=3, noise_level)`
   - `generate_3qubit_product_state()`

3. **Create Distillability Dataset Generator**
   - Replace `generate_dataset()` with distillability-labeled version
   - Use NPT oracle for labeling

4. **Validate End-to-End Pipeline**
   - 3-qubit states → 36D features → SVM → witness

### Future (After MVP)

- Update `requirements.txt` to remove unused dependencies
- Implement DPS hierarchy for rigorous labeling
- Add L1 regularization for sparse witnesses

---

## Summary

**The codebase is now lean and focused on GOAL.md.**

- Core pipeline (features → SVM → witness) is **100% ready**
- State generation needs **NPT oracle** and **3-qubit generators**
- Dead weight removed: MLP, configs, over-engineered utils

**Critical path:** NPT oracle → Distillability dataset → Training → Analysis

---

*This document is CANONICAL. Updates should maintain alignment with GOAL.md and include version increments.*
