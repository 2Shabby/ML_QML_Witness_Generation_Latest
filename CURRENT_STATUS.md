# CANONICAL: Current Codebase Status

**Document Status:** CANONICAL
**Version:** 1.0
**Last Updated:** December 17, 2025
**Aligned With:** GOAL.md v1.0

> This document describes the current state of the codebase relative to the canonical research goal: learning restricted witnesses for three-qubit distillability.

---

## Executive Summary

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
1. Distillability oracle for labeling
2. 3-qubit validation
3. QEC-relevant state generators

**Estimated effort to MVP:** 14-20 hours

---

## What the Codebase Does

### Core Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│ Quantum States  │ ──▶ │ Pauli Features   │ ──▶ │ Linear SVM      │ ──▶ │ Witness Operator │
│ (DensityMatrix) │     │ x_ρ = ⟨P₁⟩...⟨Pₙ⟩│     │ Hyperplane w·x+b│     │ W = Σ wₖPₖ       │
└─────────────────┘     └──────────────────┘     └─────────────────┘     └──────────────────┘
        ↑                       ↑                        ↑                       ↑
   NEEDS WORK              ✅ READY                 ✅ READY                 ✅ READY
  (3-qubit, QEC)        (36D restricted)         (SVMWitnessLearner)      (SparsePauliOp)
```

### Key Insight for GOAL.md

The codebase already implements the critical **methodological separation**:
- **Labeling:** Can use any oracle (currently PPT, need to add NPT/DPS)
- **Features:** Restricted to specified Pauli subset (1+2 body ready)
- **Model:** Extracts explicit Hermitian operator, not black-box

---

## Module Status (Detailed)

### 1. Quantum State Generation

**File:** `src/quantum_states/state_generation.py` (335 lines)
**GOAL Alignment:** Partial

| Function | Works? | 3-Qubit? | GOAL Relevance |
|----------|--------|----------|----------------|
| `generate_random_density_matrix(n)` | ✅ | Untested | Random mixed states |
| `generate_separable_state(n)` | ✅ | Untested | Negative examples |
| `generate_entangled_state(n, 'ghz')` | ✅ | Untested | QEC resource |
| `generate_entangled_state(n, 'w')` | ✅ | Untested | QEC resource |
| `generate_bell_state()` | ✅ | N/A (2-qubit) | Not needed |
| `generate_werner_state(n, p)` | ✅ | Untested | Benchmark |
| `generate_dataset(n, samples)` | ✅ | Untested | Training data |
| `check_ppt_criterion(rho, dims)` | ✅ | Works | Part of NPT proxy |
| `partial_transpose(rho, dims)` | ✅ | Works | Core primitive |

**For GOAL.md - What's Missing:**
- [ ] Validation tests for n=3
- [ ] Noisy cluster state generator
- [ ] Bound entangled 3-qubit state generator
- [ ] NPT check across all bipartitions (A|BC, B|AC, C|AB)

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

**Verification for GOAL.md:**
```python
# This should produce exactly 36 features for 3-qubit
from src.feature_extraction.pauli_features import create_sparse_measurement_set, get_pauli_basis

full_basis = get_pauli_basis(3)           # 63 operators
restricted = create_sparse_measurement_set(3, 'two_body')  # Should be 36

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

**Key for GOAL.md:**
- Hyperplane extraction directly maps to Hermitian operator
- Output is Qiskit `SparsePauliOp` - lab-ready
- Coefficients are interpretable measurement weights

---

### 4. MLP Witness Learner

**File:** `src/ml_models/mlp_witness.py` (364 lines)
**GOAL Alignment:** Optional (for comparison)

| Status | Notes |
|--------|-------|
| Architecture | ✅ Builds (TensorFlow/Keras) |
| Training | ⚠️ Untested end-to-end |
| Witness extraction | ❌ No operator extraction (nonlinear) |

**For GOAL.md:** Useful as nonlinear baseline comparison, but NOT primary deliverable.

---

### 5. Utilities

**Status:** ✅ COMPLETE

| Module | Purpose | GOAL Relevance |
|--------|---------|----------------|
| `config_manager.py` | YAML configs | Experiment management |
| `logger.py` | Logging | Debugging |
| `checkpoint_manager.py` | Model saving | Reproducibility |
| `reproducibility.py` | Seed control | Scientific rigor |

---

## Gap Analysis for GOAL.md

### Critical Gaps (Block MVP)

| Gap | Priority | Effort | Description |
|-----|----------|--------|-------------|
| **NPT Distillability Oracle** | P0 | 3-4h | Check NPT across all 3 bipartitions |
| **3-Qubit Validation** | P0 | 2h | Test existing code with n=3 |
| **Dataset Generation** | P0 | 4-6h | Balanced distillable/non-distillable |

### Important Gaps (Strengthen Result)

| Gap | Priority | Effort | Description |
|-----|----------|--------|-------------|
| QEC State Generators | P1 | 4h | Noisy GHZ, W, cluster states |
| DPS Hierarchy | P1 | 8-12h | Rigorous SDP labeling |
| Bound Entangled States | P1 | 4h | Hard negative examples |

### Nice-to-Have Gaps

| Gap | Priority | Effort | Description |
|-----|----------|--------|-------------|
| Noise Models | P2 | 2h | Amplitude damping, dephasing |
| L1 Sparse SVM | P2 | 2h | Measurement-optimal witnesses |
| Ablation Framework | P2 | 4h | Systematic comparison tools |

---

## Implementation Checklist

### Phase 1: Infrastructure (Day 1-2)

```
[ ] 3-Qubit Validation
    [ ] Test generate_dataset(n_qubits=3)
    [ ] Verify state dimensions (8×8 matrices)
    [ ] Test GHZ/W state generation for 3 qubits

[ ] Restricted Feature Validation
    [ ] Verify create_sparse_measurement_set(3, 'two_body') → 36 operators
    [ ] Confirm 1-body (9) + 2-body (27) = 36
    [ ] Test feature extraction produces 36D vectors

[ ] NPT Distillability Oracle
    [ ] Implement check_npt_any_bipartition(rho)
    [ ] Bipartitions: A|BC (1|23), B|AC (2|13), C|AB (3|12)
    [ ] Return: True if NPT across ANY cut → distillable

[ ] End-to-End Pipeline Test
    [ ] states → restricted features → SVM → witness
    [ ] Verify witness is SparsePauliOp with 36 terms max
```

### Phase 2: Dataset (Day 3-4)

```
[ ] State Generators
    [ ] Noisy GHZ: (1-p)|GHZ⟩⟨GHZ| + p·I/8
    [ ] Noisy W: (1-p)|W⟩⟨W| + p·I/8
    [ ] Random near-boundary states

[ ] Labeling
    [ ] Apply NPT oracle to all states
    [ ] Validate against known cases (pure GHZ → distillable)
    [ ] Balance dataset (aim for 50/50 split)

[ ] Dataset Sizes
    [ ] Training: ~4000 states
    [ ] Validation: ~500 states
    [ ] Test: ~500 states
```

### Phase 3: Witness Learning (Day 5-6)

```
[ ] Training
    [ ] Train SVM on 36D restricted features
    [ ] Extract metrics: accuracy, precision, recall

[ ] Witness Extraction
    [ ] Get witness W = Σ wₖPₖ
    [ ] Verify: only 1+2 body terms (no 3-body)
    [ ] Compute measurement cost

[ ] Evaluation
    [ ] Test set performance
    [ ] Per-family breakdown
```

### Phase 4: Analysis (Day 7-10)

```
[ ] Ablation: Restricted vs Full
    [ ] Train on 63D (full) features
    [ ] Compare accuracy: 36D vs 63D
    [ ] Quantify information loss

[ ] Failure Analysis
    [ ] Identify misclassified states
    [ ] Characterize failure modes
    [ ] Physical interpretation
```

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
from src.quantum_states.state_generation import generate_dataset
from src.feature_extraction.pauli_features import create_sparse_measurement_set, extract_features_batch

# Generate 3-qubit states
states, labels = generate_dataset(n_qubits=3, n_samples=10, seed=42)
print(f'State dimension: {states[0].dim}')  # Should be 8

# Get restricted basis
restricted = create_sparse_measurement_set(3, strategy='two_body')
print(f'Restricted basis size: {len(restricted)}')  # Should be 36

# Extract features
features = extract_features_batch(states, restricted, verbose=False)
print(f'Feature shape: {features.shape}')  # Should be (10, 36)
"
```

---

## Dependency Status

**Required and Installed:**
```
qiskit>=1.0.0           ✅ Core quantum operations
tensorflow>=2.15.0      ✅ MLP (optional comparison)
scikit-learn>=1.3.0     ✅ SVM
numpy, scipy            ✅ Numerics
cvxpy>=1.4.0            ✅ Ready for DPS (unused)
pytest                  ✅ Testing
```

**Not Needed for GOAL.md:**
```
qiskit-machine-learning  (VQC/QSVC - not in scope)
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| 3-qubit code has bugs | Low | High | Thorough validation in Phase 1 |
| NPT proxy is too weak | Medium | Medium | Implement DPS as backup |
| Linear SVM insufficient | Medium | Low | This IS the research question |
| Boundary is nonlinear | Medium | None | Negative result is still valuable |

---

## File Structure (GOAL-Relevant)

```
ML_QML_Witness_Generation/
├── GOAL.md                          ✅ CANONICAL research objective
├── CURRENT_STATUS.md                ✅ CANONICAL this document
├── src/
│   ├── quantum_states/
│   │   └── state_generation.py      ⚠️ Needs 3-qubit validation
│   ├── feature_extraction/
│   │   └── pauli_features.py        ✅ Ready (36D restricted)
│   ├── ml_models/
│   │   ├── svm_witness.py           ✅ Ready (witness extraction)
│   │   └── mlp_witness.py           ⚠️ Optional comparison
│   └── utils/                       ✅ Ready
├── tests/                           ⚠️ Need 3-qubit tests
├── config/                          ✅ Ready
└── requirements.txt                 ✅ All dependencies present
```

---

## Summary for Implementation

**Start Here:**
1. Run the quick 3-qubit verification above
2. Implement `check_npt_any_bipartition()` function
3. Generate labeled dataset
4. Train SVM and extract witness

**The codebase is ~70% ready for GOAL.md.** The missing 30% is:
- Distillability oracle (NPT/DPS)
- 3-qubit validation
- QEC state generators

**Critical path:** NPT oracle → Dataset → Training → Analysis

---

*This document is CANONICAL. Updates should maintain alignment with GOAL.md and include version increments.*
