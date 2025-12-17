# CANONICAL: Current Codebase Status

**Document Status:** CANONICAL
**Version:** 5.0
**Last Updated:** December 17, 2025
**Aligned With:** GOAL.md v1.0

> This document describes the current state of the codebase relative to the canonical research goal: learning restricted witnesses for three-qubit distillability.

---

## Executive Summary

### MVP COMPLETE WITH TRANSFORMER EXTENSION

The codebase has **all critical components implemented and verified** for the 3-qubit distillability witness learning pipeline as specified in GOAL.md, now with an additional **transformer-based pipeline** for comparison.

| Metric | Value | Status |
|--------|-------|--------|
| Tests passing | 32/32 (core) | ✅ |
| SVM test accuracy | 87% | ✅ (target: >55%) |
| NPT oracle | Verified correct | ✅ |
| SVM Pipeline | Production ready | ✅ |
| Transformer Pipeline | Ready for testing | ✅ |

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
| **Distillability labeling (NPT)** | `check_npt_any_bipartition()` | ✅ |
| **Distillability dataset** | `generate_distillability_dataset()` | ✅ |
| Distillability labeling (DPS) | Not implemented | ❌ Future |

---

## Audit Results (December 17, 2025)

See [AUDIT_REPORT.md](AUDIT_REPORT.md) for full details.

| Audit Item | Result |
|------------|--------|
| NPT oracle correctness | ✅ Verified |
| Bipartition logic (A|BC, B|AC, C|AB) | ✅ Correct |
| Cluster state (CZ gates, stabilizers) | ✅ Verified |
| Numerical stability | ✅ Passed |
| Pipeline integration | ✅ 87% accuracy |
| Edge cases | ✅ No issues |

**Key Findings:**
- NPT threshold at noise=0.80 for depolarized GHZ/W/cluster states (physically correct)
- Dataset 80/20 balance is expected (product states are only non-distillable family)
- Witness extraction produces valid 3-qubit SparsePauliOp
- Measurement cost reduced from 63 to 12 settings

---

## Current File Structure

```
ML_QML_Witness_Generation/
├── GOAL.md                              ✅ CANONICAL research objective
├── CURRENT_STATUS.md                    ✅ CANONICAL (this document, v5.0)
├── AUDIT_REPORT.md                      ✅ Verification results
├── RESTRUCTURE_PLAN.md                  ✅ Historical reference
├── README.md                            ✅ Project overview
├── requirements.txt                     ✅ Includes PyTorch for transformers
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
│   │   ├── __init__.py                  ✅ Exports SVM + Transformer learners
│   │   ├── svm_witness.py               ✅ Ready (witness extraction)
│   │   └── transformer_witness.py       ✅ NEW: Transformer + Hybrid witness
│   └── utils/
│       └── __init__.py                  ✅ Minimal (set_seed only)
│
├── scripts/
│   ├── run_experiments.py               ✅ SVM experiments
│   └── run_transformer_experiments.py   ✅ NEW: Transformer vs SVM comparison
│
└── tests/
    ├── __init__.py
    ├── test_state_generation.py         ✅ 21 tests (includes NPT oracle)
    ├── test_feature_extraction.py       ✅ 7 tests
    ├── test_integration.py              ✅ 4 tests (includes 3-qubit pipeline)
    └── test_transformer_witness.py      ✅ NEW: Transformer model tests
```

---

## Module Status (Detailed)

### 1. Quantum State Generation

**File:** `src/quantum_states/state_generation.py` (~550 lines)
**GOAL Alignment:** ✅ COMPLETE

| Function | Status | GOAL Relevance |
|----------|--------|----------------|
| `generate_random_density_matrix(n)` | ✅ | Random mixed states |
| `generate_entangled_state(n, 'ghz')` | ✅ | QEC resource |
| `generate_entangled_state(n, 'w')` | ✅ | QEC resource |
| `generate_werner_state(n, p)` | ✅ | Benchmark |
| `generate_noisy_cluster_state(n, noise)` | ✅ | QEC resource |
| `generate_3qubit_product_state(seed)` | ✅ | Separable states |
| `check_npt_any_bipartition(rho)` | ✅ | **Distillability oracle** |
| `generate_distillability_dataset(n, noise_range)` | ✅ | **Labeled dataset** |
| `partial_transpose(rho, dims)` | ✅ | Core primitive |
| `_permute_qubits(rho, perm)` | ✅ | Helper for B|AC bipartition |

### 2. Feature Extraction

**File:** `src/feature_extraction/pauli_features.py` (273 lines)
**GOAL Alignment:** ✅ READY

| Function | Status | GOAL Relevance |
|----------|--------|----------------|
| `get_pauli_basis(n_qubits)` | ✅ | Full basis (63 for 3-qubit) |
| `extract_pauli_features(rho, basis)` | ✅ | Feature vector extraction |
| `extract_features_batch(states, basis)` | ✅ | Efficient batch processing |
| `create_sparse_measurement_set(n, 'two_body')` | ✅ | **36D restricted basis** |
| `group_commuting_paulis(pauli_list)` | ✅ | Measurement optimization |
| `estimate_measurement_cost(pauli_list)` | ✅ | Experimental cost metric |

### 3. SVM Witness Learner

**File:** `src/ml_models/svm_witness.py` (327 lines)
**GOAL Alignment:** ✅ READY

| Method | Status | GOAL Relevance |
|--------|--------|----------------|
| `__init__(pauli_basis, C, kernel='linear')` | ✅ | Configure with restricted basis |
| `train(X, y, test_size)` | ✅ | Learn from labeled data |
| `predict(X)` | ✅ | Binary classification |
| `get_witness_operator()` | ✅ | **W = Σ wₖPₖ as SparsePauliOp** |
| `get_sparse_witness(threshold)` | ✅ | Measurement-efficient witness |
| `get_measurement_cost()` | ✅ | Experimental feasibility metric |

### 4. Transformer Witness Learner (NEW)

**File:** `src/ml_models/transformer_witness.py` (941 lines)
**GOAL Alignment:** ✅ EXTENSION FOR COMPARISON

Provides two architectures for comparison with SVM:

**TransformerClassifier:**
| Feature | Description |
|---------|-------------|
| Architecture | Small transformer (64D, 4 heads, 2 layers) |
| Input | 36D Pauli features as sequence |
| Output | Binary classification |
| Use case | Test if non-linear boundaries improve accuracy |

**HybridTransformerWitness:**
| Feature | Description |
|---------|-------------|
| Architecture | Constrained transformer outputting coefficients |
| Input | 36D Pauli features |
| Output | Classification + interpretable witness W = Σ wₖPₖ |
| Use case | Maintain interpretability while allowing non-linearity |

**TransformerWitnessLearner (Wrapper):**
| Method | Status | Notes |
|--------|--------|-------|
| `__init__(pauli_basis, mode='hybrid')` | ✅ | mode: 'classifier' or 'hybrid' |
| `train(X, y, test_size)` | ✅ | PyTorch training with early stopping |
| `predict(X)` | ✅ | Binary classification |
| `predict_proba(X)` | ✅ | Probability estimates |
| `get_witness_operator()` | ✅ | Only in hybrid mode |
| `get_attention_analysis(X)` | ✅ | Feature importance from attention |
| `save(path)` / `load(path)` | ✅ | Model persistence |

---

## Test Results

```
======================== 32 passed in 4.32s ========================

test_state_generation.py::TestStateGeneration (8 tests)
test_state_generation.py::TestNPTOracleAndDistillability (13 tests)
test_feature_extraction.py::TestFeatureExtraction (7 tests)
test_integration.py::TestIntegration (4 tests, includes 3-qubit pipeline)
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

## Minor Issues (Non-Blocking)

| Issue | Priority | Notes |
|-------|----------|-------|
| `__init__.py` missing new exports | Low | Direct imports work |
| `requirements.txt` has unused deps | Low | TensorFlow, cvxpy not needed |
| pytest warning (test returns dict) | Low | Cosmetic |

---

## What's Next

The codebase is **ready for**:
1. **Transformer vs SVM comparison experiments** (run `scripts/run_transformer_experiments.py`)
2. Larger-scale training experiments (5000+ samples)
3. Ablation studies: 36D restricted vs 63D full features
4. Witness coefficient analysis: compare SVM vs Hybrid Transformer witnesses
5. Noise robustness characterization

**Experiment Commands:**
```bash
# Quick comparison
python scripts/run_transformer_experiments.py --experiment comparison --n-samples 5000

# Full cross-validation
python scripts/run_transformer_experiments.py --experiment cv --n-samples 5000

# Scaling study
python scripts/run_transformer_experiments.py --experiment scaling

# All experiments
python scripts/run_transformer_experiments.py --experiment all --n-samples 10000
```

**Future work (optional):**
- DPS hierarchy for rigorous SDP-based labeling
- L1-regularized sparse SVM for minimal witnesses
- Bound entanglement detection (requires different dataset)
- Attention pattern analysis for feature importance

---

## Summary

**The codebase is MVP-complete and audit-verified for GOAL.md, with transformer extension.**

- ✅ NPT distillability oracle implemented and verified
- ✅ All 3-qubit state generators working
- ✅ Distillability dataset generator with correct labels
- ✅ End-to-end SVM pipeline: 3-qubit states → 36D features → SVM → witness
- ✅ End-to-end Transformer pipeline: 3-qubit states → 36D features → Transformer → classification/witness
- ✅ 32 core tests passing, 87% SVM test accuracy
- ✅ Transformer tests ready (requires PyTorch)

---

*This document is CANONICAL. Updates should maintain alignment with GOAL.md and include version increments.*
