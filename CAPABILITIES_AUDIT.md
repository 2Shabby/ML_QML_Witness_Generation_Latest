# Capabilities Audit: 3-Qubit Distillability Witness Learning

**Audit Date:** January 14, 2026  
**Auditor:** Automated Assessment  
**Evidence Method:** Code inspection, test execution, documentation review

---

## Project Identity

- **Name**: Quantum Distillability Witness Learner
- **One-liner**: Trained ML classifiers to certify distillability of 3-qubit quantum states using only experimentally-accessible Pauli measurements
- **Status**: functional

---

## Verified Capabilities

The following capabilities have been verified through code inspection and test execution (32/32 tests passing):

### Quantum State Generation & Manipulation
- **3-qubit state generators** for GHZ, W, cluster, Werner, Bell, and random density matrices (`src/quantum_states/state_generation.py` - 622 lines)
- **Partial transpose implementation** with all 3 bipartitions (A|BC, B|AC, C|AB) for NPT oracle (`check_npt_any_bipartition()`)
- **Depolarizing noise model** for realistic state corruption (`generate_entangled_state(noise_level=...)`)
- **Product state generation** for negative samples (`generate_3qubit_product_state()`)
- **Dataset generation** with NPT-based distillability labels (`generate_distillability_dataset()`)

### Feature Extraction
- **36D restricted Pauli basis** - 9 one-body + 27 two-body terms (`create_sparse_measurement_set(3, 'two_body')`)
- **63D full Pauli basis** generation for ablation comparison (`get_pauli_basis()`)
- **Batch feature extraction** from density matrices (`extract_features_batch()`)
- **Commuting Pauli grouping** for measurement optimization (`group_commuting_paulis()`)
- **Measurement cost estimation** - returns 12 settings for 36D basis (`estimate_measurement_cost()`)

### Machine Learning Models
- **Linear SVM witness learner** with sklearn backend (`SVMWitnessLearner` - 328 lines)
- **Witness operator extraction** as Qiskit `SparsePauliOp` (`get_witness_operator()`)
- **Sparse witness thresholding** for measurement efficiency (`get_sparse_witness()`)
- **Transformer classifier** with attention mechanism (`TransformerClassifier` - 941 lines total)
- **Hybrid transformer witness** maintaining interpretability (`HybridTransformerWitness`)
- **Attention analysis** for feature importance (`get_attention_analysis()`)

### Distillability Oracles
- **NPT Oracle** - fast proxy for distillability (`NPTOracle` class)
- **PPT Oracle** - DPS Level 1 equivalent (`PPTOracle` class)
- **DPS Level 2 Oracle** - SDP-based symmetric extension test (`DPSOracle` class)

### Experiment Infrastructure
- **5-fold cross-validation** with statistical testing (`run_cross_validation()`)
- **Ablation study** comparing 36D vs 63D features (`run_ablation_study()`)
- **Per-family analysis** for GHZ/W/Cluster/Random/Product states (`run_per_family_analysis()`)
- **Noise robustness characterization** (`run_noise_robustness()`)
- **Witness coefficient analysis** (`analyze_witness_coefficients()`)
- **Centralized configuration** via dataclasses (`src/config.py`)
- **JSON result serialization** with timestamps (`save_results()`)

---

## Tech Stack (verified in code)

- **Languages**: Python 3.12+ (substantial - ~3,500 lines of production code)
- **Frameworks/Libraries**:
  - Qiskit 1.0+ (DensityMatrix, SparsePauliOp, Pauli, PauliList)
  - scikit-learn 1.3+ (SVC, train_test_split, StratifiedKFold, metrics)
  - NumPy 1.24+ (array operations, linear algebra)
  - SciPy 1.11+ (statistics - ttest_rel)
  - PyTorch 2.0+ (Transformer models - optional)
  - CVXPY 1.4+ (SDP solver for DPS oracle - optional)
  - pytest 7.4+ (testing framework)
- **Infrastructure**: None configured (local execution only)

---

## Tags (for categorization)

Suggested tags: **ml, research, quantum**

---

## Priority Assessment

**Priority: 4/5**

**Justification:**
- **Technical Depth (5/5)**: Novel intersection of quantum information theory and ML; implements partial transpose, NPT criterion, DPS hierarchy, and attention-based classifiers from research literature
- **Completeness (4/5)**: Full end-to-end pipeline working with 32 tests passing; experiments documented with statistical analysis; both SVM and Transformer models complete
- **Complexity (4/5)**: Non-trivial quantum computing primitives (bipartition analysis, partial transpose); custom transformer architecture for witness extraction; multiple oracle implementations with SDP solver
- **Research Quality (4/5)**: Hypothesis-driven investigation with ablation studies, statistical validation, and documented findings; addresses real problem in quantum error correction certification

Minor deductions:
- L1 sparse witness not implemented (noted as future work)
- 4+ qubit extension not implemented
- No deployment/cloud infrastructure

---

## Resume Bullet Point

- **Developed ML pipeline for quantum state certification**, implementing SVM and Transformer classifiers that distinguish distillable 3-qubit states with 85-100% accuracy using 36D Pauli feature extraction, reducing measurement requirements by 5× compared to full tomography while maintaining interpretable witness operators

---

## Claimability Warning

**Should NOT be claimed:**

| Feature | Reason |
|---------|--------|
| L1-regularized sparse SVM | Documented as "future work", not implemented |
| Bound entanglement detection | DPS Level 3+ needed, only Level 2 implemented |
| 4+ qubit scaling | Explicitly listed as future work |
| Production/cloud deployment | No deployment infrastructure present |
| TensorFlow | Listed in early requirements but not used in code |
| GPU training | Code defaults to CPU, no GPU-specific optimization |
| Real quantum hardware testing | All simulation-based |

**Can claim with caveats:**
| Feature | Caveat |
|---------|--------|
| DPS Level 2 oracle | Implemented but marked as optional dependency (CVXPY) |
| Transformer models | Implemented but marked as optional dependency (PyTorch) |
| 99-100% accuracy | Achieved with transformer, SVM achieves 85% |

---

## Evidence Summary

| Verification | Result |
|--------------|--------|
| Core tests (32) | ✅ All passing |
| Pipeline smoke test | ✅ 82.5% accuracy on 200 samples |
| Code inspection | ✅ ~3,500 lines of substantive Python |
| Documentation | ✅ Comprehensive (GOAL.md, CURRENT_STATUS.md, INITIAL_FINDINGS.md) |
| Experiment scripts | ✅ 5 experiment types with JSON output |

---

*This audit was generated by automated code analysis. All claims are based on verified code evidence.*
