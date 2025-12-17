# RESTRUCTURE PLAN: Debloating for 3-Qubit Distillability Witnesses

**Document Status:** ✅ COMPLETED
**Version:** 2.0
**Date:** December 17, 2025
**Aligned With:** GOAL.md v1.0

---

## Executive Summary

This document describes the **completed** restructuring of the codebase to focus exclusively on the GOAL.md objective: **learning restricted witnesses for 3-qubit distillability using 36D (1+2 body Pauli) features**.

### Debloating Results (COMPLETED)

| Metric | Before | After | Reduction |
|--------|--------|-------|-----------|
| Source files | 9 | 5 | 44% |
| Lines of code | ~1,700 | ~950 | 44% |
| Config files | 6 | 0 | 100% |
| Dependencies needed | 16 | 8 | 50% |

### Completed Actions

- ✅ Deleted MLP witness learner (364 lines)
- ✅ Deleted over-engineered utilities (531 lines)
- ✅ Deleted all config YAML files (6 files)
- ✅ Deleted associated tests (test_mlp_witness.py, test_utils.py)
- ✅ Updated __init__.py files
- ✅ Updated CURRENT_STATUS.md to v2.0
- ✅ Updated README.md for focused goal

---

## Phase 1: Deep Audit Results

### 1.1 Code Classification

#### ESSENTIAL (Keep + Refine)

| File | Lines | Functions | GOAL Alignment |
|------|-------|-----------|----------------|
| `pauli_features.py` | 273 | 6 | **CRITICAL** - 36D feature extraction |
| `svm_witness.py` | 327 | 12 | **CRITICAL** - Witness learning core |
| `state_generation.py` | 335 | 9 (keep 6) | Partial - needs NPT oracle |

**Essential Functions:**

```
pauli_features.py (KEEP ALL):
  ├── get_pauli_basis()           # Full 63D basis
  ├── extract_pauli_features()    # Single state extraction
  ├── extract_features_batch()    # Batch processing
  ├── create_sparse_measurement_set()  # ★ CRITICAL: 36D restricted basis
  ├── group_commuting_paulis()    # Measurement optimization
  └── estimate_measurement_cost() # Experimental feasibility

svm_witness.py (KEEP ALL):
  └── SVMWitnessLearner
      ├── train()                 # Learning
      ├── predict()               # Inference
      ├── get_witness_operator()  # ★ CRITICAL: W = Σ wₖPₖ
      ├── get_sparse_witness()    # Measurement-efficient
      └── get_measurement_cost()  # M settings count

state_generation.py (KEEP 6 of 9):
  ├── generate_random_density_matrix()  # Random mixed states
  ├── generate_entangled_state('ghz')   # GHZ family
  ├── generate_entangled_state('w')     # W family
  ├── generate_werner_state()           # Benchmark
  ├── partial_transpose()               # ★ Core primitive
  └── check_ppt_criterion()             # Part of NPT check
```

#### DEAD WEIGHT (Delete)

| File | Lines | Reason for Deletion |
|------|-------|---------------------|
| `mlp_witness.py` | 364 | GOAL specifies linear SVM; MLP optional comparison only |
| `checkpoint_manager.py` | 165 | Over-engineered; SVM doesn't need epoch checkpointing |
| `logger.py` | 103 | Over-engineered; standard logging sufficient |
| `config_manager.py` | 185 | Over-engineered hierarchical configs not needed |
| `reproducibility.py` | 78 | Sets PyTorch/TF seeds unnecessarily |

**Total Dead Weight:** 895 lines (53% of codebase)

#### MISALIGNED (Must Replace)

| Component | Current Behavior | Required Behavior |
|-----------|------------------|-------------------|
| `generate_dataset()` | Labels: entangled/separable | Labels: distillable/non-distillable |
| `generate_separable_state()` | Requires even n_qubits | Must work for n=3 |
| `check_ppt_criterion()` | Single bipartition | All 3 bipartitions for NPT |
| Config defaults | `n_qubits: 2` | `n_qubits: 3` |

### 1.2 Dependency Analysis

#### ESSENTIAL Dependencies

```
# Core quantum operations
qiskit>=1.0.0                 # DensityMatrix, Pauli, SparsePauliOp

# Machine learning
scikit-learn>=1.3.0           # SVC (linear SVM)

# Numerics
numpy>=1.24.0                 # Array operations
scipy>=1.11.0                 # Linear algebra (eigvalsh for NPT)

# Testing
pytest>=7.4.0                 # Unit tests
```

#### REMOVE Dependencies

| Dependency | Current Use | Why Remove |
|------------|-------------|------------|
| `tensorflow>=2.15.0` | MLP witness | MLP being deleted |
| `qiskit-machine-learning>=0.7.0` | VQC/QSVC | Not in GOAL scope |
| `qiskit-algorithms>=0.3.0` | VQE etc. | Not needed |
| `cvxpylayers>=0.1.6` | Differentiable SDP | Not needed |
| `pandas>=2.0.0` | Data handling | numpy sufficient |
| `seaborn>=0.12.0` | Plotting | matplotlib sufficient |
| `hydra-core>=1.3.0` | Config system | Hardcoded config sufficient |

#### OPTIONAL (Keep for Nice-to-Have)

```
# Keep for analysis/visualization
matplotlib>=3.7.0             # Plotting results
tqdm>=4.65.0                  # Progress bars
joblib>=1.3.0                 # Model serialization

# Keep for Phase 2 (DPS hierarchy)
cvxpy>=1.4.0                  # SDP for rigorous labeling
```

### 1.3 Config File Analysis

#### DELETE

| Config | Reason |
|--------|--------|
| `config/experiment/bound_entanglement_3x3.yaml` | 3×3 qutrit, not 3-qubit |
| `config/experiment/incomplete_measurements.yaml` | 2-qubit focus |
| `config/model/kan.yaml` | KAN not implemented |
| `config/model/mlp.yaml` | MLP being deleted |
| `config/model/svm.yaml` | Merge into single config |
| `config/defaults.yaml` | Over-engineered |

#### CREATE

Single hardcoded configuration in Python code or one simple YAML.

---

## Phase 2: Debloating Plan

### 2.1 Proposed New Structure

```
ML_QML_Witness_Generation/
├── GOAL.md                          # CANONICAL (unchanged)
├── CURRENT_STATUS.md                # Update after restructure
├── RESTRUCTURE_PLAN.md              # This document
├── requirements.txt                 # Debloated (8 deps)
├── config.py                        # Single config file (NEW)
│
├── src/
│   ├── __init__.py
│   ├── features.py                  # Merged from pauli_features.py
│   ├── states.py                    # Merged from state_generation.py + NPT oracle
│   ├── witness.py                   # Renamed from svm_witness.py
│   └── utils.py                     # Minimal: set_seed() only
│
├── tests/
│   ├── test_features.py             # Merged/simplified
│   ├── test_states.py               # Add NPT oracle tests
│   ├── test_witness.py              # Existing SVM tests
│   └── test_pipeline.py             # 3-qubit end-to-end
│
└── scripts/
    └── train_witness.py             # Main entry point (NEW)
```

### 2.2 Deletion List with Justification

| File to Delete | Lines | Justification |
|----------------|-------|---------------|
| `src/ml_models/mlp_witness.py` | 364 | GOAL specifies linear SVM; nonlinear comparison is P2 |
| `src/utils/checkpoint_manager.py` | 165 | SVM trains in seconds; no checkpointing needed |
| `src/utils/logger.py` | 103 | `logging.basicConfig()` is sufficient |
| `src/utils/config_manager.py` | 185 | Replace with hardcoded config |
| `src/utils/reproducibility.py` | 78 | `np.random.seed()` + `random_state` sufficient |
| `tests/test_mlp_witness.py` | ~100 | Testing deleted code |
| `tests/test_utils.py` | ~80 | Testing deleted utils |
| `config/experiment/*.yaml` | ~120 | All experiment configs |
| `config/model/*.yaml` | ~150 | All model configs |
| `config/defaults.yaml` | 85 | Over-engineered |

**Total Lines Deleted:** ~1,430 lines

### 2.3 Merge/Simplify Plan

#### `src/features.py` (from `pauli_features.py`)

```python
# KEEP AS-IS (all 273 lines are goal-aligned)
# Just rename file and update imports

from features import (
    get_pauli_basis,           # Full 63D
    create_sparse_measurement_set,  # 36D restricted
    extract_pauli_features,
    extract_features_batch,
    group_commuting_paulis,
    estimate_measurement_cost
)
```

#### `src/states.py` (from `state_generation.py` + NEW)

```python
# KEEP from state_generation.py:
- generate_random_density_matrix(n_qubits, ...)
- generate_entangled_state(n_qubits, type='ghz'|'w', noise_level)
- generate_werner_state(n_qubits, p)
- partial_transpose(rho, dims, axis)

# DELETE from state_generation.py:
- generate_separable_state()  # Broken for n=3
- generate_bell_state()       # 2-qubit only
- generate_dataset()          # Wrong labeling

# NEW FUNCTIONS TO ADD:
- check_npt_any_bipartition(rho) -> bool  # ★ Critical NPT oracle
- generate_noisy_cluster_state(n_qubits, noise) -> DensityMatrix
- generate_distillability_dataset(n_samples, ...) -> (states, labels)
- generate_3qubit_separable_state() -> DensityMatrix
```

#### `src/witness.py` (from `svm_witness.py`)

```python
# KEEP ALL - already goal-aligned
# Just rename file

class SVMWitnessLearner:
    # All methods preserved
```

#### `src/utils.py` (minimal)

```python
# 10-15 lines total
import numpy as np
import random

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
```

#### `config.py` (NEW - replaces all YAML)

```python
"""Hardcoded configuration for 3-qubit distillability witness learning."""

CONFIG = {
    # System
    'n_qubits': 3,
    'feature_strategy': 'two_body',  # 36D restricted

    # Dataset
    'n_train_samples': 4000,
    'n_test_samples': 500,
    'noise_range': (0.0, 0.5),

    # SVM
    'svm_C': 1.0,
    'svm_kernel': 'linear',

    # Reproducibility
    'seed': 42,

    # Output
    'results_dir': 'results/3qubit_distillability'
}
```

### 2.4 Proposed `requirements.txt`

```
# Core (REQUIRED)
qiskit>=1.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
scipy>=1.11.0

# Testing
pytest>=7.4.0

# Nice-to-have (OPTIONAL)
matplotlib>=3.7.0
tqdm>=4.65.0
joblib>=1.3.0

# Future (P1 - DPS hierarchy)
# cvxpy>=1.4.0
```

---

## Phase 3: Implementation Skeleton

### 3.1 NPT Distillability Oracle (CRITICAL GAP)

```python
# src/states.py - NEW FUNCTION

def check_npt_any_bipartition(rho: DensityMatrix) -> bool:
    """
    Check if state is NPT (negative partial transpose) across ANY bipartition.

    For 3 qubits, checks bipartitions:
      - A|BC (qubit 0 vs qubits 1,2): dims=[2,4]
      - B|AC (qubit 1 vs qubits 0,2): dims=[2,4], requires permutation
      - C|AB (qubit 2 vs qubits 0,1): dims=[4,2]

    Returns:
        True if NPT across ANY bipartition (proxy for distillable)
        False if PPT across ALL bipartitions (may be bound entangled or separable)
    """
    rho_data = rho.data
    n = 8  # 2^3

    # Bipartition A|BC: partial transpose on A (dims=[2,4])
    rho_pt_A = partial_transpose(rho_data, dims=[2, 4], axis=0)
    if np.min(np.linalg.eigvalsh(rho_pt_A)) < -1e-10:
        return True

    # Bipartition B|AC: need to permute qubits, then PT
    # Permute: (0,1,2) -> (1,0,2), then PT on first subsystem
    rho_permuted = _permute_qubits(rho_data, [1, 0, 2])
    rho_pt_B = partial_transpose(rho_permuted, dims=[2, 4], axis=0)
    if np.min(np.linalg.eigvalsh(rho_pt_B)) < -1e-10:
        return True

    # Bipartition C|AB: partial transpose on C (dims=[4,2])
    rho_pt_C = partial_transpose(rho_data, dims=[4, 2], axis=1)
    if np.min(np.linalg.eigvalsh(rho_pt_C)) < -1e-10:
        return True

    return False  # PPT across all bipartitions


def _permute_qubits(rho: np.ndarray, perm: list) -> np.ndarray:
    """Permute qubit ordering in density matrix."""
    n_qubits = len(perm)
    # Reshape to tensor, permute, reshape back
    shape = [2] * (2 * n_qubits)
    rho_tensor = rho.reshape(shape)

    # Permute both row and column indices
    perm_full = perm + [p + n_qubits for p in perm]
    rho_permuted = np.transpose(rho_tensor, perm_full)

    return rho_permuted.reshape(2**n_qubits, 2**n_qubits)
```

### 3.2 3-Qubit State Generators (CRITICAL GAP)

```python
# src/states.py - NEW FUNCTIONS

def generate_noisy_cluster_state(
    n_qubits: int = 3,
    noise_level: float = 0.0,
    seed: Optional[int] = None
) -> DensityMatrix:
    """
    Generate 3-qubit linear cluster state with depolarizing noise.

    |cluster⟩ = CZ_{01} CZ_{12} |+⟩^⊗3

    Cluster states are resources for measurement-based quantum computing.
    """
    if seed is not None:
        np.random.seed(seed)

    # |+⟩^⊗3
    plus = np.array([1, 1]) / np.sqrt(2)
    psi = np.kron(np.kron(plus, plus), plus)  # 8-dimensional

    # CZ gates (controlled-Z)
    CZ = np.diag([1, 1, 1, -1])  # 2-qubit CZ
    I = np.eye(2)

    # CZ_{01} ⊗ I_2
    CZ_01 = np.kron(CZ, I)
    # I_0 ⊗ CZ_{12}
    CZ_12 = np.kron(I, CZ)

    # Apply CZ gates
    psi = CZ_01 @ psi
    psi = CZ_12 @ psi

    # Create density matrix
    rho = np.outer(psi, psi.conj())

    # Add depolarizing noise
    if noise_level > 0:
        identity = np.eye(8) / 8
        rho = (1 - noise_level) * rho + noise_level * identity

    return DensityMatrix(rho)


def generate_3qubit_product_state(seed: Optional[int] = None) -> DensityMatrix:
    """
    Generate a random 3-qubit product (separable) state.

    ρ = ρ_A ⊗ ρ_B ⊗ ρ_C where each is a random single-qubit state.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate 3 random single-qubit pure states
    rho = None
    for i in range(3):
        # Random point on Bloch sphere
        theta = np.arccos(2 * np.random.rand() - 1)
        phi = 2 * np.pi * np.random.rand()

        psi_i = np.array([
            np.cos(theta/2),
            np.exp(1j * phi) * np.sin(theta/2)
        ])
        rho_i = np.outer(psi_i, psi_i.conj())

        if rho is None:
            rho = rho_i
        else:
            rho = np.kron(rho, rho_i)

    return DensityMatrix(rho)
```

### 3.3 Distillability Dataset Generator (CRITICAL GAP)

```python
# src/states.py - NEW FUNCTION

def generate_distillability_dataset(
    n_samples: int = 5000,
    noise_range: Tuple[float, float] = (0.0, 0.5),
    seed: Optional[int] = None
) -> Tuple[List[DensityMatrix], np.ndarray]:
    """
    Generate labeled dataset for 3-qubit distillability learning.

    Labels:
        1 = Distillable (NPT across at least one bipartition)
        0 = Non-distillable (PPT across all bipartitions)

    State families:
        - Noisy GHZ states (varying noise)
        - Noisy W states (varying noise)
        - Noisy cluster states (varying noise)
        - Random mixed states
        - Product states (definitely separable)

    Returns:
        (states, labels) where labels are distillability indicators
    """
    if seed is not None:
        np.random.seed(seed)

    states = []
    labels = []

    # Distribute samples across state families
    n_per_family = n_samples // 5

    state_generators = [
        ('ghz', lambda s, n: generate_entangled_state(3, 'ghz', n, s)),
        ('w', lambda s, n: generate_entangled_state(3, 'w', n, s)),
        ('cluster', lambda s, n: generate_noisy_cluster_state(3, n, s)),
        ('random', lambda s, n: generate_random_density_matrix(3, seed=s)),
        ('product', lambda s, n: generate_3qubit_product_state(s)),
    ]

    for family_name, generator in state_generators:
        for i in range(n_per_family):
            # Sample noise level
            if family_name == 'product':
                noise = 0.0  # Product states are pure
            else:
                noise = np.random.uniform(*noise_range)

            # Generate state
            state = generator(seed + i if seed else None, noise)

            # Label using NPT oracle
            is_distillable = check_npt_any_bipartition(state)

            states.append(state)
            labels.append(1 if is_distillable else 0)

    # Shuffle
    indices = np.random.permutation(len(states))
    states = [states[i] for i in indices]
    labels = np.array([labels[i] for i in indices])

    return states, labels
```

### 3.4 Main Training Script Skeleton

```python
# scripts/train_witness.py

"""
Train 3-qubit distillability witness with restricted (36D) measurements.

Usage:
    python scripts/train_witness.py
"""

import numpy as np
from pathlib import Path

from src.utils import set_seed
from src.states import generate_distillability_dataset
from src.features import create_sparse_measurement_set, extract_features_batch
from src.witness import SVMWitnessLearner
from config import CONFIG


def main():
    # Setup
    set_seed(CONFIG['seed'])
    results_dir = Path(CONFIG['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training 3-qubit distillability witness")
    print(f"Feature space: 36D (1+2 body Paulis)")
    print("=" * 60)

    # Generate dataset with distillability labels
    print("\n[1/4] Generating training dataset...")
    train_states, train_labels = generate_distillability_dataset(
        n_samples=CONFIG['n_train_samples'],
        noise_range=CONFIG['noise_range'],
        seed=CONFIG['seed']
    )

    print(f"  Distillable: {sum(train_labels)} / {len(train_labels)}")
    print(f"  Non-distillable: {len(train_labels) - sum(train_labels)} / {len(train_labels)}")

    # Create restricted 36D feature basis
    print("\n[2/4] Extracting restricted features...")
    restricted_basis = create_sparse_measurement_set(
        n_qubits=CONFIG['n_qubits'],
        strategy=CONFIG['feature_strategy']
    )
    print(f"  Restricted basis size: {len(restricted_basis)} (vs 63 full)")

    features = extract_features_batch(train_states, restricted_basis, verbose=False)
    print(f"  Feature matrix: {features.shape}")

    # Train linear SVM
    print("\n[3/4] Training linear SVM witness...")
    learner = SVMWitnessLearner(
        pauli_basis=restricted_basis,
        C=CONFIG['svm_C'],
        kernel=CONFIG['svm_kernel'],
        random_state=CONFIG['seed']
    )

    metrics = learner.train(features, train_labels, test_size=0.2, verbose=True)

    # Extract witness operator
    print("\n[4/4] Extracting witness operator...")
    witness = learner.get_witness_operator()
    sparse_witness = learner.get_sparse_witness(threshold=0.01)
    measurement_cost = learner.get_measurement_cost()

    print(f"  Witness terms: {len(witness)}")
    print(f"  Sparse witness terms: {len(sparse_witness)}")
    print(f"  Measurement settings: {measurement_cost}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Test Accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"Test Precision: {metrics['test_precision']:.4f}")
    print(f"Test Recall:    {metrics['test_recall']:.4f}")
    print(f"Measurement Settings: {measurement_cost}")

    # Save witness
    # TODO: Save witness operator to file

    return metrics


if __name__ == '__main__':
    main()
```

---

## Phase 4: Validation Checklist

### 4.1 Pre-Restructure Validation

Before deleting anything, verify:

```bash
# Ensure all tests pass on current code
pytest tests/ -v

# Verify 3-qubit state generation works
python -c "
from src.quantum_states.state_generation import generate_entangled_state
ghz = generate_entangled_state(3, 'ghz', 0.0)
print(f'GHZ dimension: {ghz.dim}')  # Should be 8
"

# Verify 36D feature extraction works
python -c "
from src.feature_extraction.pauli_features import create_sparse_measurement_set
basis = create_sparse_measurement_set(3, 'two_body')
print(f'Restricted basis: {len(basis)} features')  # Should be 36
"
```

### 4.2 Post-Restructure Validation

After restructure, verify:

```bash
# All tests pass
pytest tests/ -v

# NPT oracle works correctly
python -c "
from src.states import generate_entangled_state, check_npt_any_bipartition
ghz = generate_entangled_state(3, 'ghz', 0.0)
print(f'Pure GHZ is distillable: {check_npt_any_bipartition(ghz)}')  # Should be True
"

# End-to-end pipeline works
python scripts/train_witness.py
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| NPT oracle bugs | Medium | High | Unit test against known states |
| Permutation indexing errors | High | High | Test all 3 bipartitions explicitly |
| Dataset imbalance | Medium | Medium | Monitor class distribution |
| Deleting needed code | Low | High | Git history for recovery |

---

## Implementation Order

### Day 1: Foundation
1. Create branch `restructure/debloat`
2. Delete MLP and associated tests
3. Delete over-engineered utils
4. Delete all config YAML files
5. Create minimal `config.py`
6. Run tests to verify nothing breaks

### Day 2: NPT Oracle
1. Implement `check_npt_any_bipartition()`
2. Implement `_permute_qubits()` helper
3. Unit test against known states:
   - Pure GHZ → distillable
   - Pure W → distillable
   - Product state → not distillable
   - Werner state threshold
4. Add to `src/states.py`

### Day 3: State Generators
1. Implement `generate_noisy_cluster_state()`
2. Implement `generate_3qubit_product_state()`
3. Implement `generate_distillability_dataset()`
4. Unit test each generator

### Day 4: Integration
1. Create `scripts/train_witness.py`
2. End-to-end pipeline test
3. Update tests to reflect new structure
4. Update documentation

### Day 5: Cleanup
1. Flatten directory structure if beneficial
2. Update `requirements.txt`
3. Update `CURRENT_STATUS.md`
4. Final validation

---

## Appendix: Full File Deletion List

```bash
# Delete these files
rm src/ml_models/mlp_witness.py
rm src/utils/checkpoint_manager.py
rm src/utils/logger.py
rm src/utils/config_manager.py
rm src/utils/reproducibility.py
rm tests/test_mlp_witness.py
rm tests/test_utils.py
rm config/defaults.yaml
rm config/experiment/bound_entanglement_3x3.yaml
rm config/experiment/incomplete_measurements.yaml
rm config/model/kan.yaml
rm config/model/mlp.yaml
rm config/model/svm.yaml

# Remove empty directories
rmdir config/experiment
rmdir config/model
rmdir config
```

---

## Appendix: New `requirements.txt`

```
# Core (REQUIRED for GOAL.md)
qiskit>=1.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
scipy>=1.11.0

# Testing
pytest>=7.4.0

# Optional (nice-to-have)
matplotlib>=3.7.0
tqdm>=4.65.0
joblib>=1.3.0

# Future (DPS hierarchy - uncomment when needed)
# cvxpy>=1.4.0
```

---

*This document is the authoritative restructure plan. Proceed with deletions and additions as specified. All changes should be committed incrementally with clear messages.*
