# 3-Qubit Distillability Witness Learning

**Learning Restricted Witnesses for Three-Qubit Distillability Using 1+2 Body Pauli Measurements**

## Status: MVP Complete and Audited

| Metric | Value |
|--------|-------|
| Tests | 32/32 passing |
| Test Accuracy | 87% |
| NPT Oracle | Verified correct |
| Pipeline | Production ready |

## Research Goal

This project investigates whether **distillability of 3-qubit QEC resource states** can be certified using only experimentally accessible measurements (1-body and 2-body Pauli operators), without requiring full state tomography or computationally expensive SDP methods.

See [GOAL.md](GOAL.md) for the complete research objective.

## Key Question

> Does there exist a physically measurable, low-weight Hermitian operator W, expressible as a linear combination of one- and two-body Pauli operators, whose expectation value reliably distinguishes distillable three-qubit states from non-distillable ones?

## Approach

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│ 3-Qubit States  │ ──▶ │ 36D Features     │ ──▶ │ Linear SVM      │ ──▶ │ Witness Operator │
│ (DensityMatrix) │     │ (1+2 body Pauli) │     │ Hyperplane w·x+b│     │ W = Σ wₖPₖ       │
└─────────────────┘     └──────────────────┘     └─────────────────┘     └──────────────────┘
```

**Feature Space (36D):**
- 9 single-qubit Paulis: X₁, Y₁, Z₁, X₂, Y₂, Z₂, X₃, Y₃, Z₃
- 27 two-qubit correlators: X₁X₂, X₁Y₂, ..., Z₂Z₃
- **Excluded:** 27 three-body terms (experimentally costly)

**Labeling:** NPT-based distillability oracle (separate from features)

## Project Structure

```
ML_QML_Witness_Generation/
├── GOAL.md                     # CANONICAL research objective
├── CURRENT_STATUS.md           # CANONICAL codebase status (v4.0)
├── AUDIT_REPORT.md             # Verification results
├── RESTRUCTURE_PLAN.md         # Historical reference
│
├── src/
│   ├── quantum_states/
│   │   └── state_generation.py # State generators, NPT oracle
│   ├── feature_extraction/
│   │   └── pauli_features.py   # 36D restricted feature extraction
│   ├── ml_models/
│   │   └── svm_witness.py      # Linear SVM witness learner
│   └── utils/
│       └── __init__.py         # Minimal utilities
│
├── tests/
│   ├── test_state_generation.py
│   ├── test_feature_extraction.py
│   └── test_integration.py
│
└── requirements.txt
```

## Installation

```bash
# Clone repository
git clone <repo-url>
cd ML_QML_Witness_Generation

# Install dependencies
pip install qiskit scikit-learn numpy scipy pytest

# Verify installation
python -c "
from src.quantum_states.state_generation import (
    generate_entangled_state,
    check_npt_any_bipartition
)
from src.feature_extraction.pauli_features import create_sparse_measurement_set
from src.ml_models import SVMWitnessLearner

ghz = generate_entangled_state(3, 'ghz', noise_level=0.1)
basis = create_sparse_measurement_set(3, 'two_body')
print(f'GHZ dimension: {ghz.dim}')       # Should be 8
print(f'Restricted basis: {len(basis)}') # Should be 36
print(f'GHZ distillable: {check_npt_any_bipartition(ghz)}')  # Should be True
"
```

## Quick Start

```python
from src.quantum_states.state_generation import (
    generate_entangled_state,
    generate_distillability_dataset,
    check_npt_any_bipartition
)
from src.feature_extraction.pauli_features import (
    create_sparse_measurement_set,
    extract_features_batch
)
from src.ml_models import SVMWitnessLearner

# Generate 3-qubit distillability dataset
states, labels = generate_distillability_dataset(n_samples=500, seed=42)
print(f"Dataset: {len(states)} states, {sum(labels)} distillable")

# Create 36D restricted basis (1+2 body only)
basis = create_sparse_measurement_set(3, strategy='two_body')

# Extract features
features = extract_features_batch(states, basis, verbose=False)
print(f"Feature matrix: {features.shape}")  # (500, 36)

# Train linear SVM witness
learner = SVMWitnessLearner(pauli_basis=basis, C=1.0, kernel='linear')
metrics = learner.train(features, labels, test_size=0.2)
print(f"Test accuracy: {metrics['test_accuracy']:.2%}")

# Extract witness operator
witness = learner.get_witness_operator()
print(f"Witness terms: {len(witness)}")
```

## Component Status

| Component | Status | Notes |
|-----------|--------|-------|
| NPT distillability oracle | ✅ Ready | All 3 bipartitions verified |
| 3-qubit state generators | ✅ Ready | GHZ, W, cluster, product |
| 36D feature extraction | ✅ Ready | 1+2 body Paulis only |
| Linear SVM witness learner | ✅ Ready | With SparsePauliOp extraction |
| Distillability dataset | ✅ Ready | Correct NPT-based labels |

See [CURRENT_STATUS.md](CURRENT_STATUS.md) for detailed module status.

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run NPT oracle tests
python -m pytest tests/test_state_generation.py::TestNPTOracleAndDistillability -v

# Run integration pipeline test
python -m pytest tests/test_integration.py::TestIntegration::test_3qubit_distillability_pipeline -v -s
```

## Documents

| Document | Purpose |
|----------|---------|
| [GOAL.md](GOAL.md) | **CANONICAL** research objective |
| [CURRENT_STATUS.md](CURRENT_STATUS.md) | **CANONICAL** codebase status |
| [AUDIT_REPORT.md](AUDIT_REPORT.md) | Verification results |
| [RESTRUCTURE_PLAN.md](RESTRUCTURE_PLAN.md) | Historical reference |

## Dependencies

**Required:**
```
qiskit>=1.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
scipy>=1.11.0
pytest>=7.4.0
```

## License

MIT License
