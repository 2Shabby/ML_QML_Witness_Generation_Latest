# 3-Qubit Distillability Witness Learning

**Learning Restricted Witnesses for Three-Qubit Distillability Using 1+2 Body Pauli Measurements**

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
├── CURRENT_STATUS.md           # CANONICAL codebase status
├── RESTRUCTURE_PLAN.md         # Debloating plan (reference)
│
├── src/
│   ├── quantum_states/
│   │   └── state_generation.py # State generators, partial transpose
│   ├── feature_extraction/
│   │   └── pauli_features.py   # 36D restricted feature extraction
│   ├── ml_models/
│   │   └── svm_witness.py      # Linear SVM witness learner
│   └── utils/
│       └── __init__.py         # Minimal utilities (set_seed)
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
from src.quantum_states.state_generation import generate_entangled_state
from src.feature_extraction.pauli_features import create_sparse_measurement_set
from src.ml_models import SVMWitnessLearner

ghz = generate_entangled_state(3, 'ghz', noise_level=0.1)
basis = create_sparse_measurement_set(3, 'two_body')
print(f'GHZ dimension: {ghz.dim}')       # Should be 8
print(f'Restricted basis: {len(basis)}') # Should be 36
"
```

## Quick Start

```python
from src.quantum_states.state_generation import generate_entangled_state
from src.feature_extraction.pauli_features import (
    create_sparse_measurement_set,
    extract_pauli_features
)
from src.ml_models import SVMWitnessLearner

# Generate 3-qubit GHZ state with noise
ghz = generate_entangled_state(3, 'ghz', noise_level=0.1)

# Create 36D restricted basis (1+2 body only)
restricted_basis = create_sparse_measurement_set(3, strategy='two_body')

# Extract features
features = extract_pauli_features(ghz, restricted_basis)
print(f"Feature vector: {len(features)} dimensions")  # 36
```

## Current Status

| Component | Status |
|-----------|--------|
| 36D feature extraction | ✅ Ready |
| Linear SVM witness learner | ✅ Ready |
| Witness operator extraction | ✅ Ready |
| NPT distillability oracle | ❌ **Needed** |
| 3-qubit dataset generation | ❌ **Needed** |
| Cluster state generator | ❌ **Needed** |

See [CURRENT_STATUS.md](CURRENT_STATUS.md) for detailed module status.

## Dependencies

**Required:**
```
qiskit>=1.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
scipy>=1.11.0
pytest>=7.4.0
```

## Running Tests

```bash
pytest tests/ -v
```

## Documents

| Document | Purpose |
|----------|---------|
| [GOAL.md](GOAL.md) | **CANONICAL** research objective |
| [CURRENT_STATUS.md](CURRENT_STATUS.md) | **CANONICAL** codebase status |
| [RESTRUCTURE_PLAN.md](RESTRUCTURE_PLAN.md) | Debloating plan and implementation skeleton |
| [NEXT_SESSION_PROMPT.md](NEXT_SESSION_PROMPT.md) | Prompt for continuing development |

## Next Steps

1. **Implement NPT Oracle** - Check NPT across all 3 bipartitions
2. **Add State Generators** - Cluster states, 3-qubit product states
3. **Create Distillability Dataset** - Proper labeling for training
4. **Validate Pipeline** - End-to-end 3-qubit test

## License

MIT License
