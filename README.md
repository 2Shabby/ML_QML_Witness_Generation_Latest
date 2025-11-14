# ML/QML Quantum Witness Generation Framework

A unified framework for machine learning-based quantum resource witnessing and classification in computationally intractable regimes.

This implementation follows the approach described in `ML_QML_Quantum_Witness_Framework.md`.

## Overview

This framework implements:

- **Quantum State Generation**: Generate separable, entangled, and mixed quantum states
- **Feature Extraction**: Extract Pauli/Bloch vector representations from density matrices
- **ML Witness Learning**: Train SVM-based witnesses for entanglement detection
- **Witness Extraction**: Convert trained classifiers into measurable witness operators
- **Measurement Optimization**: Compute minimal measurement sets for experimental implementation

## Project Structure

```
ML_QML_Witness_Generation/
├── src/
│   ├── quantum_states/        # Quantum state generation
│   ├── feature_extraction/    # Pauli feature extraction
│   ├── ml_models/             # SVM and other ML models
│   ├── witnesses/             # Witness operators and utilities
│   └── utils/                 # Helper functions
├── tests/                     # Unit and integration tests
├── notebooks/                 # Jupyter notebooks for experiments
├── config/                    # Configuration files
└── requirements.txt           # Python dependencies
```

## Installation

### Step 1: Create a Python Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate  # On Windows
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: This will install:
- Qiskit (quantum computing framework)
- TensorFlow (deep learning)
- scikit-learn (classical ML)
- NumPy, SciPy (numerical computing)
- pytest (testing)

### Step 3: Verify Installation

Run the test suite to ensure everything is installed correctly:

```bash
pytest tests/ -v
```

## Quick Start

### Example 1: Basic SVM Witness Learning

```python
import sys
sys.path.insert(0, '/home/user/ML_QML_Witness_Generation')

from src.quantum_states.state_generation import generate_dataset
from src.feature_extraction.pauli_features import get_pauli_basis, extract_features_batch
from src.ml_models.svm_witness import SVMWitnessLearner

# Generate dataset
states, labels = generate_dataset(
    n_qubits=2,
    n_samples=200,
    entangled_fraction=0.5,
    seed=42
)

# Extract features
pauli_basis = get_pauli_basis(n_qubits=2, include_identity=False)
features = extract_features_batch(states, pauli_basis)

# Train SVM witness
svm_learner = SVMWitnessLearner(pauli_basis=pauli_basis, C=1.0)
metrics = svm_learner.train(features, labels)

# Extract witness operator
witness = svm_learner.get_witness_operator()
print(f"Witness operator: {witness}")
print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
```

See `examples/basic_svm_witness.py` for a complete example.

### Example 2: Sparse Witness for Minimal Measurements

```python
# After training (as above)...

# Get sparse witness with threshold
sparse_witness = svm_learner.get_sparse_witness(threshold=0.1)
measurement_cost = svm_learner.get_measurement_cost()

print(f"Sparse witness terms: {len(sparse_witness)}")
print(f"Measurement settings required: {measurement_cost}")
```

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Modules

```bash
# Test state generation
pytest tests/test_state_generation.py -v

# Test feature extraction
pytest tests/test_feature_extraction.py -v

# Test integration (end-to-end)
pytest tests/test_integration.py -v -s
```

### Run Tests with Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

## Testing Commands (For User)

After installation, run the following commands to test the implementation:

### 1. Basic Unit Tests

```bash
# Test quantum state generation
pytest tests/test_state_generation.py::TestStateGeneration::test_bell_states -v

# Test feature extraction
pytest tests/test_feature_extraction.py::TestFeatureExtraction::test_pauli_basis_generation -v
```

### 2. Integration Tests

```bash
# Full end-to-end SVM witness pipeline
pytest tests/test_integration.py::TestIntegration::test_end_to_end_svm_witness -v -s

# Test with incomplete measurements
pytest tests/test_integration.py::TestIntegration::test_incomplete_measurements_pipeline -v -s

# Test on known quantum states
pytest tests/test_integration.py::TestIntegration::test_witness_on_known_states -v -s
```

### 3. Run All Tests

```bash
pytest tests/ -v -s
```

### 4. Run Example Script

```bash
python examples/basic_svm_witness.py
```

## Expected Test Results

When running the integration tests, you should see:

- **State Generation**: 200 states generated (100 separable, 100 entangled)
- **Feature Extraction**: Feature matrix of shape (200, 15) for 2-qubit systems
- **SVM Training**: Test accuracy > 70-80%
- **Witness Extraction**: Witness operator with 10-15 non-zero Pauli terms
- **Measurement Cost**: 5-10 measurement settings required

## Key Modules

### 1. Quantum State Generation (`src/quantum_states/`)

- `generate_separable_state()`: Generate separable (product) states
- `generate_entangled_state()`: Generate entangled states (Bell, GHZ, W, random)
- `generate_dataset()`: Generate labeled dataset for training
- `check_ppt_criterion()`: Check PPT criterion for 2x2 and 2x3 systems

### 2. Feature Extraction (`src/feature_extraction/`)

- `get_pauli_basis()`: Generate complete Pauli basis for n qubits
- `extract_pauli_features()`: Extract Bloch vector representation
- `create_sparse_measurement_set()`: Create incomplete measurement sets
- `group_commuting_paulis()`: Group operators for co-measurement

### 3. ML Models (`src/ml_models/`)

- `SVMWitnessLearner`: Linear SVM for witness learning
  - `train()`: Train on labeled data
  - `get_witness_operator()`: Extract witness as SparsePauliOp
  - `get_sparse_witness()`: Get thresholded sparse witness
  - `get_measurement_cost()`: Estimate experimental cost

## Framework Approach

This implementation follows **Section 10** of the framework document:

1. **Qiskit** for quantum operations (states, operators, measurements)
2. **scikit-learn** for SVM training
3. **TensorFlow** (future) for deep learning models (MLP, KAN)

### Current Implementation: Phase 1

- ✅ Quantum state generation (separable, entangled, mixed)
- ✅ Pauli basis feature extraction
- ✅ Linear SVM witness learning
- ✅ Witness operator extraction
- ✅ Measurement optimization
- ✅ Unit and integration tests

### Future Phases

- **Phase 2**: MLP/ANN for nonlinear witnesses (Failure Mode 4)
- **Phase 3**: KAN for interpretable witness discovery (Failure Mode 2)
- **Phase 4**: Hybrid ML+SDP for provable witnesses
- **Phase 5**: QML with VQC and QSVC

## Use Cases

### Use-Case 1: Entanglement Detection (2-qubit systems)

For 2-qubit systems, PPT is necessary and sufficient. The SVM learns a tighter witness than analytic witnesses.

```bash
pytest tests/test_integration.py::TestIntegration::test_end_to_end_svm_witness -v -s
```

### Use-Case 2: Incomplete Measurements (Failure Mode 4)

Detect entanglement from sparse measurement sets (e.g., only local observables).

```bash
pytest tests/test_integration.py::TestIntegration::test_incomplete_measurements_pipeline -v -s
```

## Troubleshooting

### Issue: Import errors

**Solution**: Ensure the virtual environment is activated and all dependencies are installed:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Qiskit version conflicts

**Solution**: Update Qiskit to the latest version:

```bash
pip install --upgrade qiskit qiskit-aer qiskit-machine-learning
```

### Issue: Tests fail with numerical errors

**Solution**: Some tests use random state generation. Try running with a different seed or increasing tolerance thresholds.

## References

- Framework Document: `ML_QML_Quantum_Witness_Framework.md`
- Qiskit Documentation: https://qiskit.org/documentation/
- scikit-learn Documentation: https://scikit-learn.org/

## License

MIT License

## Contributing

Contributions are welcome! Please follow the existing code structure and ensure all tests pass before submitting.

## Contact

For questions or issues, please open an issue on GitHub.
