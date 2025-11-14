# Testing Guide

This document provides comprehensive testing commands for the ML/QML Quantum Witness Generation Framework.

## Prerequisites

Ensure you have:
1. Activated the virtual environment: `source venv/bin/activate`
2. Installed all dependencies: `pip install -r requirements.txt`

## Quick Test

Run all tests to verify the installation:

```bash
pytest tests/ -v
```

## Detailed Testing Commands

### 1. Unit Tests - Quantum State Generation

Test the quantum state generation module:

```bash
# Test all state generation functions
pytest tests/test_state_generation.py -v

# Test specific functions
pytest tests/test_state_generation.py::TestStateGeneration::test_bell_states -v
pytest tests/test_state_generation.py::TestStateGeneration::test_separable_state -v
pytest tests/test_state_generation.py::TestStateGeneration::test_werner_state -v
pytest tests/test_state_generation.py::TestStateGeneration::test_ppt_criterion -v
```

**Expected Results:**
- All Bell states should be detected as entangled (not PPT)
- Separable states should satisfy PPT criterion
- Werner states should transition from entangled to separable as mixing parameter changes

### 2. Unit Tests - Feature Extraction

Test the Pauli feature extraction:

```bash
# Test all feature extraction functions
pytest tests/test_feature_extraction.py -v

# Test specific functions
pytest tests/test_feature_extraction.py::TestFeatureExtraction::test_pauli_basis_generation -v
pytest tests/test_feature_extraction.py::TestFeatureExtraction::test_extract_pauli_features -v
pytest tests/test_feature_extraction.py::TestFeatureExtraction::test_sparse_measurement_sets -v
pytest tests/test_feature_extraction.py::TestFeatureExtraction::test_measurement_cost_estimation -v
```

**Expected Results:**
- For 2 qubits: 15 Pauli basis operators (4² - 1)
- Feature vectors should be real-valued
- Sparse measurement sets should have fewer operators than full basis

### 3. Integration Tests

#### Test 1: End-to-End SVM Witness Learning

This is the main integration test demonstrating the complete pipeline:

```bash
pytest tests/test_integration.py::TestIntegration::test_end_to_end_svm_witness -v -s
```

**What it tests:**
1. Generate 200 quantum states (100 separable, 100 entangled)
2. Extract Pauli features (15 features for 2 qubits)
3. Train linear SVM
4. Extract witness operator
5. Create sparse witness
6. Estimate measurement cost

**Expected Output:**
```
[Step 1] Generating quantum states dataset...
Generated 200 states (100 separable, 100 entangled)

[Step 2] Extracting Pauli features...
Feature matrix shape: (200, 15)

[Step 3] Training SVM witness learner...
Test Accuracy: 0.85-0.95
Test Precision: 0.80-1.00
Test Recall: 0.80-0.95

[Step 4] Extracting witness operator...
Witness operator terms: 10-15

[Step 5] Creating sparse witness...
Sparse witness terms: 5-10

[Step 6] Estimating measurement cost...
Measurement settings required: 3-8
```

#### Test 2: Incomplete Measurements Pipeline

Test witness learning from sparse measurements (Failure Mode 4):

```bash
pytest tests/test_integration.py::TestIntegration::test_incomplete_measurements_pipeline -v -s
```

**What it tests:**
- Learning from only 2-body correlations instead of full basis
- Classification with ~30 features instead of full 15 for 2 qubits
- Should still achieve >60% accuracy

**Expected Output:**
```
Sparse basis size: 21 (vs full: 15)
Test Accuracy: 0.60-0.80
```

#### Test 3: Known States Validation

Test that the learned witness correctly classifies well-known quantum states:

```bash
pytest tests/test_integration.py::TestIntegration::test_witness_on_known_states -v -s
```

**What it tests:**
- Bell states (all 4 types) should be detected as entangled
- Product states should be detected as separable

**Expected Output:**
```
Bell state predictions: [1 1 1 1] (or mostly 1s)
Separable state predictions: [0 0 0 0 0] (or mostly 0s)
```

### 4. Run Example Script

Run the complete example demonstrating the framework:

```bash
python examples/basic_svm_witness.py
```

**Expected Output:**

The script will show:
1. Dataset generation (300 states)
2. Feature extraction (15 Pauli operators)
3. SVM training metrics
4. Witness operator with coefficients
5. Sparse witness for measurement
6. Testing on Bell states

**Key Metrics to Check:**
- Test Accuracy: Should be >80%
- Test Precision: Should be high (>0.8) to avoid false positives
- All Bell states should be classified as entangled

### 5. Test with Coverage

Generate code coverage report:

```bash
pytest tests/ --cov=src --cov-report=html --cov-report=term
```

Then open `htmlcov/index.html` in a browser to see detailed coverage.

**Expected Coverage:**
- `src/quantum_states/`: >90%
- `src/feature_extraction/`: >85%
- `src/ml_models/`: >80%

## Understanding Test Results

### Success Criteria

A successful test run should show:

1. **State Generation:**
   - ✓ Separable states satisfy PPT criterion
   - ✓ Bell states violate PPT criterion
   - ✓ All density matrices are valid (Hermitian, trace=1, positive)

2. **Feature Extraction:**
   - ✓ Pauli basis has correct size (4ⁿ - 1 for n qubits)
   - ✓ Feature vectors are real-valued
   - ✓ Sparse measurements reduce feature count

3. **SVM Training:**
   - ✓ Test accuracy >70% (ideally >80%)
   - ✓ Test precision >80% (critical: few false positives)
   - ✓ Witness operator successfully extracted

4. **Integration:**
   - ✓ Complete pipeline runs without errors
   - ✓ Bell states detected as entangled
   - ✓ Product states detected as separable

### Common Issues and Solutions

**Issue 1: Low Test Accuracy (<70%)**

Possible causes:
- Dataset too noisy (increase sample size)
- Regularization too strong (decrease C parameter)
- Random seed variation

Solution:
```python
# In the code, try increasing C or sample size
svm_learner = SVMWitnessLearner(C=10.0)  # Less regularization
```

**Issue 2: Numerical Errors in PPT Test**

Possible causes:
- Floating point precision errors

Solution: Tests use tolerance of 1e-10 for eigenvalue checks. This should be sufficient.

**Issue 3: Import Errors**

Solution:
```bash
# Ensure you're in the project root and venv is activated
cd /home/user/ML_QML_Witness_Generation
source venv/bin/activate
export PYTHONPATH=/home/user/ML_QML_Witness_Generation:$PYTHONPATH
```

## Performance Benchmarks

On a typical system, expect:

- **State Generation**: ~1-2 seconds for 200 states
- **Feature Extraction**: ~2-5 seconds for 200 states
- **SVM Training**: ~1-3 seconds
- **Total Pipeline**: ~5-10 seconds

For 1000 states, multiply by ~5x.

## Next Steps After Testing

Once all tests pass, you can:

1. **Experiment with parameters**: Modify `n_qubits`, `n_samples`, regularization
2. **Try sparse measurements**: Use `create_sparse_measurement_set()` with different strategies
3. **Visualize results**: Create plots of witness violations
4. **Scale up**: Test on 3-qubit systems (64-dimensional Hilbert space)

## Continuous Testing During Development

While developing new features:

```bash
# Run tests on file save (requires pytest-watch)
pip install pytest-watch
ptw tests/
```

This will automatically re-run tests when files change.

## Test Data

All tests use fixed random seeds (`seed=42`) for reproducibility. Running the same test multiple times should give identical results.
