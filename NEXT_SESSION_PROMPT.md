# Next Session Prompt: Implementing NPT Oracle and Distillability Dataset

## Context

This repository contains an ML framework for learning quantum entanglement witnesses, now **pivoted and debloated** to focus on 3-qubit distillability.

## Canonical Documents (READ FIRST)

1. `GOAL.md` - CANONICAL research objective: Learning restricted witnesses for 3-qubit distillability using only 1+2 body Pauli measurements (36D features).

2. `CURRENT_STATUS.md` - CANONICAL codebase status aligned with GOAL.md.

3. `RESTRUCTURE_PLAN.md` - Completed debloating plan with implementation skeleton for new code.

## What Has Been Done

### Debloating (COMPLETED)
- ✅ Deleted MLP witness learner (not needed for linear SVM goal)
- ✅ Deleted over-engineered utilities (checkpoint_manager, logger, config_manager, reproducibility)
- ✅ Deleted all config YAML files
- ✅ Deleted legacy documentation (GAP_ANALYSIS, IMPLEMENTATION_ROADMAP, etc.)
- ✅ Updated README.md, CURRENT_STATUS.md, RESTRUCTURE_PLAN.md

### What's Ready
- ✅ `pauli_features.py` - 36D restricted feature extraction
- ✅ `svm_witness.py` - Linear SVM witness learner with operator extraction
- ✅ `state_generation.py` - GHZ, W, Werner state generators, partial transpose

## Your Task: Implement Missing Components

### Priority 1: NPT Distillability Oracle (CRITICAL)

Implement in `src/quantum_states/state_generation.py`:

```python
def check_npt_any_bipartition(rho: DensityMatrix) -> bool:
    """
    Check if state is NPT (negative partial transpose) across ANY bipartition.

    For 3 qubits, checks bipartitions:
      - A|BC (qubit 0 vs qubits 1,2)
      - B|AC (qubit 1 vs qubits 0,2)
      - C|AB (qubit 2 vs qubits 0,1)

    Returns:
        True if NPT across ANY bipartition (proxy for distillable)
        False if PPT across ALL bipartitions
    """
```

**Implementation hints:**
- Use existing `partial_transpose()` function
- Need helper `_permute_qubits()` for B|AC bipartition
- Check if min eigenvalue < -1e-10

### Priority 2: 3-Qubit State Generators

Add to `src/quantum_states/state_generation.py`:

```python
def generate_noisy_cluster_state(n_qubits: int = 3, noise_level: float = 0.0) -> DensityMatrix:
    """Generate 3-qubit linear cluster state with depolarizing noise."""

def generate_3qubit_product_state(seed: Optional[int] = None) -> DensityMatrix:
    """Generate random 3-qubit product (separable) state."""
```

### Priority 3: Distillability Dataset Generator

```python
def generate_distillability_dataset(
    n_samples: int = 5000,
    noise_range: Tuple[float, float] = (0.0, 0.5),
    seed: Optional[int] = None
) -> Tuple[List[DensityMatrix], np.ndarray]:
    """
    Generate labeled dataset for 3-qubit distillability.

    Labels:
        1 = Distillable (NPT across at least one bipartition)
        0 = Non-distillable (PPT across all bipartitions)
    """
```

### Priority 4: End-to-End Validation

Create or update test to verify:
```python
# 3-qubit states → 36D features → SVM → witness
def test_3qubit_distillability_pipeline():
    # Generate dataset with distillability labels
    states, labels = generate_distillability_dataset(n_samples=500)

    # Extract 36D features
    basis = create_sparse_measurement_set(3, 'two_body')
    features = extract_features_batch(states, basis)

    # Train SVM
    learner = SVMWitnessLearner(basis)
    metrics = learner.train(features, labels)

    # Extract witness
    witness = learner.get_witness_operator()
    assert len(witness) <= 36  # Only 1+2 body terms
```

## Implementation Guidelines

1. **Test-Driven**: Write unit tests for NPT oracle first
   - Pure GHZ → should be distillable (NPT)
   - Pure W → should be distillable (NPT)
   - Product state → should NOT be distillable (PPT)

2. **Validate bipartitions carefully**: The qubit permutation for B|AC is tricky

3. **Keep it simple**: No over-engineering, just the functions needed

4. **Update CURRENT_STATUS.md** after implementation to check off completed items

## Code Skeleton from RESTRUCTURE_PLAN.md

See `RESTRUCTURE_PLAN.md` Phase 3 for detailed implementation skeleton including:
- `check_npt_any_bipartition()`
- `_permute_qubits()` helper
- `generate_noisy_cluster_state()`
- `generate_3qubit_product_state()`
- `generate_distillability_dataset()`

## Success Criteria

1. NPT oracle correctly identifies:
   - Pure GHZ as distillable
   - Pure W as distillable
   - Product states as non-distillable
   - Werner state threshold behavior

2. Dataset generation produces balanced classes

3. End-to-end pipeline runs without errors

4. SVM achieves >60% accuracy (baseline, may improve with tuning)

## Files to Modify

- `src/quantum_states/state_generation.py` - Add NPT oracle and generators
- `tests/test_state_generation.py` - Add NPT oracle tests
- `tests/test_integration.py` - Add 3-qubit distillability pipeline test
- `CURRENT_STATUS.md` - Update status after implementation

## Do NOT

- Add new dependencies
- Create new files unless necessary
- Over-engineer (no config files, no complex abstractions)
- Change the feature extraction or SVM code (already working)

---

*This prompt is for continuing development of the 3-qubit distillability witness learning project.*
