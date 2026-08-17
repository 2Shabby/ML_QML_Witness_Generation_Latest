# Current Implementation Status

**Status:** Live code inventory

**Last audited:** 2026-08-17

**Research direction:** [three_qubit_distillability_research_context.md](three_qubit_distillability_research_context.md)

**Results provenance:** [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)

This document describes what exists in the current checkout. It intentionally separates implementation presence from experimental validation.

## Summary

The repository implements the original three-qubit classical pipeline and several nonlinear extensions. The next research phase described in the consolidated context—controlled amplitude-encoded QML classification—is not yet implemented.

| Area | Current state |
|---|---|
| Three-qubit state generation | Implemented |
| NPT labeling across three bipartitions | Implemented |
| Restricted 36D Pauli features | Implemented |
| Linear SVM witness | Implemented |
| MLP classifier | Implemented |
| Transformer classifiers | Implemented |
| Variational density-matrix POVM | Implemented in PyTorch |
| Noise and supplementary classifier scripts | Implemented |
| DPS separability helper | Implemented, with important limitations below |
| PennyLane amplitude-encoded classifier | Not implemented |
| L2-normalized classical control | Not implemented as a documented experiment |
| Frozen result artifacts | Absent; `results/` contains only `.gitkeep` |

## Checkout verification

The following statements were checked directly on 2026-08-17:

- 34 Python files are present under the repository.
- Seven test modules define 118 test functions.
- `python3 -m compileall -q src scripts tests` succeeds.
- The working environment does not provide SciPy, scikit-learn, Qiskit, PyTorch, CVXPY, pytest, or PennyLane.
- Consequently, no unit test or experiment was executed during this audit.
- There are no committed experiment JSON files from which the manuscript metrics can be regenerated or independently inspected.

Test counts by module:

| Module | Test functions |
|---|---:|
| `test_dps_oracle.py` | 24 |
| `test_feature_extraction.py` | 7 |
| `test_integration.py` | 5 |
| `test_mlp_classifier.py` | 24 |
| `test_state_generation.py` | 21 |
| `test_transformer_witness.py` | 18 |
| `test_variational_povm.py` | 19 |
| **Total** | **118** |

The former statement “115/118 tests pass” is retained only as a legacy report in Git history; it was not revalidated here.

## Implemented modules

### Configuration

`src/config.py` defines defaults for:

- Dataset size, noise range, seeds, and cross-validation.
- Linear SVM parameters.
- Transformer architecture and training.
- MLP architecture and training.
- Variational POVM architecture and training.
- Feature selection, witness sparsification, logging, and paths.

### State generation and labels

`src/quantum_states/state_generation.py` provides:

- Random density matrices.
- GHZ, W, Bell, Werner, and cluster states.
- Three-qubit product states.
- Depolarized state families and the five-family dataset generator.
- Partial transpose and NPT checks over `A|BC`, `B|AC`, and `C|AB`.

The current manuscript's answer key is the NPT rule: a state is labeled positive when at least one bipartition has a negative partial-transpose eigenvalue.

`src/quantum_states/distillability_oracles.py` provides `NPTOracle`, `PPTOracle`, and `DPSOracle`. Important qualification: `DPSOracle.is_distillable()` currently returns the NPT result and classifies every all-PPT state as non-distillable. The Level-2 symmetric-extension routine is exposed through `check_separability()`; it is not used to create a more refined binary distillability label. The implementation also documents a real-matrix simplification in its SDP construction. It should therefore not be described as rigorous general ground truth without further validation.

### Restricted measurements

`src/feature_extraction/pauli_features.py` provides full Pauli bases, feature extraction, the 36D one- and two-body subset, commuting grouping, and measurement-cost estimation.

### Models

`src/ml_models/svm_witness.py` implements a linear SVM and conversion of its fixed coefficient vector into a `SparsePauliOp`.

`src/ml_models/mlp_classifier.py` implements the `36 -> 128 -> 64 -> 32 -> 2` PyTorch classifier with batch normalization, LeakyReLU, and dropout.

`src/ml_models/transformer_witness.py` implements:

- A standard nonlinear transformer classifier.
- A hybrid model that produces state-dependent weights.
- Training, inference, persistence, and analysis helpers.

The hybrid output is state-adaptive and should not be conflated with one fixed linear witness operator.

`src/ml_models/variational_povm.py` implements a PyTorch parameterized unitary and learned two-outcome measurement operating on density matrices. This is a simulated variational POVM, not a PennyLane circuit and not an amplitude-encoding implementation.

### Experiment scripts

| Script | Scope |
|---|---|
| `run_experiments.py` | SVM ablation, cross-validation, family, noise, and witness studies |
| `run_mlp_experiments.py` | MLP baseline, family, and multi-seed studies |
| `run_transformer_experiments.py` | Transformer comparison, scaling, family, ablation, and witness analysis |
| `run_povm_experiments.py` | Variational POVM baseline, comparison, depth, and multi-seed studies |
| `run_noise_experiments.py` | Depolarizing, dephasing, and amplitude-damping sweeps |
| `run_supplementary_classifiers.py` | Random-forest and gradient-boosting controls |
| `run_comparative_analysis.py` | Combined model evaluation and plots |
| `investigate_negative_results.py` | Adversarial noise and boundary investigations |
| `plot_results.py` | Result visualization |

## Known documentation and reproducibility gaps

1. Reported metrics exist in the manuscript/context but not as committed machine-readable artifacts.
2. Several historical values differ: SVM accuracy appears as 85.3%, 85.6%, and 86.3% for different analyses or summaries.
3. Dataset splits are generated by scripts but are not frozen as reusable split artifacts.
4. Environment and dependency lock files are absent.
5. CVXPY is commented out in `requirements.txt`, although SDP tests require it.
6. The near-perfect nonlinear results have not yet been documented with leakage and family-held-out controls.
7. No PennyLane dependency, quantum circuit, amplitude embedding, or normalization-control pipeline exists.

## Current research priorities

1. Install and record a reproducible environment, then run all 118 tests.
2. Reproduce the current classical manuscript table and commit result artifacts.
3. Freeze seeds and train/test indices so every model uses identical splits.
4. Run raw, L2-normalized, and L2-normalized-plus-norm classical controls.
5. Stress-test MLP and transformer performance under family-held-out and boundary-focused evaluation.
6. Repeat the 36D-vs-63D comparison for nonlinear models.
7. Implement the proposed six-qubit amplitude-encoded classifier separately from direct-state input experiments.

## Status discipline

- “Implemented” means code is present; it does not mean the latest checkout was executed successfully.
- “Reported” means a manuscript or prior project record states the result.
- “Verified” requires a recorded command and artifact from the identified checkout.
- New metrics belong in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md), not in this inventory.
