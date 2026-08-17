# Current Implementation Status

**Status:** Live code inventory

**Last audited:** 2026-08-17

**Research direction:** [three_qubit_distillability_research_context.md](three_qubit_distillability_research_context.md)

**Results provenance:** [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)

This document describes what exists in the current checkout. It intentionally separates implementation presence from experimental validation.

## Summary

The repository implements the original three-qubit classical pipeline and several nonlinear extensions. A local ROCm/PennyLane environment and a differentiable amplitude-embedding feasibility check are now verified. The next research phase described in the consolidated context—a reusable, controlled amplitude-encoded QML classifier and experiment—is not yet implemented.

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
| Local Python 3.12 / ROCm research environment | Installed and GPU-verified |
| PennyLane amplitude-embedding feasibility spike | Verified on ROCm; not repository code |
| PennyLane amplitude-encoded classifier | Not implemented |
| L2-normalized classical control | Not implemented as a documented experiment |
| Frozen result artifacts | Absent; `results/` contains only `.gitkeep` |

## Checkout verification

The following statements were checked directly on 2026-08-17:

- 34 Python files are present under the repository.
- Seven test modules define 112 focused test functions.
- `python3 -m compileall -q src scripts tests` succeeds.
- The ignored local environment `env/rocm` provides Python 3.12, PyTorch 2.12.1 + ROCm 7.2, PennyLane 0.45.1, CVXPY 1.9.2, and the declared project dependencies.
- PyTorch detects the Radeon RX 7800 XT (`gfx1101`) through ROCm and successfully executes GPU tensor operations.
- `env/rocm/bin/python -m pytest -q` reports 112 passed and five warnings.
- `MLPDiscriminator` uses standard batch-normalization behavior for ordinary batches and running statistics for a singleton training batch, preventing the former `BatchNorm1d` exception while retaining gradient flow.
- A standalone smoke check successfully zero-padded and normalized 36 values with PennyLane `AmplitudeEmbedding` on six qubits, executed through the PyTorch interface on the ROCm device, and produced finite gradients. This verifies technical feasibility only; it is not a trained classifier or research result.
- There are no committed experiment JSON files from which the manuscript metrics can be regenerated or independently inspected.

Test counts by module:

| Module | Test functions |
|---|---:|
| `test_dps_oracle.py` | 21 |
| `test_feature_extraction.py` | 7 |
| `test_integration.py` | 4 |
| `test_mlp_classifier.py` | 22 |
| `test_state_generation.py` | 21 |
| `test_transformer_witness.py` | 18 |
| `test_variational_povm.py` | 19 |
| **Total** | **112** |

The current checkout result is 112/112 tests passing. Six artifact tests were removed because they duplicated stronger coverage or asserted only that an oracle returned a Boolean. Integration-test logging, ignored return values, direct-execution boilerplate, and unused imports were also removed.

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
4. CVXPY remains outside the general `requirements.txt`, but it is included in the machine-specific `requirements-rocm.lock` used by the audited environment.
5. The near-perfect nonlinear results have not yet been documented with leakage and family-held-out controls.
6. PennyLane is installed and locked for this machine and a smoke check passed, but no reusable quantum circuit, classifier, training pipeline, test, or normalization-control experiment exists in the repository.

## Current research priorities

1. Reproduce the current classical manuscript table and commit result artifacts.
2. Freeze seeds and train/test indices so every model uses identical splits.
3. Run raw, L2-normalized, and L2-normalized-plus-norm classical controls.
4. Stress-test MLP and transformer performance under family-held-out and boundary-focused evaluation.
5. Repeat the 36D-vs-63D comparison for nonlinear models.
6. Implement the proposed six-qubit amplitude-encoded classifier separately from direct-state input experiments.

## Status discipline

- “Implemented” means code is present; it does not mean the latest checkout was executed successfully.
- “Reported” means a manuscript or prior project record states the result.
- “Verified” requires a recorded command and artifact from the identified checkout.
- New metrics belong in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md), not in this inventory.
