# Current Implementation Status

**Status:** Live code inventory

**Last audited:** 2026-08-17

**Research direction:** [three_qubit_distillability_research_context.md](three_qubit_distillability_research_context.md)

**Results provenance:** [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)

This document describes what exists in the current checkout. It intentionally separates implementation presence from experimental validation.

## Summary

The repository implements the original three-qubit classical pipeline, several nonlinear extensions, a reusable six-qubit amplitude-encoded QML classifier with controlled classical comparisons, and a separate PennyLane direct-state classifier. Final report-scale validation and baseline reproduction remain deferred until the implementation stabilizes.

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
| PennyLane amplitude-encoded classifier | Implemented and ROCm-verified |
| Classical amplitude-encoding controls | Raw, L2-normalized, and normalized-plus-norm inputs implemented |
| Identical-split QML/MLP comparison | Implemented and smoke-verified on ROCm |
| PennyLane direct-state QML | Implemented and ROCm smoke-verified |
| Unified result artifacts | Generated under `results/manuscript/`; awaiting final review/commit |

## Active implementation plan

| Step | Status | Scope |
|---:|---|---|
| 1 | Complete | Six-qubit amplitude-encoded QML classifier for the restricted 36D input |
| 2 | Complete | Raw, L2-normalized, and L2-normalized-plus-original-norm MLP controls |
| 3 | Complete | All four models evaluated through one reusable stratified split contract |
| 4 | Complete | PennyLane circuit that accepts the three-qubit density matrix directly, kept separate from the restricted-measurement experiment |
| 5 | Deferred | Report-scale runs, baseline reproduction, and frozen result/split artifacts after the implementation stabilizes |

The existing PyTorch variational POVM and the new PennyLane direct-state classifier are separate implementations. The latter uses PennyLane to construct a trainable circuit unitary and applies it to batched density matrices on ROCm.

## Checkout verification

The following statements were checked directly on 2026-08-17:

- 30 Python files are present under the repository.
- Ten test modules define 124 focused test functions.
- `python3 -m compileall -q src scripts tests` succeeds.
- The ignored local environment `env/rocm` provides Python 3.12, PyTorch 2.12.1 + ROCm 7.2, PennyLane 0.45.1, CVXPY 1.9.2, and the declared project dependencies.
- PyTorch detects the Radeon RX 7800 XT (`gfx1101`) through ROCm and successfully executes GPU tensor operations.
- `env/rocm/bin/python -m pytest -q` reports 124 passed and five warnings.
- `MLPDiscriminator` uses standard batch-normalization behavior for ordinary batches and running statistics for a singleton training batch, preventing the former `BatchNorm1d` exception while retaining gradient flow.
- The amplitude-QML learner completed batched optimizer steps and the CLI completed state generation, 36D extraction, training, prediction, split capture, and JSON serialization on ROCm. These are implementation checks, not research results.
- The controlled-comparison CLI completed a deliberately undersized ROCm smoke run across the three MLP controls and amplitude QML using one shared split. Its metrics are not research evidence.
- The direct-state CLI completed a deliberately undersized, one-epoch ROCm smoke run from generated density matrices through training and JSON serialization. Its metrics are not research evidence.
- The unified JSON and CSV artifacts now contain exact dataset hashes, split indices, source hash, environment versions, and scalar metrics.

Test counts by module:

| Module | Test functions |
|---|---:|
| `test_amplitude_qml.py` | 5 |
| `test_controlled_comparison.py` | 3 |
| `test_direct_state_qml.py` | 3 |
| `test_dps_oracle.py` | 21 |
| `test_feature_extraction.py` | 7 |
| `test_integration.py` | 4 |
| `test_mlp_classifier.py` | 22 |
| `test_state_generation.py` | 22 |
| `test_transformer_witness.py` | 18 |
| `test_variational_povm.py` | 19 |
| **Total** | **124** |

The current checkout result is 124/124 tests passing.

## Implemented modules

### Configuration

`src/config.py` defines defaults for:

- Dataset size, noise range, seeds, and cross-validation.
- Linear SVM parameters.
- Transformer architecture and training.
- MLP architecture and training.
- Variational POVM architecture and training.
- Six-qubit amplitude-QML architecture and training.
- Three-qubit direct-state QML architecture and training.
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

`src/feature_extraction/preprocessing.py` provides the three amplitude-encoding controls: raw 36D features, row-wise L2-normalized 36D features, and normalized features with the discarded norm appended as feature 37.

### Models

`src/ml_models/svm_witness.py` implements a linear SVM and conversion of its fixed coefficient vector into a `SparsePauliOp`.

`src/ml_models/mlp_classifier.py` implements the `36 -> 128 -> 64 -> 32 -> 2` PyTorch classifier with batch normalization, LeakyReLU, and dropout.

`src/ml_models/transformer_witness.py` implements:

- A standard nonlinear transformer classifier.
- A hybrid model that produces state-dependent weights.
- Training, inference, persistence, and analysis helpers.

The hybrid output is state-adaptive and should not be conflated with one fixed linear witness operator.

`src/ml_models/variational_povm.py` implements a PyTorch parameterized unitary and learned two-outcome measurement operating on density matrices. This is a simulated variational POVM, not a PennyLane circuit and not an amplitude-encoding implementation.

`src/ml_models/amplitude_qml.py` implements 36D-to-64D zero-padded amplitude embedding on six qubits, `StronglyEntanglingLayers`, two Pauli-Z output logits, batched training, early stopping, prediction, persistence, and explicit split capture. Zero-norm vectors are rejected because they cannot define an amplitude-encoded state.

`src/ml_models/direct_state_qml.py` accepts batched three-qubit density matrices directly. PennyLane constructs the trainable `StronglyEntanglingLayers` unitary; PyTorch applies `U rho U†` on ROCm and evaluates two Pauli-Z logits. This avoids the 36-feature measurement stage but assumes the density matrix can be supplied in simulation.

`src/ml_models/qml_training.py` contains the shared split, optimization, inference, metrics, and persistence behavior used by both PennyLane classifiers.

### Experiment pipeline

`scripts/run_manuscript_experiments.py` is the sole result-generation entry point. Its classical stage records dataset leakage checks, family-only shortcuts, family-held-out evaluation, paired 36D-vs-63D nonlinear ablations, connected-correlator controls, and boundary generalization. Its QML stage evaluates the three normalization controls, amplitude QML, and direct-state QML on one split. It emits one detailed JSON artifact and one derived metrics CSV; it does not generate LaTeX.

## Known documentation and reproducibility gaps

1. The unified artifacts are generated locally but are not yet committed as the reviewed baseline.
2. Several historical values differ: SVM accuracy appears as 85.3%, 85.6%, and 86.3% for different analyses or summaries.
3. Shared split indices and paired test predictions are emitted by the controlled comparison, but no report-scale split artifact has been frozen yet.
4. CVXPY remains outside the general `requirements.txt`, but it is included in the machine-specific `requirements-rocm.lock` used by the audited environment.
5. The near-perfect nonlinear results are now shown to be confounded by family identity: every negative standard-dataset example is a product state.
6. Classical boundary evaluation and three-seed QML comparison are complete, but broader generators with negatives in multiple families remain necessary.
7. The direct-state classifier does not establish a practical mixed-state preparation protocol.

## Current research priorities

1. Scientifically review and commit the unified result artifacts.
2. Redesign the dataset so negative labels occur in multiple state families, then repeat the same unified pipeline.
3. Treat quantum-advantage and robust-classification claims as unsupported until that distribution-shift test succeeds.

## Status discipline

- “Implemented” means code is present; it does not mean the latest checkout was executed successfully.
- “Reported” means a manuscript or prior project record states the result.
- “Verified” requires a recorded command and artifact from the identified checkout.
- New metrics belong in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md), not in this inventory.
