# Current Implementation Status

**Status:** Live code inventory

**Last audited:** 2026-08-18

**Research direction:** [three_qubit_distillability_research_context.md](three_qubit_distillability_research_context.md)

**Results provenance:** [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)

This document describes what exists in the current checkout. It intentionally separates implementation presence from experimental validation.

## Summary

The repository implements the confound-resistant balanced dataset generator, the restricted 36D / full 63D Pauli feature machinery, the classical comparators (linear SVM, MLP, transformer), a reusable six-qubit amplitude-encoded QML classifier, a separate PennyLane direct-state classifier, and one consolidated result-generation entry point. On 2026-08-18 the checkout was reduced by 44.6% (9,061 → 5,017 Python LoC): the legacy 5-family generator, superseded experiment scripts, the unused variational-POVM and oracle modules, and their tests were removed (see `EXPERIMENT_LOG.md`, entry `CLEANUP_2026_08_18`).

| Area | Current state |
|---|---|
| Balanced boundary-mixture dataset generator | Implemented (frozen; seeds 2026-2030) |
| NPT labeling across three bipartitions | Implemented |
| Restricted 36D and full 63D Pauli features | Implemented |
| Linear SVM witness | Implemented |
| MLP classifier | Implemented |
| Transformer classifiers (standard + hybrid) | Implemented |
| Amplitude-encoded QML (PennyLane, 6 qubits) | Implemented and ROCm-verified |
| Classical amplitude-encoding controls (raw / L2 / L2+norm) | Implemented |
| Direct-state QML (PennyLane, 3-qubit density matrix) | Implemented and ROCm-verified |
| Local Python 3.12 / ROCm research environment | Installed and GPU-verified |
| Consolidated clean-dataset experiment runner | Implemented; first full run in progress |

## Active implementation plan

| Step | Status | Scope |
|---:|---|---|
| 1 | Complete | Confound-resistant balanced dataset generator |
| 2 | Complete | Amplitude QML, classical normalization controls, direct-state QML |
| 3 | Complete | One consolidated runner (`scripts/run_clean_dataset_experiments.py`) with the deterministic 64/16/20 split for seeds 2026-2030 |
| 4 | In progress | Full clean-dataset run: seeds 2026-2029 completed once on the pre-cleanup tree; the run must be re-run end to end post-cleanup (a GPU hang aborted the previous run on seed 2030) |
| 5 | Deferred | Report-scale claims review after the consolidated artifact exists |

## Checkout verification (2026-08-18, post-cleanup)

- 18 Python files under `src`/`scripts` (5,017 LoC total across `src`/`scripts`/`tests`).
- Nine test modules define 71 test functions; `env/rocm/bin/python -m pytest -q` reports 71 passed.
- `python3 -m compileall -q src scripts tests` succeeds.
- The `--smoke` mode of the clean-dataset runner completed end to end on ROCm (state generation → features → all classical models → amplitude QML → direct-state QML → JSON). Smoke metrics are not research evidence.

Test counts by module:

| Module | Test functions |
|---|---:|
| `test_amplitude_qml.py` | 5 |
| `test_balanced_dataset.py` | 5 |
| `test_controlled_comparison.py` | 3 |
| `test_direct_state_qml.py` | 3 |
| `test_feature_extraction.py` | 4 |
| `test_integration.py` | 2 |
| `test_mlp_classifier.py` | 22 |
| `test_state_generation.py` | 9 |
| `test_transformer_witness.py` | 18 |
| **Total** | **71** |

## Implemented modules

### Configuration

`src/config.py` holds only what live code consumes: `TransformerConfig` (architecture + training defaults) and `DEFAULT_LOG_FORMAT`. The MLP, SVM, and QML learners keep their hyperparameters as constructor defaults; the experiment runner records exact settings in its JSON artifact.

### State generation and labels

`src/quantum_states/balanced_dataset.py` generates the confound-resistant dataset: boundary-crossing mixtures `rho(q) = q * rho_NPT + (1-q) * rho_PPT` with `q` sampled around the pair's PPT/NPT boundary `q*` (bisection). Families: generalized GHZ, generalized W, cluster, Haar-random pure (NPT anchors) mixed with non-product classically-correlated PPT anchors. Both labels occur in every family.

`src/quantum_states/state_generation.py` keeps only the label-oracle primitives: `partial_transpose`, `_permute_qubits`, and `check_npt_any_bipartition` (the NPT answer key: positive if any of `A|BC`, `B|AC`, `C|AB` has a negative partial-transpose eigenvalue).

### Restricted measurements

`src/feature_extraction/pauli_features.py` provides the full Pauli basis, the 36D one-/two-body measurement set, commuting grouping, and measurement-cost estimation. `src/feature_extraction/preprocessing.py` provides the raw, L2-normalized, and normalized-plus-norm amplitude-encoding controls.

### Models

- `src/ml_models/svm_witness.py`: linear SVM with witness → `SparsePauliOp` conversion.
- `src/ml_models/mlp_classifier.py`: `in -> 128 -> 64 -> 32 -> 2` PyTorch classifier (batch norm, LeakyReLU, dropout), with validation early stopping.
- `src/ml_models/transformer_witness.py`: standard and hybrid (state-dependent weights) transformer classifiers with training, inference, persistence, and witness/attention analysis helpers.
- `src/ml_models/amplitude_qml.py`: 36D-to-64D zero-padded amplitude embedding on six qubits, `StronglyEntanglingLayers`, two Pauli-Z logits, batched training with early stopping.
- `src/ml_models/direct_state_qml.py`: PennyLane unitary applied directly to batched three-qubit density matrices on ROCm. Separate from the restricted-measurement experiment.
- `src/ml_models/qml_training.py`: shared split, optimization, inference, metrics, and persistence for both PennyLane classifiers.

### Experiment pipeline

`scripts/run_clean_dataset_experiments.py` is the sole result-generation entry point. Per seed (2026-2030) it runs: sanity/shortcut controls (family balance, family-only, norm-only, purity diagnostic, shallow tree); 36D linear SVM, MLP, standard + hybrid transformer; 36D-vs-63D ablation for SVM and MLP (n=5 seed-level paired differences, descriptive); amplitude QML vs matched L2-normalized MLP; and direct-state QML vs the 63D full-state-information MLP. All models share one deterministic stratified 64/16/20 split (split seed 0); models without early stopping ignore the validation set. It emits one consolidated JSON with per-seed results, mean/std, per-family accuracy, protocol, model settings, and provenance.

## Known documentation and reproducibility gaps

1. The consolidated clean-dataset artifact does not exist yet; the previous full run aborted on a ROCm GPU hang (seed 2030, amplitude QML stage) and is being re-run.
2. Historical manuscript numbers (see `EXPERIMENT_LOG.md`) are superseded by the pending clean-dataset artifact and must not be mixed with it.
3. CVXPY is no longer a project dependency after the oracle module removal.

## Current research priorities

1. Complete the clean-dataset run (all five seeds) and commit the consolidated JSON as the canonical artifact.
2. Treat quantum-advantage and robust-classification claims as unsupported until that artifact is reviewed.

## Status discipline

- “Implemented” means code is present; it does not mean the latest checkout was executed successfully.
- “Reported” means a manuscript or prior project record states the result.
- “Verified” requires a recorded command and artifact from the identified checkout.
- New metrics belong in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md), not in this inventory.
