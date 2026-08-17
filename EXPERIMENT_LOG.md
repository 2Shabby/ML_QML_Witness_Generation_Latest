# Experiment and Validation Log

**Status:** Active research record
**Last consolidated:** 2026-08-17
**Authoritative research context:** [three_qubit_distillability_research_context.md](three_qubit_distillability_research_context.md)

This file records reported experiments and validation work without treating an untracked or unreproduced number as a fresh result. New experiments should append a dated entry and link their machine-readable artifacts in `results/`.

## Evidence labels

- **MANUSCRIPT-REPORTED:** taken from the current supplied manuscript; not reproduced in this checkout.
- **LEGACY-REPORTED:** taken from the December 2025 project reports; not reproduced in this checkout.
- **CHECKOUT-VERIFIED:** established directly from the current files or a command recorded here.
- **PROPOSED:** not yet performed.

## Current baseline reported by the manuscript

**Date recorded:** 2026-08-17
**Status:** MANUSCRIPT-REPORTED
**Dataset:** 5,000 synthetic three-qubit states; 1,000 each from noisy GHZ, noisy W, noisy cluster, Haar-random mixed, and product-state families.
**Labels:** NPT across at least one of `A|BC`, `B|AC`, or `C|AB`.
**Restricted input:** 36 one- and two-body Pauli expectation values, grouped into 12 measurement settings.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Linear SVM | 86.3% | 86.9% | 97.5% | 91.9% |
| MLP | 100.0% | 100.0% | 100.0% | 100.0% |
| Transformer, standard | 99.5% | 99.4% | 100.0% | 99.7% |
| Transformer, hybrid | 100.0% | 100.0% | 100.0% | 100.0% |

The older December 2025 summary reported 85.6% SVM accuracy and 99.1% recall. The current manuscript table takes precedence until regenerated artifacts establish a canonical run.

### Per-family accuracy

| Family | SVM | MLP | Transformer |
|---|---:|---:|---:|
| Noisy GHZ | 100.0% | 100.0% | 100.0% |
| Noisy W | 100.0% | 100.0% | 100.0% |
| Noisy cluster | 100.0% | 100.0% | 100.0% |
| Random mixed | 98.1% | 100.0% | 99.8% |
| Product | 32.2% | 100.0% | 99.0% |

These nearly perfect nonlinear results require leakage, shortcut, family-generalization, and boundary stress tests before being treated as evidence of general sufficiency.

## EXPERIMENT_2025_SVM_36D_VS_63D

**Status:** LEGACY-REPORTED
**Question:** Does adding three-body Pauli terms improve a linear SVM?
**Method:** Five-fold cross-validation on the synthetic dataset.

| Feature set | Accuracy |
|---|---:|
| 36D one- and two-body | 85.3% +/- 0.6% |
| 63D all non-identity Paulis | 84.2% +/- 1.6% |

Paired test: `p = 0.148`. The supported conclusion is only that this run found no statistically significant linear-SVM improvement from the 27 three-body terms. It does not establish that three-body terms are unnecessary for nonlinear models or broader state distributions.

## EXPERIMENT_2025_FEATURE_AND_BOUNDARY_ANALYSIS

**Status:** LEGACY-REPORTED

- 67.5% of linear-SVM coefficient weight was assigned to two-body terms.
- Pearson correlation between SVM and transformer feature-importance summaries was reported as `r = 0.26`.
- Accuracy near the PPT/NPT boundary was approximately 60%.
- The main linear-model failure was the product-state family, consistent with nonlinear factorization relations such as `<P_i Q_j> = <P_i><Q_j>`.

The earlier report's statement that the transformer yields an “extractable witness operator” should be interpreted carefully: the hybrid model produces state-dependent weights, not one fixed linear Hermitian witness valid for every state.

## EXPERIMENT_2025_NOISE_STUDIES

**Status:** LEGACY-REPORTED

Depolarizing-noise results reported by the current manuscript:

| Noise `p` | SVM | MLP | Transformer |
|---:|---:|---:|---:|
| 0.0 | 84.0% | 100.0% | 99.5% |
| 0.1 | 85.5% | 100.0% | 99.7% |
| 0.2 | 89.0% | 100.0% | 99.8% |
| 0.3 | 91.0% | 100.0% | 100.0% |
| 0.4 | 83.5% | 100.0% | 99.2% |

Legacy tests also reported 100% classification on 100 samples each for dephasing, amplitude damping, and asymmetric local noise. These small studies are preliminary.

## VALIDATION_2025_NPT_AND_PIPELINE

**Date:** 2025-12-17
**Status:** LEGACY-REPORTED
**Original scope:** 56-test version of the repository.

Durable validation findings from the old audit:

- Partial transpose and all three three-qubit bipartitions were checked on known states.
- The qubit-permutation helper was reported to preserve trace, Hermiticity, and positivity.
- Pure three-qubit cluster-state generation matched a manual CZ construction and its stabilizers.
- Product states were reported PPT and pure GHZ/W/cluster states NPT.
- Depolarized GHZ/W/cluster states crossed the NPT threshold near noise `0.8` under the implemented noise convention.
- The 36 restricted Pauli features stayed in `[-1, 1]` and grouped into 12 settings.

This was a point-in-time audit, not verification of the present checkout.

## CHECKOUT_2026_08_17

**Status:** CHECKOUT-VERIFIED

- 42 Python files are present.
- Ten test modules contain 123 focused test functions: 5 amplitude QML, 3 controlled comparison, 3 direct-state QML, 21 DPS, 7 feature extraction, 4 integration, 22 MLP, 21 state generation, 18 transformer, and 19 variational POVM.
- `python3 -m compileall -q src scripts tests` succeeds.
- `results/` contains only `.gitkeep`; no committed JSON result artifacts support the reported metrics.
- An ignored local Python 3.12 environment was installed at `env/rocm` with PyTorch 2.12.1 + ROCm 7.2, PennyLane 0.45.1, CVXPY 1.9.2, and all declared project dependencies. `pip check` reports no broken requirements.
- The Radeon RX 7800 XT (`gfx1101`) completed GPU tensor operations through PyTorch's ROCm backend.
- `env/rocm/bin/python -m pytest -q` reports 123 passed and five warnings.
- `src/ml_models/amplitude_qml.py`, `scripts/run_amplitude_qml_experiment.py`, and `tests/test_amplitude_qml.py` implement and verify the six-qubit amplitude-encoded classifier path.
- A standalone feasibility check successfully amplitude-embedded a 36-value GPU tensor into six qubits with zero padding and normalization, executed a `StronglyEntanglingLayers` circuit through PennyLane's PyTorch interface, and backpropagated finite gradients on the ROCm device. This is environment validation, not a trained-model result.
- A small checkout smoke run verified batched optimization and the complete CLI pipeline on ROCm. Its deliberately undersized, one-epoch metrics are not recorded as experimental evidence.
- `src/feature_extraction/preprocessing.py` implements raw, row-wise L2-normalized, and L2-normalized-plus-original-norm controls.
- `scripts/run_controlled_qml_comparison.py` uses one stored stratified split for all three MLP controls and amplitude QML, and records aligned test labels and predictions. A 20-sample, one-epoch ROCm smoke run verified execution and serialization; its metrics are intentionally not recorded as evidence.
- `src/ml_models/direct_state_qml.py` uses PennyLane to construct a trainable circuit unitary and applies it directly to batched three-qubit density matrices on ROCm. A 20-sample, one-epoch CLI smoke run verified generated-state conversion, training, splitting, and serialization; its metrics are intentionally not recorded as evidence.
- `src/ml_models/qml_training.py` consolidates the shared learner behavior used by the amplitude and direct-state classifiers.
- The implemented variational POVM is a PyTorch density-matrix classifier. It is distinct from the proposed six-qubit amplitude-encoded variational quantum classifier.

## Proposed next experiments

Implementation steps 1–4 are complete: amplitude QML, its three classical normalization controls, one identical-split comparison pipeline, and the separate PennyLane direct-state classifier. The remaining work is validation:

1. Audit nonlinear results for leakage, family shortcuts, margin effects, and distribution shift.
2. Repeat 36D-vs-63D ablations for nonlinear models.
3. Run the controlled and direct-state QML experiments at report scale and freeze final baselines only after the implementation stabilizes.

## Entry template

```text
## EXPERIMENT_<DATE>_<ID>
STATUS: PROPOSED | RUNNING | COMPLETE | SUPERSEDED
QUESTION:
CODE_COMMIT:
ENVIRONMENT:
INPUT:
DATASET_VERSION:
SPLIT_ARTIFACT:
METHOD:
BASELINES:
SEEDS:
METRICS:
RESULT:
INTERPRETATION:
LIMITATIONS:
ARTIFACTS:
```
