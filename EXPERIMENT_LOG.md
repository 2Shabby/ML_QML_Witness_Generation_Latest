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

This was a point-in-time audit, not verification of the present 118-test checkout.

## CHECKOUT_2026_08_17

**Status:** CHECKOUT-VERIFIED

- 34 Python files are present.
- Seven test modules contain 118 test functions: 24 DPS, 7 feature extraction, 5 integration, 24 MLP, 21 state generation, 18 transformer, and 19 variational POVM.
- `python3 -m compileall -q src scripts tests` succeeds.
- `results/` contains only `.gitkeep`; no committed JSON result artifacts support the reported metrics.
- The available system Python lacks SciPy, scikit-learn, Qiskit, PyTorch, CVXPY, pytest, and PennyLane, so tests and experiments were not run during this consolidation.
- No PennyLane or amplitude-embedding implementation exists in `src/`, `scripts/`, or `tests/`.
- The implemented variational POVM is a PyTorch density-matrix classifier. It is distinct from the proposed six-qubit amplitude-encoded variational quantum classifier.

## Proposed next experiments

1. Reproduce and freeze the classical baseline with environment metadata, split indices, seeds, configuration, and JSON outputs.
2. Run the raw-vs-L2-normalized classical control required for a fair amplitude-encoding comparison.
3. Audit nonlinear results for leakage, family shortcuts, margin effects, and distribution shift.
4. Repeat 36D-vs-63D ablations for MLP, transformer, and future QML models.
5. Implement and evaluate the six-qubit amplitude-encoded classifier on the identically transformed input.
6. Treat direct density-matrix/state input as a separate operational experiment.

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
