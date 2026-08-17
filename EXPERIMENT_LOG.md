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

- 30 Python files are present.
- Ten test modules contain 124 focused test functions: 5 amplitude QML, 3 controlled comparison, 3 direct-state QML, 21 DPS, 7 feature extraction, 4 integration, 22 MLP, 22 state generation, 18 transformer, and 19 variational POVM.
- `python3 -m compileall -q src scripts tests` succeeds.
- `results/` contains only `.gitkeep`; no committed JSON result artifacts support the reported metrics.
- An ignored local Python 3.12 environment was installed at `env/rocm` with PyTorch 2.12.1 + ROCm 7.2, PennyLane 0.45.1, CVXPY 1.9.2, and all declared project dependencies. `pip check` reports no broken requirements.
- The Radeon RX 7800 XT (`gfx1101`) completed GPU tensor operations through PyTorch's ROCm backend.
- `env/rocm/bin/python -m pytest -q` reports 124 passed and five warnings.
- `src/ml_models/amplitude_qml.py` and `tests/test_amplitude_qml.py` implement and verify the six-qubit amplitude-encoded classifier path.
- A standalone feasibility check successfully amplitude-embedded a 36-value GPU tensor into six qubits with zero padding and normalization, executed a `StronglyEntanglingLayers` circuit through PennyLane's PyTorch interface, and backpropagated finite gradients on the ROCm device. This is environment validation, not a trained-model result.
- A small checkout smoke run verified batched optimization and the complete CLI pipeline on ROCm. Its deliberately undersized, one-epoch metrics are not recorded as experimental evidence.
- `src/feature_extraction/preprocessing.py` implements raw, row-wise L2-normalized, and L2-normalized-plus-original-norm controls.
- The unified pipeline uses one stored stratified split for all three MLP controls and amplitude QML. A 20-sample, one-epoch ROCm smoke run verified execution and serialization; its metrics are intentionally not recorded as evidence.
- `src/ml_models/direct_state_qml.py` uses PennyLane to construct a trainable circuit unitary and applies it directly to batched three-qubit density matrices on ROCm. A 20-sample, one-epoch CLI smoke run verified generated-state conversion, training, splitting, and serialization; its metrics are intentionally not recorded as evidence.
- `src/ml_models/qml_training.py` consolidates the shared learner behavior used by the amplitude and direct-state classifiers.
- `scripts/run_manuscript_experiments.py` is the only result generator. Legacy generators, result consumers tied to their incompatible schemas, and hard-coded legacy TeX manuscripts were removed. The unified JSON and derived CSV are intended as inputs to later manuscript work; the pipeline does not generate LaTeX.
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

## EXPERIMENT_2026_08_17_UNIFIED_VALIDATION

**Status:** COMPLETE; manuscript-usable artifact generated, scientific caveats required.

- **Code identity:** base commit `14aa6da91c7fe44b35a2eb0ae28fac88a99fe6aa`; exact dirty-tree Python source SHA-256 `6b60f0caead06f913a97f712fc1f0e38e45241ea12ad8637867e8c3671a0c23d`.
- **Environment:** Python 3.12.13, NumPy 2.5.2, scikit-learn 1.9.0, PyTorch 2.12.1+rocm7.2, ROCm 7.2.53211, Radeon RX 7800 XT.
- **Artifacts:** `results/manuscript/results.json` is authoritative; `results/manuscript/metrics.csv` is its flat scalar-metric view.
- **Classical dataset:** 5,000 states, seed 42, exact state/label hash and split indices stored in JSON. Boundary evaluation used 1,500 separately generated noisy GHZ/W/cluster states.
- **QML datasets:** 1,000 states for each of seeds 42, 123, and 456; each dataset hash and split is stored independently; maximum 20 epochs.

### Central audit finding

The standard generator produced 4,000 positive states and 1,000 negative states. Every negative state was a product state; GHZ, W, cluster, and random-mixed families were entirely positive. Consequently, a classifier using only the family identity achieved 100% random-split accuracy. There were no exact feature-row overlaps between train and test, so the main shortcut is family/label confounding rather than direct row duplication.

The nonlinear random-split accuracies of 100% therefore do not establish general distillability recognition. On the separately generated near-boundary entangled-family set (789 negative, 711 positive), linear SVM, RBF SVM, random forest, and gradient boosting each achieved 47.4% accuracy, consistent with predicting only the positive class. Product-family-held-out training was not estimable because the remaining training set contained only positive labels.

### Feature and QML results

- Paired five-fold 36D-vs-63D mean accuracy: linear SVM 85.32% vs 84.20% (`p=0.148`); RBF SVM 100% vs 100%; random forest 100% vs 99.96%; gradient boosting 100% vs 100%. These random-fold results remain subject to the family shortcut.
- Linear controls on one random split: local 9D 80.0%, restricted 36D 86.3%, connected 27D 83.8%, restricted-plus-connected 63D 85.4%, full Pauli 63D 83.6%.
- Across three QML seeds, accuracy mean ± sample standard deviation: raw MLP 100.0% ± 0.0%; L2-normalized MLP 83.67% ± 1.04%; normalized-plus-original-norm MLP 100.0% ± 0.0%; amplitude QML 81.33% ± 2.31%; direct-state QML 80.67% ± 1.76%.
- Corresponding balanced accuracy: raw MLP 100.0%, L2-normalized MLP 81.67%, normalized-plus-norm MLP 100.0%, amplitude QML 73.96%, direct-state QML 70.42%.

### Interpretation limits

These results support two defensible conclusions: the original synthetic random split has a severe family-label shortcut, and amplitude normalization discards predictive norm information. They do **not** support a quantum advantage claim or robust out-of-distribution distillability classification. Classical validation currently uses one dataset seed (with five fixed CV folds), QML uses three seeds, labels remain the implemented NPT proxy, and no hardware experiment was performed.
