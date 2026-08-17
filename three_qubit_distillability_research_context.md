# Three-Qubit Distillability from Restricted Pauli Measurements
## Consolidated Research Context

**Project:** Machine-Learning Classification of Three-Qubit Distillability from Restricted Pauli Measurements
**Working subtitle:** ML Classification for Quantum Information Processing
**Primary researcher:** Shahbaz Shaik
**Collaborators / advisors in supplied correspondence:** Dr. Pratibha Hegde, Dr. Indranil Chakrabarty
**Context assembled:** 2026-08-17
**Purpose:** Single, machine-parsable research context containing the supplied email discussion, current manuscript claims/results, QML extension suggestions, verified PennyLane implementation facts, and open questions.
**Document status:** AUTHORITATIVE RESEARCH DIRECTION
**Reported-results record:** [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)
**Live implementation status:** [CURRENT_STATUS.md](CURRENT_STATUS.md)

This document supersedes the former `GOAL.md`. It defines the active research questions and backlog; it does not by itself certify that a reported experiment has been reproduced from the current checkout.

---

# 1. RESEARCH_QUESTION

Can three-qubit distillability be classified using only experimentally cheaper one-body and two-body Pauli expectation values, rather than reconstructing the full quantum state by tomography?

The current classical-ML study tests whether a 36-dimensional vector of restricted Pauli expectation values contains enough information for this decision.

---

# 2. CORE_EXPERIMENT

## 2.1 INPUT_FEATURES

The classifier input is a 36-dimensional real-valued feature vector:

- 9 one-body Pauli expectation values:
  - `<X_i>`
  - `<Y_i>`
  - `<Z_i>`
  - for qubits `i = 1,2,3`
- 27 two-body Pauli expectation values:
  - `<P_i tensor Q_j>`
  - `P,Q in {X,Y,Z}`
  - `i < j`

These 36 observables can be obtained from 12 grouped measurement settings.

Three-body Pauli terms are deliberately excluded from the restricted-input experiment.

## 2.2 LABEL_ORACLE

The full density matrix is used only to generate the ground-truth label.

For each of the three bipartitions:

- `A | BC`
- `B | AC`
- `C | AB`

the partial transpose is evaluated.

The manuscript labels a state as distillable when at least one bipartition is NPT, i.e. when the minimum eigenvalue of a relevant partial transpose is negative.

Important separation:

- **Oracle / answer key sees:** full density matrix.
- **Classifier sees:** only the 36 restricted Pauli features.

## 2.3 DATASET

Total: **5,000 three-qubit states**, with **1,000 states per family**.

Families:

1. Noisy GHZ states
2. Noisy W states
3. Noisy three-qubit cluster states
4. Haar-random mixed states
5. Product states

The noisy structured families use depolarizing noise in the main dataset.

Product states are explicitly included as non-distillable examples that may nevertheless show non-zero two-body correlations.

---

# 3. MODELS

## 3.1 LINEAR_SVM

- Input: 36 features
- Linear decision function
- `C = 1.0`
- Interpretable as a restricted linear witness

## 3.2 MLP

Architecture reported in manuscript:

`36 -> 128 -> 64 -> 32 -> 1`

With:

- LeakyReLU(0.2)
- batch normalization
- dropout 0.3
- sigmoid output
- approximately 12k parameters
- Adam
- learning rate `1e-3`

## 3.3 TRANSFORMER

Single-layer encoder:

- projection dimension: 16
- 2 attention heads
- head dimension: 8
- feed-forward dimension: 32
- dropout: 0.1
- approximately 3k parameters

Two modes:

### Standard
Transformer encoder followed by classification head.

### Hybrid
Transformer generates a state-dependent weight vector `w(x)` and evaluates:

`score = w(x) dot x + b`

Interpretation proposed in manuscript: a state-adaptive witness.

---

# 4. ORIGINAL_FINDINGS

## 4.1 OVERALL_TEST_METRICS

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Linear SVM | 86.3% | 86.9% | 97.5% | 91.9% |
| MLP | 100.0% | 100.0% | 100.0% | 100.0% |
| Transformer (standard) | 99.5% | 99.4% | 100.0% | 99.7% |
| Transformer (hybrid) | 100.0% | 100.0% | 100.0% | 100.0% |

Note: an earlier email summary mentioned approximately 85.6% SVM overall accuracy and 99.1% recall. The current supplied manuscript reports 86.3% accuracy and 97.5% recall. Treat the manuscript table as the current version unless the experiment logs establish otherwise.

## 4.2 PER_FAMILY_ACCURACY

| Family | SVM | MLP | Transformer |
|---|---:|---:|---:|
| GHZ (noisy) | 100.0% | 100.0% | 100.0% |
| W (noisy) | 100.0% | 100.0% | 100.0% |
| Cluster (noisy) | 100.0% | 100.0% | 100.0% |
| Random mixed | 98.1% | 100.0% | 99.8% |
| Product | 32.2% | 100.0% | 99.0% |

## 4.3 INTERPRETATION_OF_SVM_PRODUCT_FAILURE

Product states can have non-zero two-body correlations satisfying factorization relations such as:

`<Z_i tensor Z_j> = <Z_i><Z_j>`

The manuscript interpretation is that a simple linear boundary cannot reliably distinguish such factorized correlations from correlations induced by entanglement.

Nonlinear models learn the relevant nonlinear/factorization structure on the present dataset.

## 4.4 THREE_BODY_ABLATION

Linear SVM, 5-fold cross-validation:

| Feature set | Accuracy |
|---|---:|
| 36D: one-body + two-body | 85.3% +/- 0.6% |
| 63D: complete non-identity Pauli set | 84.2% +/- 1.6% |

Paired t-test:

`p = 0.148`

Current manuscript conclusion:

For the linear SVM experiment, adding the 27 three-body terms does **not** produce a statistically significant improvement.

Important limitation:

This ablation has not yet established the same conclusion for nonlinear models at larger scale.

## 4.5 FEATURE_IMPORTANCE

Reported:

- 67.5% of the linear SVM's discriminative weight lies on two-body correlations.
- Pearson correlation between SVM and transformer feature importances: `r = 0.26`.

Manuscript interpretation: the architectures may rely on substantially different subsets/structures in the measurement data.

## 4.6 NOISE_RESULTS

Reported depolarizing-noise sweep:

| p | SVM | MLP | Transformer |
|---:|---:|---:|---:|
| 0.0 | 84.0% | 100.0% | 99.5% |
| 0.1 | 85.5% | 100.0% | 99.7% |
| 0.2 | 89.0% | 100.0% | 99.8% |
| 0.3 | 91.0% | 100.0% | 100.0% |
| 0.4 | 83.5% | 100.0% | 99.2% |

Smaller tests of 100 samples each under:

- dephasing
- amplitude damping
- asymmetric local noise

are reported as reaching 100% accuracy for the nonlinear classifiers in the manuscript, with the explicit caveat that the sample size is small.

## 4.7 PPT_NPT_BOUNDARY

States near the PPT/NPT decision boundary, i.e. states with very small absolute minimum partial-transpose eigenvalues, reach only approximately **60% accuracy**.

This is an important present limitation.

---

# 5. CURRENT_MANUSCRIPT_CLAIM

The current practical protocol is:

1. Measure 12 grouped Pauli settings.
2. Construct the 36-dimensional vector of one- and two-body expectation values.
3. Pass the vector to a trained classifier.
4. Predict whether the state is distillable.

The proposed advantage is direct classification from a restricted measurement set rather than complete state reconstruction.

---

# 6. CURRENT_LIMITATIONS_IN_MANUSCRIPT

1. Synthetic dataset.
2. Only five structured state families.
3. No real-hardware validation.
4. Approximately 60% accuracy near the PPT/NPT boundary.
5. Three-body ablation has only been established for the linear classifier.
6. Three-qubit scope.
7. Scaling beyond three qubits changes the entanglement/distillability problem; the simple three-qubit NPT-based labeling logic does not transfer generally.

---

# 7. EMAIL_CHAIN

## EMAIL_1 — Shahbaz Shaik

> Extremely sorry for the delay, please find attached the document/manuscript as discussed on Wednesday.
>
> Thanks,
> Shahbaz Shaik

## EMAIL_2 — Dr. Pratibha Hegde

> Thanks for the notes Shahbaz.
>
> Are you at the institute today to meet? Let me know.
>
> Best regards,
> Pratibha.

## EMAIL_3 — Shahbaz Shaik

> Hi @Dr. Pratibha Hegde, I am currently sick and won't be able to make it today, sorry for the last moment update. I would be available throughout the next week, could we schedule a meeting anytime in the next week?
>
> Thanks,
> Shahbaz

## EMAIL_4 — Dr. Pratibha Hegde

> Ah, sorry to know. Hope you feel better soon. Yes, we can meet early next week.
>
> Best regards,
> Pratibha

## EMAIL_5 — Dr. Indranil Chakrabarty

> Dear Shabhaz,
>
> Can you explain these in details. A little more writing will help
>
> Best
> Indranil

## EMAIL_6 — Shahbaz Shaik

> Adding additional details for the above manuscript. I will add all of these into the manuscript and share the same by next week.
>
> The manuscript focuses on a simple question: can one tell whether a three-qubit state is distillable, meaning whether useful entanglement can be extracted from it, using only a limited set of measurement data rather than full quantum state tomography?
>
> The input to the classifier is a 36-dimensional set of measurement values made from all single-qubit and two-qubit Pauli expectations. These can be obtained from 12 grouped measurement settings, so the method is much lighter than full tomography. We deliberately leave out the three-qubit terms, since they are harder to measure reliably in practice.
>
> For the labels, the full state is used only to decide the ground truth through the NPT test across the three possible bipartitions. In other words, the exact state is used only to generate the answer key, while the model itself sees only the reduced measurement data. The main question is whether this reduced data still keeps enough information to detect distillability.
>
> The dataset contains 5,000 examples from five classes of states: noisy GHZ, noisy W, noisy cluster, random mixed states, and product states. Product states are included because they are not entangled, but can still show nonzero two-qubit correlations, which makes them the main source of possible confusion.
>
> The main finding so far is that these restricted measurements already contain most of the useful signal. A linear SVM reaches 85.6% overall accuracy with 99.1% recall on distillable states, meaning it rarely misses a genuinely distillable state. Its main weakness is with product states (32.2% accuracy there), where a simple linear boundary is not flexible enough to separate factorized correlations from entanglement-induced ones. Nonlinear models resolve this completely on the present dataset, which suggests that the limitation is not in the measurements themselves, but in the expressivity of the classifier.
>
> We also compared this restricted 36-feature version with the full 63-term Pauli description. In the present setting, adding the three-qubit terms did not produce a statistically meaningful improvement under the linear classifier. Whether this conclusion extends to other nonlinear models at larger scale is still open. But for the linear case, the lower-measurement approach retains the relevant information while being much cheaper experimentally.
>
> The practical point is that one may be able to avoid full state reconstruction and instead make the distillability decision directly from a much smaller measurement set.

## EMAIL_7 — Dr. Pratibha Hegde

> Ah, I think I now understand better. Thank you.
>
> Best regards,
> Pratibha.

## EMAIL_8 — Dr. Pratibha Hegde

> Dear Shahbaz,
>
> Thank you for the notes and explanation. My suggestion for a quick test using qml would be to use quantum circuits for classification. For example, you can give 36-dimensional data input using amplitude encoding and perform a classification task with labels just like MLP.
>
> Another way could be to prepare the entangled state directly using quantum circuits (state preparation) and continue to use the trainable gates as the second part of the circuit, and continue to classify states as before. This way, you don't have to perform measurements. However, state preparation could be difficult in practice. I think there is no harm in doing a theoretical study and verifying.
>
> Let me know what you think. We can have a chat next week if you are feeling better.
>
> Best regards,
> Pratibha.

---

# 8. QML_EXTENSION_SUGGESTIONS_FROM_EMAIL

These are **advisor suggestions**, not yet established findings.

## SUGGESTION_A — CLASSICAL_36D_TO_QML

Pipeline:

`36 Pauli features -> amplitude encoding -> trainable quantum circuit -> classification label`

Purpose:

Compare a variational quantum classifier against the classical MLP / SVM / transformer using the same reduced-measurement information.

## SUGGESTION_B — DIRECT_STATE_INPUT

Pipeline:

`three-qubit quantum state -> trainable quantum circuit -> output observable / classifier`

Purpose:

Test classification directly from a prepared quantum state, rather than first converting the state into the 36 measured classical features.

This is theoretically interesting but changes the operational setup.

---

# 9. VERIFIED_PENNYLANE_IMPLEMENTATION_FACTS

These items were checked against PennyLane documentation on 2026-08-17.

## 9.1 AMPLITUDE_EMBEDDING_DIMENSION

PennyLane `AmplitudeEmbedding` encodes `2^n` amplitudes into `n` qubits.

Therefore:

- 5 qubits provide 32 amplitudes: insufficient for 36 features.
- 6 qubits provide 64 amplitudes: sufficient for 36 features.

A 36-dimensional input can therefore be zero-padded to 64 entries for a 6-qubit amplitude embedding.

Source:
https://docs.pennylane.ai/en/stable/code/api/pennylane.AmplitudeEmbedding.html

## 9.2 NORMALIZATION

Amplitude embedding requires a valid normalized state vector.

The PennyLane implementation supports normalization and padding.

This creates a methodological issue for comparison with classical models:

`x -> x / ||x||_2`

may remove the original vector norm as a usable feature.

Therefore, before attributing a change in performance to a quantum classifier, a classical control experiment using the identically normalized input should be run.

This final comparison recommendation is an **inference / proposed experimental control**, not a result in the current manuscript.

## 9.3 MIXED_DENSITY_MATRIX_INPUT

PennyLane provides `QubitDensityMatrix`, which can initialize a subsystem using a supplied density matrix on compatible mixed-state simulation devices.

This matters because the current dataset contains mixed states, so direct state-input simulation cannot be represented solely as pure-state vector preparation.

Source:
https://docs.pennylane.ai/en/stable/code/api/pennylane.QubitDensityMatrix.html

## 9.4 QML_STILL_HAS_OUTPUT_MEASUREMENT

A variational quantum classifier still obtains a classical prediction from quantum measurements such as observable expectation values.

Therefore, the phrase "you don't have to perform measurements" in the email is best interpreted operationally as:

> you would not have to first measure the 36 Pauli observables to construct the classical feature vector.

It should **not** be interpreted literally as saying that the classifier requires no quantum measurement at all.

PennyLane measurement documentation:
https://docs.pennylane.ai/en/stable/introduction/measurements.html

Example of a variational circuit returning an expectation value:
https://docs.pennylane.ai/en/stable/introduction/operations.html

## 9.5 LOCAL_IMPLEMENTATION_FEASIBILITY_CHECK

**Status:** checkout-verified on 2026-08-17; environment validation only.

An ignored Python 3.12 environment at `env/rocm` contains PyTorch 2.12.1 + ROCm 7.2 and PennyLane 0.45.1. On the Radeon RX 7800 XT (`gfx1101`), a standalone smoke check:

1. supplied 36 values to `AmplitudeEmbedding` on six qubits,
2. used zero padding and normalization,
3. applied one `StronglyEntanglingLayers` block,
4. returned a Pauli-Z expectation value, and
5. backpropagated finite gradients through PennyLane's PyTorch interface on the ROCm device.

This establishes that the proposed software path can execute locally. It does **not** implement the repository classifier, choose a justified ansatz, train on the distillability dataset, provide controlled baselines, or establish a QML result.

---

# 10. IMPORTANT_METHOD_DISAMBIGUATION

There are two distinct research questions.

## QUESTION_A — SAME_INFORMATION_DIFFERENT_CLASSIFIER

> Given the same restricted 36-dimensional classical measurement vector, can a quantum classifier perform useful classification?

This preserves the paper's restricted-measurement premise.

Comparison set:

- Linear SVM
- MLP
- Transformer
- Quantum variational classifier

## QUESTION_B — QUANTUM_STATE_DIRECTLY

> Given direct access to the three-qubit quantum state, can a trainable quantum circuit classify its distillability?

This is a different operational setting.

It potentially avoids the 36-observable feature-construction stage, but requires assumptions about access to or preparation of the state and still needs an output measurement.

These two experiments should not be conflated.

---

# 11. PROPOSED_CONTROL_EXPERIMENT_0

**Status:** proposed in discussion; not yet performed.

Before implementing the QML amplitude-encoding classifier:

1. Take the existing 36D input vectors.
2. L2-normalize each vector exactly as required by amplitude encoding.
3. Train/evaluate the existing MLP on those normalized inputs.
4. Compare against the raw-feature MLP.

Reason:

If normalization itself changes classification accuracy, then a QML-vs-classical comparison must control for that information transformation.

Possible variants for rigorous comparison:

- Raw 36D classical baseline.
- L2-normalized 36D classical baseline.
- L2-normalized + original norm appended as a 37th classical feature.
- Amplitude-encoded QML using normalized 36D data.

---

# 12. OPEN_QUESTIONS

## OPEN_QUESTION_1
Does L2 normalization of the 36D Pauli feature vector reduce classical classification performance?

## OPEN_QUESTION_2
If performance changes, how much predictive information is carried by the norm of the 36D correlation vector?

## OPEN_QUESTION_3
How should the 36D-to-64D amplitude padding be handled to ensure a fair classical/QML comparison?

## OPEN_QUESTION_4
What variational ansatz and circuit depth are sufficient for the 6-qubit amplitude-encoded classifier?

## OPEN_QUESTION_5
Should the QML experiment be positioned as:
- a comparative classifier experiment,
- a hybrid quantum-classical witness construction,
- or a direct-state quantum classification study?

## OPEN_QUESTION_6
Does the 36D-vs-63D "three-body terms are unnecessary" result continue to hold for:
- MLP,
- transformer,
- QML classifier,
- larger datasets?

## OPEN_QUESTION_7
Why do nonlinear models reach essentially perfect accuracy on the present synthetic dataset?

This should be stress-tested for:
- train/test leakage,
- state-family shortcuts,
- noise-generation artifacts,
- label-margin effects,
- distribution shift,
- harder near-boundary samples.

## OPEN_QUESTION_8
Can explicitly constructed factorization features improve the linear/product-state problem?

Examples:
`<P_i Q_j> - <P_i><Q_j>`

These connected correlators may separate product-state factorization from entanglement-related correlation more directly.

## OPEN_QUESTION_9
Can the measurement set be reduced below 12 grouped settings while maintaining useful recall/accuracy?

## OPEN_QUESTION_10
How does performance change on real hardware or simulated correlated hardware noise?

---

# 13. NEAR_TERM_EXPERIMENT_BACKLOG

Ordered for information gain, not commitment.

Completed enabling work: the local ROCm/PennyLane environment and a differentiable six-qubit amplitude-embedding smoke check are verified. The following research and implementation work remains:

1. **Restore a fully passing test suite and record portable environment constraints.**
2. **Reproduce and freeze current classical results.**
3. **Run raw, L2-normalized, and norm-preserving classical controls.**
4. **Audit nonlinear 100% accuracy for leakage / shortcut learning.**
5. **Repeat 36D-vs-63D ablation for nonlinear classical models.**
6. **Implement 6-qubit amplitude-encoded variational quantum classifier.**
7. **Compare QML against classical model using identically transformed input.**
8. **Separately test direct density-matrix/state-input quantum classification in simulation.**
9. **Expand near-PPT/NPT-boundary evaluation.**
10. **Investigate connected-correlator / factorization-aware features.**
11. **Investigate sparse measurement subsets and hardware validation.**

---

# 14. PROVENANCE_MAP

| Item | Provenance |
|---|---|
| 36D restricted feature space | Supplied manuscript |
| 12 grouped measurement settings | Supplied manuscript |
| 5,000-state / five-family dataset | Supplied manuscript |
| NPT-based label oracle | Supplied manuscript |
| SVM / MLP / transformer results | Supplied manuscript |
| Product-state failure explanation | Supplied manuscript |
| 36D vs 63D ablation | Supplied manuscript |
| Boundary-state limitation | Supplied manuscript |
| Amplitude-encoding QML suggestion | Dr. Pratibha Hegde email |
| Direct-state QML suggestion | Dr. Pratibha Hegde email |
| 6-qubit requirement for 36D amplitude encoding | Mathematical consequence + PennyLane docs |
| Amplitude normalization requirement | PennyLane docs |
| L2-normalized classical baseline | Proposed control from ChatGPT discussion |
| Mixed density-matrix support | PennyLane docs |
| Output measurement still required in VQC | PennyLane docs / standard variational-circuit structure |
| Connected-correlator experiment | Proposed research direction |
| Leakage/shortcut audit | Proposed research validation |

---

# 15. SOURCE_MANUSCRIPT_REFERENCE

Attached manuscript supplied in ChatGPT conversation:

**Machine-Learning Classification of Three-Qubit Distillability from Restricted Pauli Measurements**
4 pages, current supplied version dated by upload context: 2026-08-17.

This context file intentionally preserves the manuscript's current claims without silently treating proposed follow-up experiments as completed results.

---

# 16. UPDATE_PROTOCOL

When future experiments are added, use the following structure:

```text
## EXPERIMENT_<ID>
DATE:
QUESTION:
INPUT:
DATASET_VERSION:
METHOD:
BASELINES:
METRICS:
RESULT:
INTERPRETATION:
LIMITATIONS:
ARTIFACTS:
STATUS: PROPOSED | RUNNING | COMPLETE | SUPERSEDED
```

When a result changes, retain the old result and mark it `SUPERSEDED` rather than deleting it. This makes the file suitable as a research log and as persistent LLM context.
