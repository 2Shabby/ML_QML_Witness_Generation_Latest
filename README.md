# Three-Qubit Distillability from Restricted Pauli Measurements

This repository studies whether three-qubit distillability can be classified from 36 experimentally accessible one- and two-body Pauli expectation values instead of full state tomography.

The classical baseline and a controlled six-qubit amplitude-QML comparison are implemented. The active research direction is to validate these models rigorously and evaluate the separate direct-state circuit proposal.

## Research sources of truth

| Document | Purpose |
|---|---|
| [three_qubit_distillability_research_context.md](three_qubit_distillability_research_context.md) | Authoritative research question, manuscript context, QML proposals, limitations, and backlog |
| [CURRENT_STATUS.md](CURRENT_STATUS.md) | Live implementation and verification status |
| [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) | Dated reported results, validation provenance, and reproduction status |

Reported metrics are not presented as newly reproduced results. The repository currently has no committed result JSON files; see the experiment log for evidence labels and exact qualifications.

## Core experiment

```text
three-qubit density matrix
          |
          +--> full-state NPT calculation --> training/evaluation label
          |
          +--> 36 one- and two-body Pauli expectations --> classifier
```

The classifier receives:

- 9 single-qubit expectations: `X`, `Y`, and `Z` on each qubit.
- 27 pairwise expectations: every `P_i Q_j` with `P,Q in {X,Y,Z}` and `i < j`.
- No three-body Pauli terms in the restricted-input experiment.

The 36 observables can be grouped into 12 measurement settings. The full density matrix is used to generate NPT-based labels, not as input to the restricted classical classifiers.

## Implemented models

| Component | Input | Role |
|---|---|---|
| Linear SVM | 36D Pauli features | Interpretable fixed linear witness baseline |
| MLP | 36D Pauli features | Nonlinear classical baseline |
| Transformer | 36D Pauli features | Nonlinear standard and state-adaptive hybrid models |
| Amplitude QML | 36D Pauli features | Six-qubit zero-padded amplitude-encoded classifier |
| Variational POVM | Three-qubit density matrix | PyTorch simulation of a learned measurement |
| Random forest / gradient boosting | 36D Pauli features | Supplementary nonlinear controls |

The amplitude-QML and direct-state variational POVM models represent different operational questions and should not be conflated.

## Project layout

```text
.
├── README.md
├── CURRENT_STATUS.md
├── EXPERIMENT_LOG.md
├── three_qubit_distillability_research_context.md
├── requirements.txt
├── requirements-rocm.lock
├── src/
│   ├── config.py
│   ├── feature_extraction/pauli_features.py
│   ├── feature_extraction/preprocessing.py
│   ├── ml_models/
│   │   ├── amplitude_qml.py
│   │   ├── svm_witness.py
│   │   ├── mlp_classifier.py
│   │   ├── transformer_witness.py
│   │   └── variational_povm.py
│   ├── quantum_states/
│   │   ├── state_generation.py
│   │   └── distillability_oracles.py
│   └── utils/__init__.py
├── scripts/
├── tests/
└── results/
```

## Installation

The RX 7800 XT ROCm environment is the supported project environment. The existing ignored environment can be activated with:

```bash
source env/rocm/bin/activate
```

It uses Python 3.12, PyTorch 2.12.1 + ROCm 7.2, PennyLane 0.45.1, and CVXPY 1.9.2. This is the only supported environment for the current project. Recreate it from the machine-specific lock with:

```bash
uv python install 3.12
uv venv env/rocm --python 3.12
env/rocm/bin/python -m ensurepip --upgrade
env/rocm/bin/python -m pip install -r requirements-rocm.lock
```

## Quick start

```python
from src.feature_extraction.pauli_features import (
    create_sparse_measurement_set,
    extract_features_batch,
)
from src.ml_models import SVMWitnessLearner
from src.quantum_states.state_generation import generate_distillability_dataset

states, labels = generate_distillability_dataset(n_samples=500, seed=42)
basis = create_sparse_measurement_set(3, strategy="two_body")
features = extract_features_batch(states, basis, verbose=False)

learner = SVMWitnessLearner(pauli_basis=basis, C=1.0, kernel="linear")
metrics = learner.train(features, labels, test_size=0.2)
witness = learner.get_witness_operator()

print(features.shape)  # (500, 36)
print(metrics)
print(witness)
```

## Tests

The checkout contains 120 focused test functions across nine modules. Run them in the ROCm environment:

```bash
python -m pytest -q
```

Useful subsets:

```bash
python -m pytest tests/test_state_generation.py tests/test_feature_extraction.py -q
python -m pytest tests/test_mlp_classifier.py tests/test_transformer_witness.py -q
python -m pytest tests/test_variational_povm.py -q
python -m pytest tests/test_amplitude_qml.py -q
python -m pytest tests/test_controlled_comparison.py -q
python -m pytest tests/test_dps_oracle.py -q
```

The verified ROCm environment currently reports 120 passed and five warnings; see [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## Experiments

Classical SVM suite:

```bash
python scripts/run_experiments.py --experiment all --n-samples 5000 --seed 42
```

Transformer suite:

```bash
python scripts/run_transformer_experiments.py --experiment all --n-samples 5000 --seed 42
```

Additional suites:

```bash
python scripts/run_mlp_experiments.py
python scripts/run_povm_experiments.py
python scripts/run_noise_experiments.py
python scripts/run_supplementary_classifiers.py
python scripts/run_comparative_analysis.py --n-samples 5000 --seed 42 --save-plots
```

Amplitude-encoded QML:

```bash
python scripts/run_amplitude_qml_experiment.py \
  --n-samples 5000 --seed 42 --n-epochs 50 \
  --output results/amplitude_qml.json
```

Controlled normalization comparison on one shared split:

```bash
python scripts/run_controlled_qml_comparison.py \
  --n-samples 5000 --seed 42 --mlp-epochs 100 --qml-epochs 50 \
  --output results/controlled_qml_comparison.json
```

Plot saved result artifacts:

```bash
python scripts/plot_results.py --plot all --results-dir results --save
```

New reported results should be stored in `results/` and entered in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) with the commit, environment, seeds, split provenance, and artifact paths.

## Immediate research backlog

1. Evaluate the direct-state circuit proposal separately.
2. Audit nonlinear models for leakage and state-family shortcuts.
3. Repeat the 36D-vs-63D ablation for nonlinear models.
4. Run the controlled comparison at report scale and freeze final baselines after implementation stabilizes.

## License

MIT License.
