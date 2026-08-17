# Three-Qubit Distillability from Restricted Pauli Measurements

This repository studies whether three-qubit distillability can be classified from 36 experimentally accessible one- and two-body Pauli expectation values instead of full state tomography.

The classical baseline is implemented. The active research direction is to validate the reported nonlinear results and compare them fairly with quantum-classification approaches, including a proposed six-qubit amplitude-encoded variational classifier.

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
| Variational POVM | Three-qubit density matrix | PyTorch simulation of a learned measurement |
| Random forest / gradient boosting | 36D Pauli features | Supplementary nonlinear controls |

The variational POVM is not a PennyLane QML circuit and is not the proposed amplitude-encoded classifier.

## Project layout

```text
.
├── README.md
├── CURRENT_STATUS.md
├── EXPERIMENT_LOG.md
├── three_qubit_distillability_research_context.md
├── requirements.txt
├── src/
│   ├── config.py
│   ├── feature_extraction/pauli_features.py
│   ├── ml_models/
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

Create an isolated Python environment, then install the declared dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CVXPY is presently commented out in `requirements.txt`. Install it separately before running SDP-specific tests:

```bash
python -m pip install 'cvxpy>=1.4.0'
```

PennyLane is not yet a declared project dependency because the proposed amplitude-encoded QML extension has not been implemented. Install it for QML development with:

```bash
python -m pip install 'pennylane>=0.45,<0.46'
```

### Verified local ROCm environment

On the audited RX 7800 XT workstation, the ignored environment at `env/rocm` is already installed and can be activated with:

```bash
source env/rocm/bin/activate
```

It uses Python 3.12, PyTorch 2.12.1 + ROCm 7.2, PennyLane 0.45.1, and CVXPY 1.9.2. For a fresh compatible environment, install the ROCm Torch build before the general requirements so `requirements.txt` does not select the default CUDA build:

```bash
uv python install 3.12
uv venv env/rocm --python 3.12
env/rocm/bin/python -m ensurepip --upgrade
env/rocm/bin/python -m pip install torch==2.12.1 \
  --index-url https://download.pytorch.org/whl/rocm7.2
env/rocm/bin/python -m pip install -r requirements.txt \
  'cvxpy>=1.4.0' 'pennylane>=0.45,<0.46'
```

This ROCm command is machine-specific. A portable lock/constraints strategy and documented CPU fallback remain pending.

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

The checkout contains 118 test functions across seven modules. Run them in a fully installed environment:

```bash
python -m pytest -q
```

Useful subsets:

```bash
python -m pytest tests/test_state_generation.py tests/test_feature_extraction.py -q
python -m pytest tests/test_mlp_classifier.py tests/test_transformer_witness.py -q
python -m pytest tests/test_variational_povm.py -q
python -m pytest tests/test_dps_oracle.py -q
```

The verified ROCm environment currently reports 117 passed and one failed test. `TestMLPDiscriminator.test_single_sample` fails because `BatchNorm1d` receives one sample while the model is in training mode. Six warnings are also reported; see [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

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

Plot saved result artifacts:

```bash
python scripts/plot_results.py --plot all --results-dir results --save
```

New reported results should be stored in `results/` and entered in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) with the commit, environment, seeds, split provenance, and artifact paths.

## Immediate research backlog

1. Fix the single-sample MLP/BatchNorm failure and record portable environment constraints.
2. Reproduce and freeze the classical baseline.
3. Compare raw, L2-normalized, and norm-preserving classical controls.
4. Audit nonlinear models for leakage and state-family shortcuts.
5. Repeat the 36D-vs-63D ablation for nonlinear models.
6. Implement the six-qubit amplitude-encoded quantum classifier.
7. Compare classical and quantum models using identical transformed inputs.
8. Evaluate the direct-state circuit proposal separately.

## License

MIT License.
