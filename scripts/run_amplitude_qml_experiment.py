#!/usr/bin/env python
"""Train the six-qubit amplitude-encoded classifier on restricted features."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extraction.pauli_features import (
    create_sparse_measurement_set,
    extract_features_batch,
)
from src.ml_models.amplitude_qml import AmplitudeQMLClassifierLearner
from src.quantum_states.state_generation import generate_distillability_dataset
from src.utils import convert_to_json_serializable


def run_experiment(n_samples: int, seed: int, n_epochs: int) -> dict:
    """Generate data, train the QML classifier, and return run metadata."""
    states, labels = generate_distillability_dataset(n_samples=n_samples, seed=seed)
    basis = create_sparse_measurement_set(3, "two_body")
    features = extract_features_batch(states, basis, verbose=False)
    learner = AmplitudeQMLClassifierLearner(
        n_epochs=n_epochs,
        random_state=seed,
    )
    metrics = learner.train(features, np.asarray(labels), verbose=True)
    return {
        "model": "six_qubit_amplitude_qml",
        "n_samples": n_samples,
        "seed": seed,
        "n_epochs": n_epochs,
        "metrics": metrics,
        "split_indices": learner.split_indices,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = convert_to_json_serializable(
        run_experiment(args.n_samples, args.seed, args.n_epochs)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
