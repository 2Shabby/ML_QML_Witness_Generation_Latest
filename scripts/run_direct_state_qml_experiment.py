#!/usr/bin/env python
"""Train the PennyLane classifier directly on three-qubit density matrices."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_models.direct_state_qml import DirectStateQMLClassifierLearner
from src.quantum_states.state_generation import generate_distillability_dataset
from src.utils import convert_to_json_serializable


def run_experiment(n_samples: int, seed: int, n_epochs: int) -> dict:
    """Generate labeled states and train the direct-state classifier."""
    states, labels = generate_distillability_dataset(n_samples=n_samples, seed=seed)
    density_matrices = np.asarray([np.asarray(state) for state in states])
    learner = DirectStateQMLClassifierLearner(
        n_epochs=n_epochs,
        random_state=seed,
    )
    metrics = learner.train(density_matrices, labels, verbose=True)
    return {
        "model": "three_qubit_direct_state_qml",
        "operational_setting": "full density matrix supplied directly to the circuit",
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
