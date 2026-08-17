#!/usr/bin/env python
"""Compare amplitude QML with classical normalization controls on one split."""

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
from src.feature_extraction.preprocessing import create_amplitude_encoding_controls
from src.ml_models.amplitude_qml import AmplitudeQMLClassifierLearner
from src.ml_models.mlp_classifier import MLPClassifierLearner
from src.quantum_states.state_generation import generate_distillability_dataset
from src.utils import convert_to_json_serializable, stratified_split_indices


def run_comparison(
    n_samples: int,
    seed: int,
    mlp_epochs: int,
    qml_epochs: int,
    test_size: float = 0.2,
) -> dict:
    """Train every model on the same generated data and split indices."""
    states, labels = generate_distillability_dataset(n_samples=n_samples, seed=seed)
    labels = np.asarray(labels)
    basis = create_sparse_measurement_set(3, "two_body")
    features = extract_features_batch(states, basis, verbose=False)
    controls = create_amplitude_encoding_controls(features)
    split = stratified_split_indices(labels, test_size=test_size, random_state=seed)
    train_indices = split["train"]
    test_indices = split["test"]

    model_results = {}
    for control_name, control_features in controls.items():
        learner = MLPClassifierLearner(
            n_features=control_features.shape[1],
            n_epochs=mlp_epochs,
            random_state=seed,
        )
        metrics = learner.fit(
            control_features[train_indices],
            labels[train_indices],
            control_features[test_indices],
            labels[test_indices],
            verbose=True,
        )
        model_results[f"mlp_{control_name}"] = {
            "input_dimension": control_features.shape[1],
            "metrics": metrics,
            "test_predictions": learner.predict(control_features[test_indices]),
        }

    qml = AmplitudeQMLClassifierLearner(
        n_epochs=qml_epochs,
        random_state=seed,
    )
    qml_metrics = qml.fit(
        controls["l2_normalized"][train_indices],
        labels[train_indices],
        controls["l2_normalized"][test_indices],
        labels[test_indices],
        verbose=True,
    )
    model_results["amplitude_qml_l2_normalized"] = {
        "input_dimension": 36,
        "metrics": qml_metrics,
        "test_predictions": qml.predict(
            controls["l2_normalized"][test_indices]
        ),
    }

    return {
        "experiment": "controlled_amplitude_qml_comparison",
        "n_samples": n_samples,
        "seed": seed,
        "test_size": test_size,
        "training": {"mlp_epochs": mlp_epochs, "qml_epochs": qml_epochs},
        "feature_basis": [str(pauli) for pauli in basis],
        "split_indices": split,
        "test_labels": labels[test_indices],
        "controls": {
            "raw": "original 36 Pauli features",
            "l2_normalized": "36 features divided by each sample's L2 norm",
            "l2_plus_norm": "L2-normalized features plus original norm as feature 37",
        },
        "models": model_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlp-epochs", type=int, default=100)
    parser.add_argument("--qml-epochs", type=int, default=50)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = convert_to_json_serializable(
        run_comparison(
            n_samples=args.n_samples,
            seed=args.seed,
            mlp_epochs=args.mlp_epochs,
            qml_epochs=args.qml_epochs,
            test_size=args.test_size,
        )
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
