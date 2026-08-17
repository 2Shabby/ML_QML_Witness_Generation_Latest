#!/usr/bin/env python
"""Single authoritative pipeline for manuscript-grade result generation."""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import sklearn
import torch
from qiskit.quantum_info import DensityMatrix
from scipy.stats import ttest_rel
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extraction.pauli_features import (
    create_sparse_measurement_set,
    extract_features_batch,
    get_pauli_basis,
)
from src.feature_extraction.preprocessing import create_amplitude_encoding_controls
from src.ml_models.amplitude_qml import AmplitudeQMLClassifierLearner
from src.ml_models.direct_state_qml import DirectStateQMLClassifierLearner
from src.ml_models.mlp_classifier import MLPClassifierLearner
from src.quantum_states.state_generation import (
    generate_distillability_dataset,
    generate_entangled_state,
    generate_noisy_cluster_state,
)
from src.utils import convert_to_json_serializable, stratified_split_indices


def classifier_factories(seed: int) -> dict:
    """Return the fixed classical model set used throughout validation."""
    return {
        "linear_svm": lambda: SVC(kernel="linear", C=1.0, random_state=seed),
        "rbf_svm": lambda: SVC(kernel="rbf", C=1.0, random_state=seed),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=-1
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=seed
        ),
    }


def metrics(y_true, y_pred) -> dict:
    """Return manuscript-facing binary classification metrics."""
    balanced_accuracy = (
        balanced_accuracy_score(y_true, y_pred)
        if len(np.unique(y_true)) == 2
        else None
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }


def connected_features(features: np.ndarray, basis) -> np.ndarray:
    """Compute connected two-body correlators from the ordered 36D basis."""
    labels = [str(pauli) for pauli in basis]
    columns = {label: features[:, index] for index, label in enumerate(labels)}
    connected = []
    for label in labels:
        active = [index for index, symbol in enumerate(label) if symbol != "I"]
        if len(active) != 2:
            continue
        first, second = active
        first_label = "".join(
            symbol if index == first else "I" for index, symbol in enumerate(label)
        )
        second_label = "".join(
            symbol if index == second else "I" for index, symbol in enumerate(label)
        )
        connected.append(columns[label] - columns[first_label] * columns[second_label])
    return np.column_stack(connected)


def dataset_hash(states, labels) -> str:
    """Hash state bytes and labels to identify the exact generated dataset."""
    digest = hashlib.sha256()
    for state in states:
        digest.update(np.asarray(state, dtype=np.complex128).tobytes())
    digest.update(np.asarray(labels, dtype=np.int8).tobytes())
    return digest.hexdigest()


def source_hash() -> str:
    """Hash all Python sources so dirty-tree runs remain exactly identifiable."""
    digest = hashlib.sha256()
    paths = sorted(
        path for root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts")
        for path in root.rglob("*.py")
    )
    for path in paths:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def family_counts(labels, metadata) -> dict:
    """Count labels within every state family."""
    result = {}
    for family in sorted({item["family"] for item in metadata}):
        family_labels = labels[
            np.array([item["family"] == family for item in metadata])
        ]
        result[family] = {
            "n": len(family_labels),
            "negative": int(np.sum(family_labels == 0)),
            "positive": int(np.sum(family_labels == 1)),
        }
    return result


def generate_boundary_dataset(n_samples: int, seed: int):
    """Generate noisy entangled families spanning the NPT/PPT transition."""
    rng = np.random.default_rng(seed)
    states = []
    families = []
    generators = {
        "ghz": lambda noise, state_seed: generate_entangled_state(
            3, "ghz", noise_level=noise, seed=state_seed
        ),
        "w": lambda noise, state_seed: generate_entangled_state(
            3, "w", noise_level=noise, seed=state_seed
        ),
        "cluster": lambda noise, state_seed: generate_noisy_cluster_state(
            3, noise_level=noise, seed=state_seed
        ),
    }
    from src.quantum_states.state_generation import check_npt_any_bipartition

    for index in range(n_samples):
        family = tuple(generators)[index % 3]
        noise = rng.uniform(0.65, 0.95)
        states.append(generators[family](noise, seed + index))
        families.append(family)
    labels = np.asarray([int(check_npt_any_bipartition(state)) for state in states])
    return states, labels, families


def run_classical(n_samples: int, boundary_samples: int, seed: int) -> dict:
    """Run leakage, shift, ablation, boundary, and feature-control stages."""
    print(f"[classical] generating {n_samples} states", flush=True)
    states, labels, metadata = generate_distillability_dataset(
        n_samples=n_samples, seed=seed, return_metadata=True
    )
    basis_9 = create_sparse_measurement_set(3, "local")
    basis_36 = create_sparse_measurement_set(3, "two_body")
    basis_63 = get_pauli_basis(3, include_identity=False)
    print("[classical] extracting 9D, 36D, and 63D features", flush=True)
    X9 = extract_features_batch(states, basis_9, verbose=False)
    X36 = extract_features_batch(states, basis_36, verbose=False)
    X63 = extract_features_batch(states, basis_63, verbose=False)
    Xconnected = connected_features(X36, basis_36)
    split = stratified_split_indices(labels, random_state=seed)
    train, test = split["train"], split["test"]
    families = np.asarray([item["family"] for item in metadata])
    factories = classifier_factories(seed)

    duplicate_rows = len(X63) - len(np.unique(X63, axis=0))
    train_hashes = {hashlib.sha256(row.tobytes()).digest() for row in X63[train]}
    test_hashes = {hashlib.sha256(row.tobytes()).digest() for row in X63[test]}
    audit = {
        "family_label_counts": family_counts(labels, metadata),
        "duplicate_feature_rows": duplicate_rows,
        "train_test_duplicate_rows": len(train_hashes & test_hashes),
    }

    print("[classical] evaluating family-label shortcut", flush=True)
    family_names = sorted(set(families))
    family_one_hot = np.column_stack([families == family for family in family_names])
    shortcut = LogisticRegression(max_iter=1000, random_state=seed)
    shortcut.fit(family_one_hot[train], labels[train])
    audit["family_only_classifier"] = metrics(
        labels[test], shortcut.predict(family_one_hot[test])
    )

    random_split = {}
    fitted = {}
    for name, factory in factories.items():
        print(f"[classical] random split: {name}", flush=True)
        model = factory()
        model.fit(X36[train], labels[train])
        fitted[name] = model
        random_split[name] = metrics(labels[test], model.predict(X36[test]))

    held_out = {}
    for family in family_names:
        print(f"[classical] held out family: {family}", flush=True)
        family_test = np.flatnonzero(families == family)
        family_train = np.flatnonzero(families != family)
        held_out[family] = {}
        if len(np.unique(labels[family_train])) < 2:
            held_out[family]["status"] = "not_estimable_single_class_training_set"
            continue
        for name, factory in factories.items():
            model = factory()
            model.fit(X36[family_train], labels[family_train])
            held_out[family][name] = metrics(
                labels[family_test], model.predict(X36[family_test])
            )

    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X36, labels))
    ablation = {}
    for name, factory in factories.items():
        print(f"[classical] paired 36D/63D CV: {name}", flush=True)
        ablation[name] = {}
        fold_scores = {}
        for feature_name, feature_matrix in {"36d": X36, "63d": X63}.items():
            scores = []
            for fold_train, fold_test in folds:
                model = factory()
                model.fit(feature_matrix[fold_train], labels[fold_train])
                scores.append(accuracy_score(labels[fold_test], model.predict(feature_matrix[fold_test])))
            fold_scores[feature_name] = scores
        if np.allclose(fold_scores["63d"], fold_scores["36d"]):
            statistic, p_value = None, None
        else:
            statistic, p_value = ttest_rel(fold_scores["63d"], fold_scores["36d"])
        ablation[name] = {
            "36d_fold_accuracy": fold_scores["36d"],
            "63d_fold_accuracy": fold_scores["63d"],
            "36d_mean": np.mean(fold_scores["36d"]),
            "63d_mean": np.mean(fold_scores["63d"]),
            "mean_gap_63d_minus_36d": np.mean(fold_scores["63d"]) - np.mean(fold_scores["36d"]),
            "paired_t": statistic,
            "paired_p": p_value,
        }

    feature_controls = {}
    print("[classical] connected and sparse feature controls", flush=True)
    for feature_name, feature_matrix in {
        "local_9d": X9,
        "restricted_36d": X36,
        "connected_27d": Xconnected,
        "restricted_plus_connected_63d": np.column_stack((X36, Xconnected)),
        "full_63d": X63,
    }.items():
        model = SVC(kernel="linear", C=1.0, random_state=seed)
        model.fit(feature_matrix[train], labels[train])
        feature_controls[feature_name] = metrics(
            labels[test], model.predict(feature_matrix[test])
        )

    print(f"[classical] boundary generalization: {boundary_samples} states", flush=True)
    boundary_states, boundary_labels, boundary_families = generate_boundary_dataset(
        boundary_samples, seed + 50000
    )
    boundary_36 = extract_features_batch(boundary_states, basis_36, verbose=False)
    boundary = {
        "n_samples": boundary_samples,
        "negative": int(np.sum(boundary_labels == 0)),
        "positive": int(np.sum(boundary_labels == 1)),
        "families": {family: boundary_families.count(family) for family in set(boundary_families)},
        "models": {
            name: metrics(boundary_labels, model.predict(boundary_36))
            for name, model in fitted.items()
        },
    }

    return {
        "dataset": {
            "sha256": dataset_hash(states, labels),
            "n_samples": n_samples,
            "seed": seed,
            "label_counts": {"negative": int(np.sum(labels == 0)), "positive": int(np.sum(labels == 1))},
            "feature_basis_36d": [str(pauli) for pauli in basis_36],
        },
        "split_indices": split,
        "audit": audit,
        "random_split_36d": random_split,
        "family_held_out_36d": held_out,
        "feature_ablation_cv": ablation,
        "linear_feature_controls": feature_controls,
        "boundary_generalization": boundary,
    }


def run_qml(n_samples: int, epochs: int, seed: int) -> dict:
    """Run QML and normalization controls on one identical split."""
    print(f"[qml] generating {n_samples} states", flush=True)
    states, labels = generate_distillability_dataset(n_samples=n_samples, seed=seed)
    density_matrices = np.asarray([np.asarray(state) for state in states])
    basis = create_sparse_measurement_set(3, "two_body")
    features = extract_features_batch(states, basis, verbose=False)
    controls = create_amplitude_encoding_controls(features)
    split = stratified_split_indices(labels, random_state=seed)
    train, test = split["train"], split["test"]
    results = {}

    for control_name, control_features in controls.items():
        print(f"[qml] MLP control: {control_name}", flush=True)
        learner = MLPClassifierLearner(
            n_features=control_features.shape[1], n_epochs=epochs, random_state=seed
        )
        learner.fit(
            control_features[train], labels[train],
            control_features[test], labels[test], verbose=False
        )
        results[f"mlp_{control_name}"] = metrics(
            labels[test], learner.predict(control_features[test])
        )

    print("[qml] amplitude-encoded classifier", flush=True)
    amplitude = AmplitudeQMLClassifierLearner(n_epochs=epochs, random_state=seed)
    amplitude.fit(
        controls["l2_normalized"][train], labels[train],
        controls["l2_normalized"][test], labels[test], verbose=False
    )
    results["amplitude_qml"] = metrics(
        labels[test], amplitude.predict(controls["l2_normalized"][test])
    )

    print("[qml] direct-state classifier", flush=True)
    direct = DirectStateQMLClassifierLearner(n_epochs=epochs, random_state=seed)
    direct.fit(
        density_matrices[train], labels[train],
        density_matrices[test], labels[test], verbose=False
    )
    results["direct_state_qml"] = metrics(
        labels[test], direct.predict(density_matrices[test])
    )
    return {
        "dataset_sha256": dataset_hash(states, labels),
        "n_samples": n_samples,
        "seed": seed,
        "epochs": epochs,
        "split_indices": split,
        "models": results,
    }


def aggregate_qml_runs(runs: dict) -> dict:
    """Aggregate scalar QML metrics across independently generated seeds."""
    model_names = next(iter(runs.values()))["models"]
    aggregate = {}
    for model_name in model_names:
        aggregate[model_name] = {}
        for metric_name in ("accuracy", "balanced_accuracy", "precision", "recall", "f1"):
            values = [run["models"][model_name][metric_name] for run in runs.values()]
            aggregate[model_name][f"{metric_name}_mean"] = np.mean(values)
            aggregate[model_name][f"{metric_name}_std"] = np.std(values, ddof=1)
    return aggregate


def flatten_metrics(result: dict) -> list[dict]:
    """Flatten every metric dictionary into CSV rows without losing JSON detail."""
    rows = []

    def visit(path, value):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(path + [key], child)
        elif isinstance(value, (int, float, np.integer, np.floating)) and (
            path[-1] in {
                "accuracy", "balanced_accuracy", "precision", "recall", "f1",
                "36d_mean", "63d_mean", "mean_gap_63d_minus_36d", "paired_p"
            }
            or path[-1].endswith(("_mean", "_std"))
        ):
            rows.append({"path": ".".join(path[:-1]), "metric": path[-1], "value": float(value)})

    visit([], result)
    return rows


def provenance() -> dict:
    """Capture code and library provenance for the result artifact."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
        capture_output=True, check=True
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True,
        capture_output=True, check=True
    ).stdout)
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "python_source_sha256": source_hash(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "rocm": torch.version.hip,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("classical", "qml", "all"), default="all")
    parser.add_argument("--classical-samples", type=int, default=5000)
    parser.add_argument("--boundary-samples", type=int, default=1500)
    parser.add_argument("--qml-samples", type=int, default=1000)
    parser.add_argument("--qml-epochs", type=int, default=20)
    parser.add_argument("--qml-seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "manuscript")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "results.json"
    csv_path = args.output_dir / "metrics.csv"
    if args.stage != "all" and json_path.exists():
        result = json.loads(json_path.read_text())
    else:
        result = {"schema_version": 1}
    result["provenance"] = provenance()
    result["configuration"] = {
        "seed": args.seed,
        "classical_samples": args.classical_samples,
        "boundary_samples": args.boundary_samples,
        "qml_samples": args.qml_samples,
        "qml_epochs": args.qml_epochs,
        "qml_seeds": args.qml_seeds,
    }
    if args.stage in ("classical", "all"):
        result["classical"] = run_classical(args.classical_samples, args.boundary_samples, args.seed)
    if args.stage in ("qml", "all"):
        qml_runs = {
            str(seed): run_qml(args.qml_samples, args.qml_epochs, seed)
            for seed in args.qml_seeds
        }
        result["qml"] = {
            "runs": qml_runs,
            "aggregate": aggregate_qml_runs(qml_runs),
        }

    serializable = convert_to_json_serializable(result)
    json_path.write_text(json.dumps(serializable, indent=2, allow_nan=False) + "\n")
    rows = flatten_metrics(result)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "metric", "value"))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "metric_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
