"""
Full clean-dataset experiment run on the frozen confound-resistant generator.

Single, internally consistent result set:

- generator: ``generate_balanced_distillability_dataset`` (frozen, as committed)
- dataset size: 4000 states per seed (generator default)
- seeds: 2026-2030
- ONE deterministic split helper for every seed (stratified, split seed 0):
    64% train / 16% validation / 20% untouched test
  Every model is trained on the same 64% training indices.  Models without
  early stopping simply ignore the validation set; torch models use the
  16% validation set for early stopping and are scored only on the
  untouched 20% test set.

Experiments (all architectures/hyperparameters are the existing
implementations/config defaults; nothing is tuned here):

  sanity   family balance, family-only, 36D-norm-only, purity diagnostic,
           shallow depth-2 tree on 36D
  36d      linear SVM, existing MLP, standard and hybrid transformer
  ablation 36D vs 63D for linear SVM and MLP (seed-level paired differences)
  qml_36d  amplitude QML vs matched L2-normalized 36D MLP
  qml_full direct-state QML vs 63D MLP (full-state-information classical
           comparator; NOT to be read against the 36D models)

Emits one consolidated JSON artifact:
  results/clean_dataset_experiments/clean_dataset_results.json

Usage:
    python -m scripts.run_clean_dataset_experiments
    python -m scripts.run_clean_dataset_experiments --smoke   # tiny 1-seed check
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import qiskit
import pennylane as qml
from qiskit.quantum_info import PauliList

from src.feature_extraction.pauli_features import (
    create_sparse_measurement_set,
    get_pauli_basis,
)
from src.feature_extraction.preprocessing import create_amplitude_encoding_controls
from src.ml_models.amplitude_qml import AmplitudeQMLClassifierLearner
from src.ml_models.direct_state_qml import DirectStateQMLClassifierLearner
from src.ml_models.mlp_classifier import MLPClassifierLearner
from src.ml_models.transformer_witness import TransformerWitnessLearner
from src.quantum_states.balanced_dataset import (
    generate_balanced_distillability_dataset,
)


# ---------------------------------------------------------------------------
# Frozen protocol
# ---------------------------------------------------------------------------

SPLIT_SEED = 0
TEST_FRAC = 0.20
VAL_OF_TRAIN_FRAC = 0.20  # 0.8 * 0.20 = 16% of total validation, 64% train


def make_splits(labels: np.ndarray) -> dict:
    """Single deterministic stratified 64/16/20 split helper (split seed 0)."""
    labels = np.asarray(labels)
    idx = np.arange(len(labels))
    idx_train, idx_test = train_test_split(
        idx, test_size=TEST_FRAC, random_state=SPLIT_SEED, stratify=labels
    )
    idx_train, idx_val = train_test_split(
        idx_train,
        test_size=VAL_OF_TRAIN_FRAC,
        random_state=SPLIT_SEED,
        stratify=labels[idx_train],
    )
    return {"train": idx_train, "val": idx_val, "test": idx_test}


def pauli_features_batch(states, pauli_basis: PauliList) -> np.ndarray:
    """Vectorized Tr(rho P_k) for every state and basis element."""
    M = np.asarray(pauli_basis.to_matrix(), dtype=np.complex128)  # (k, d, d)
    R = np.stack([np.asarray(s, dtype=np.complex128) for s in states])
    return np.einsum("kij,bji->bk", M, R).real


def dataset_hash(states, labels) -> str:
    digest = hashlib.sha256()
    for state in states:
        digest.update(np.asarray(state, dtype=np.complex128).tobytes())
    digest.update(np.asarray(labels, dtype=np.int8).tobytes())
    return digest.hexdigest()


def source_hash() -> str:
    digest = hashlib.sha256()
    paths = sorted(
        path
        for root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts")
        for path in root.rglob("*.py")
    )
    for path in paths:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def provenance() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True,
        capture_output=True, check=True,
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
        "qiskit": qiskit.__version__,
        "pennylane": qml.__version__,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def full_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def per_family_accuracy(y_true, y_pred, families) -> dict:
    out = {}
    for family in sorted(set(families)):
        sel = families == family
        out[family] = {
            "n": int(sel.sum()),
            "accuracy": float(accuracy_score(y_true[sel], y_pred[sel])),
        }
    return out


# ---------------------------------------------------------------------------
# Per-seed experiment
# ---------------------------------------------------------------------------

def run_seed(n_samples: int, seed: int) -> dict:
    t0 = time.time()
    states, labels, metadata = generate_balanced_distillability_dataset(
        n_samples=n_samples, seed=seed
    )
    t_gen = time.time() - t0

    basis_36 = create_sparse_measurement_set(3, strategy="two_body")
    basis_63 = get_pauli_basis(3, include_identity=False)
    X36 = pauli_features_batch(states, basis_36)
    X63 = pauli_features_batch(states, basis_63)
    t_feat = time.time() - t0

    rho = np.asarray([np.asarray(s) for s in states], dtype=np.complex128)
    controls = create_amplitude_encoding_controls(X36)
    X_l2 = controls["l2_normalized"]

    families = np.asarray([m["family"] for m in metadata])
    purity = np.asarray([m["purity"] for m in metadata])
    near = np.asarray([m["sample_mode"] for m in metadata]) == "near"

    split = make_splits(labels)
    tr, va, te = split["train"], split["val"], split["test"]
    y = labels

    print(f"[seed {seed}] generated {n_samples} states in {t_gen:.1f}s, "
          f"features in {t_feat:.1f}s; "
          f"train/val/test = {len(tr)}/{len(va)}/{len(te)}", flush=True)

    # ------------------------------------------------------------------
    # 1. Sanity / shortcut controls (validation set unused)
    # ------------------------------------------------------------------
    fam_counts = {}
    for fam in sorted(set(families)):
        sel = families == fam
        fam_counts[fam] = {
            "n": int(sel.sum()),
            "positive": int(y[sel].sum()),
            "negative": int((1 - y[sel]).sum()),
            "positive_fraction": float(y[sel].mean()),
        }

    fam_only_pred = np.empty(len(y), dtype=int)
    for fam in sorted(set(families)):
        sel = families == fam
        fam_only_pred[sel] = int(y[sel].mean() >= 0.5)

    norm36 = np.linalg.norm(X36, axis=1)
    sanity = {
        "overall": {
            "n": int(len(y)),
            "positive": int(y.sum()),
            "negative": int((1 - y).sum()),
            "positive_fraction": float(y.mean()),
        },
        "per_family": fam_counts,
        "near_boundary_band": {
            "n": int(near.sum()),
            "positive_fraction": float(y[near].mean()) if near.any() else None,
        },
        "family_only_accuracy": float(accuracy_score(y[te], fam_only_pred[te])),
    }

    def scalar_logistic(x):
        scaler = StandardScaler().fit(x[tr].reshape(-1, 1))
        model = LogisticRegression(max_iter=5000)
        model.fit(scaler.transform(x[tr].reshape(-1, 1)), y[tr])
        return float(accuracy_score(y[te], model.predict(
            scaler.transform(x[te].reshape(-1, 1)))))

    tree = DecisionTreeClassifier(max_depth=2, random_state=0)
    tree.fit(X36[tr], y[tr])
    sanity["norm_only_accuracy"] = scalar_logistic(norm36)
    sanity["purity_only_accuracy_DIAGNOSTIC"] = scalar_logistic(purity)
    sanity["shallow_tree_36d_accuracy"] = float(
        accuracy_score(y[te], tree.predict(X36[te])))

    # ------------------------------------------------------------------
    # 2. Classical 36D models
    # ------------------------------------------------------------------
    models_36d = {}

    scaler36 = StandardScaler().fit(X36[tr])
    svm = LinearSVC(max_iter=5000)
    svm.fit(scaler36.transform(X36[tr]), y[tr])
    pred = svm.predict(scaler36.transform(X36[te]))
    models_36d["linear_svm"] = full_metrics(y[te], pred)
    models_36d["linear_svm"]["per_family"] = per_family_accuracy(y[te], pred, families[te])

    mlp36 = MLPClassifierLearner(n_features=36, random_state=seed)
    mlp36.fit(X36[tr], y[tr], X36[va], y[va], verbose=False)
    pred = mlp36.predict(X36[te])
    models_36d["mlp"] = full_metrics(y[te], pred)
    models_36d["mlp"]["per_family"] = per_family_accuracy(y[te], pred, families[te])
    models_36d["mlp"]["epochs_run"] = len(mlp36.training_history)

    for mode in ("classifier", "hybrid"):
        print(f"[seed {seed}] transformer mode={mode}", flush=True)
        t1 = time.time()
        trans = TransformerWitnessLearner(
            pauli_basis=basis_36, mode=mode, random_state=seed
        )
        trans.fit(X36[tr], y[tr], X36[va], y[va], verbose=False)
        pred = trans.predict(X36[te])
        models_36d[f"transformer_{mode}"] = full_metrics(y[te], pred)
        models_36d[f"transformer_{mode}"]["per_family"] = per_family_accuracy(
            y[te], pred, families[te])
        models_36d[f"transformer_{mode}"]["epochs_run"] = len(trans.training_history)
        print(f"[seed {seed}] transformer {mode} done in {time.time() - t1:.1f}s "
              f"({models_36d[f'transformer_{mode}']['epochs_run']} epochs)", flush=True)

    # ------------------------------------------------------------------
    # 3. 36D vs 63D ablation (linear SVM, MLP)
    # ------------------------------------------------------------------
    ablation = {}

    scaler63 = StandardScaler().fit(X63[tr])
    svm63 = LinearSVC(max_iter=5000)
    svm63.fit(scaler63.transform(X63[tr]), y[tr])
    acc63 = float(accuracy_score(y[te], svm63.predict(scaler63.transform(X63[te]))))
    ablation["linear_svm"] = {
        "acc_36d": models_36d["linear_svm"]["accuracy"],
        "acc_63d": acc63,
        "diff_63d_minus_36d": acc63 - models_36d["linear_svm"]["accuracy"],
    }

    mlp63 = MLPClassifierLearner(n_features=63, random_state=seed)
    mlp63.fit(X63[tr], y[tr], X63[va], y[va], verbose=False)
    acc63_mlp = float(accuracy_score(y[te], mlp63.predict(X63[te])))
    ablation["mlp"] = {
        "acc_36d": models_36d["mlp"]["accuracy"],
        "acc_63d": acc63_mlp,
        "diff_63d_minus_36d": acc63_mlp - models_36d["mlp"]["accuracy"],
    }

    # ------------------------------------------------------------------
    # 4. Amplitude QML vs matched L2-normalized 36D MLP
    # ------------------------------------------------------------------
    print(f"[seed {seed}] L2 MLP control", flush=True)
    mlp_l2 = MLPClassifierLearner(n_features=36, random_state=seed)
    mlp_l2.fit(X_l2[tr], y[tr], X_l2[va], y[va], verbose=False)
    acc_l2 = float(accuracy_score(y[te], mlp_l2.predict(X_l2[te])))

    print(f"[seed {seed}] amplitude QML", flush=True)
    t1 = time.time()
    amp = AmplitudeQMLClassifierLearner(random_state=seed)
    amp.fit(X_l2[tr], y[tr], X_l2[va], y[va], verbose=False)
    acc_amp = float(accuracy_score(y[te], amp.predict(X_l2[te])))
    print(f"[seed {seed}] amplitude QML done in {time.time() - t1:.1f}s", flush=True)

    qml_36d = {
        "mlp_l2_normalized": acc_l2,
        "amplitude_qml": acc_amp,
        "amplitude_beats_mlp_l2": bool(acc_amp > acc_l2),
    }

    # ------------------------------------------------------------------
    # 5. Direct-state QML vs 63D MLP (full-information classical comparator)
    # ------------------------------------------------------------------
    print(f"[seed {seed}] direct-state QML", flush=True)
    t1 = time.time()
    direct = DirectStateQMLClassifierLearner(random_state=seed)
    direct.fit(rho[tr], y[tr], rho[va], y[va], verbose=False)
    acc_direct = float(accuracy_score(y[te], direct.predict(rho[te])))
    print(f"[seed {seed}] direct-state QML done in {time.time() - t1:.1f}s", flush=True)

    qml_full = {
        "mlp_63d": acc63_mlp,
        "direct_state_qml": acc_direct,
        "diff_direct_minus_mlp63d": acc_direct - acc63_mlp,
        "note": ("63 non-identity Pauli expectation values fully determine a "
                 "trace-1 Hermitian 3-qubit state, so the 63D MLP is the "
                 "full-state-information classical comparator"),
    }

    return {
        "seed": seed,
        "n_samples": n_samples,
        "dataset_sha256": dataset_hash(states, labels),
        "runtime_s": {"generation": round(t_gen, 1), "features": round(t_feat, 1)},
        "sanity": sanity,
        "models_36d": models_36d,
        "ablation_36d_vs_63d": ablation,
        "qml_36d": qml_36d,
        "qml_full": qml_full,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _agg(values):
    arr = np.asarray(values, dtype=float)
    out = {"mean": float(arr.mean()),
           "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}
    if len(arr) > 1:
        half = stats.t.ppf(0.975, len(arr) - 1) * arr.std(ddof=1) / np.sqrt(len(arr))
        out["ci95"] = [float(arr.mean() - half), float(arr.mean() + half)]
    return out


def aggregate(per_seed: list) -> dict:
    agg = {}

    sanity_scalar = {
        "family_only_accuracy", "norm_only_accuracy",
        "purity_only_accuracy_DIAGNOSTIC", "shallow_tree_36d_accuracy",
    }
    agg["sanity"] = {
        key: _agg([r["sanity"][key] for r in per_seed]) | {
            "per_seed": [r["sanity"][key] for r in per_seed]}
        for key in sanity_scalar
    }

    model_names = list(per_seed[0]["models_36d"].keys())
    agg["models_36d"] = {}
    for name in model_names:
        agg["models_36d"][name] = {}
        for metric in ("accuracy", "balanced_accuracy", "precision", "recall", "f1"):
            agg["models_36d"][name][metric] = _agg(
                [r["models_36d"][name][metric] for r in per_seed])
            agg["models_36d"][name][metric]["per_seed"] = [
                r["models_36d"][name][metric] for r in per_seed]
        family_names = sorted(per_seed[0]["models_36d"][name]["per_family"].keys())
        agg["models_36d"][name]["per_family_accuracy"] = {
            fam: _agg([r["models_36d"][name]["per_family"][fam]["accuracy"]
                       for r in per_seed]) |
            {"per_seed": [r["models_36d"][name]["per_family"][fam]["accuracy"]
                           for r in per_seed]}
            for fam in family_names
        }

    for name in ("linear_svm", "mlp"):
        diffs = [r["ablation_36d_vs_63d"][name]["diff_63d_minus_36d"] for r in per_seed]
        pos = [r["ablation_36d_vs_63d"][name]["acc_36d"] for r in per_seed]
        neg = [r["ablation_36d_vs_63d"][name]["acc_63d"] for r in per_seed]
        t_stat, p_value = (None, None)
        if len(set(diffs)) > 1:
            t_stat, p_value = stats.ttest_rel(neg, pos)
        agg["ablation_36d_vs_63d"] = agg.get("ablation_36d_vs_63d", {})
        agg["ablation_36d_vs_63d"][name] = {
            "acc_36d": _agg(pos) | {"per_seed": pos},
            "acc_63d": _agg(neg) | {"per_seed": neg},
            "diff_63d_minus_36d": _agg(diffs) | {"per_seed": diffs},
            "seeds_where_63d_wins": int(sum(d > 0 for d in diffs)),
            "n_seeds": len(per_seed),
            "paired_t": t_stat,
            "paired_p": p_value,
            "note": ("n=5 seed-level pairs; descriptive only, preliminary "
                     "and not a significance claim"),
        }

    agg["qml_36d"] = {}
    for key in ("mlp_l2_normalized", "amplitude_qml"):
        agg["qml_36d"][key] = _agg([r["qml_36d"][key] for r in per_seed]) | {
            "per_seed": [r["qml_36d"][key] for r in per_seed]}
    agg["qml_36d"]["amplitude_beats_mlp_l2"] = {
        "seeds_where_amplitude_wins": int(sum(
            r["qml_36d"]["amplitude_beats_mlp_l2"] for r in per_seed)),
        "n_seeds": len(per_seed),
    }

    agg["qml_full"] = {}
    for key in ("mlp_63d", "direct_state_qml"):
        agg["qml_full"][key] = _agg([r["qml_full"][key] for r in per_seed]) | {
            "per_seed": [r["qml_full"][key] for r in per_seed]}
    diffs = [r["qml_full"]["diff_direct_minus_mlp63d"] for r in per_seed]
    agg["qml_full"]["diff_direct_minus_mlp63d"] = _agg(diffs) | {"per_seed": diffs}
    agg["qml_full"]["seeds_where_direct_wins"] = int(sum(d > 0 for d in diffs))
    return agg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[2026, 2027, 2028, 2029, 2030])
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT_ROOT / "results" / "clean_dataset_experiments")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny single-seed run to verify the pipeline end to end")
    args = parser.parse_args()

    if args.smoke:
        args.n_samples = 300
        args.seeds = [2026]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_seed = []
    for seed in args.seeds:
        per_seed.append(run_seed(args.n_samples, seed))

    result = {
        "schema_version": 1,
        "run_name": "clean_dataset_full_run",
        "provenance": provenance(),
        "protocol": {
            "generator": "generate_balanced_distillability_dataset (frozen, defaults)",
            "n_samples_per_seed": args.n_samples,
            "seeds": args.seeds,
            "split": ("single stratified split helper, split seed 0: "
                      "64% train / 16% validation / 20% untouched test; "
                      "same training indices for every model; torch models "
                      "early-stop on the 16% validation set and are scored "
                      "only on the untouched 20% test set; classical models "
                      "ignore the validation set"),
            "models": {
                "linear_svm": "sklearn LinearSVC(max_iter=5000) on StandardScaler features (scaler fit on train)",
                "mlp": "MLPClassifierLearner defaults: 36/63 -> 128 -> 64 -> 32 -> 2, LeakyReLU, batchnorm, dropout 0.3, Adam lr 1e-3, batch 64, max 100 epochs, patience 15",
                "transformer_classifier": "TransformerWitnessLearner mode=classifier, config defaults: d_model 16, heads 2, layers 1, d_ff 32, dropout 0.1, AdamW lr 1e-3 wd 1e-4, batch 64, max 100 epochs, patience 15",
                "transformer_hybrid": "same architecture, mode=hybrid",
                "amplitude_qml": "AmplitudeQMLClassifierLearner defaults: 36D L2-normalized input, 6 qubits, 2 StronglyEntanglingLayers, lr 1e-2, batch 16, max 50 epochs, patience 10",
                "direct_state_qml": "DirectStateQMLClassifierLearner defaults: 3-qubit density-matrix input, 2 layers, lr 1e-2, batch 16, max 50 epochs, patience 10",
                "sanity_scalar": "LogisticRegression(max_iter=5000) on a StandardScaler-scaled single scalar",
                "shallow_tree": "DecisionTreeClassifier(max_depth=2, random_state=0)",
            },
            "torch_random_state": "dataset seed",
        },
        "per_seed": per_seed,
        "aggregate": aggregate(per_seed),
    }

    out_path = args.out_dir / ("smoke.json" if args.smoke else "clean_dataset_results.json")
    out_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
