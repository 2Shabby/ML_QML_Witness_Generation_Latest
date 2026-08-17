"""
Tiny classical baseline experiment on the confound-resistant dataset.

Fixed dataset (n=4000, seed=2026 -- the audited dataset) -> 36D Pauli
features -> one stratified 80/20 split -> four untuned classical models:

  1. norm_only : logistic regression on the 36D feature norm (1 scalar)
  2. tree_d2   : depth-2 decision tree on raw 36D features (shallow
                 36D-derived baseline)
  3. linear_svm: LinearSVC on standardized 36D features
  4. mlp_64_32 : small MLP (64 -> 32, ReLU) on standardized 36D features

Reports overall and per-family TEST accuracy (family is diagnostic
metadata only, not a model input). Single seed, no tuning.

Usage:
    python -m scripts.run_36d_baseline_classifiers
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quantum_states.balanced_dataset import generate_balanced_distillability_dataset
from scripts.validate_dataset_confound import extract_36d_features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--out-dir", type=str, default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "baseline_36d"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    states, labels, metadata = generate_balanced_distillability_dataset(
        n_samples=args.n_samples, seed=args.seed)
    t1 = time.time()
    feats = extract_36d_features(states)
    t2 = time.time()

    norm36 = np.linalg.norm(feats, axis=1)
    families = np.array([m["family"] for m in metadata])

    # Single stratified 80/20 split on indices (keeps all arrays aligned).
    idx = np.arange(len(labels))
    idx_train, idx_test = train_test_split(
        idx, test_size=args.test_frac, random_state=0, stratify=labels)

    # Per-model training/evaluation representations.
    norm_sc = StandardScaler().fit(norm36[idx_train].reshape(-1, 1))
    feat_sc = StandardScaler().fit(feats[idx_train])

    models = {
        "norm_only": (LogisticRegression(max_iter=5000),
                      norm_sc.transform(norm36[idx_train].reshape(-1, 1)),
                      norm_sc.transform(norm36[idx_test].reshape(-1, 1))),
        "tree_d2": (DecisionTreeClassifier(max_depth=2, random_state=0),
                    feats[idx_train], feats[idx_test]),
        "linear_svm": (LinearSVC(max_iter=5000),
                       feat_sc.transform(feats[idx_train]),
                       feat_sc.transform(feats[idx_test])),
        "mlp_64_32": (MLPClassifier(hidden_layer_sizes=(64, 32),
                                    max_iter=1000, random_state=0),
                      feat_sc.transform(feats[idx_train]),
                      feat_sc.transform(feats[idx_test])),
    }

    y_train, y_test = labels[idx_train], labels[idx_test]
    fam_test = families[idx_test]
    fam_names = sorted(set(fam_test))

    results = {}
    for name, (model, X_tr, X_te) in models.items():
        model.fit(X_tr, y_train)
        pred = model.predict(X_te)
        per_fam = {}
        for f in fam_names:
            sel = fam_test == f
            per_fam[f] = {
                "n": int(sel.sum()),
                "accuracy": round(float(accuracy_score(y_test[sel], pred[sel])), 4),
            }
        results[name] = {
            "test_accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "per_family": per_fam,
        }

    report = {
        "n_samples": int(args.n_samples),
        "seed": args.seed,
        "test_frac": args.test_frac,
        "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)),
        "test_positive_fraction": round(float(y_test.mean()), 4),
        "runtime_s": {"generation": round(t1 - t0, 1),
                      "features": round(t2 - t1, 1)},
        "models": results,
        "note": ("single-seed, untuned classical baselines; family is "
                 "diagnostic metadata, not a model input"),
    }
    out_path = os.path.join(args.out_dir, "baseline_report.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"\n36D baseline classifiers (n={args.n_samples}, seed={args.seed}, "
          f"test_frac={args.test_frac})")
    print(f"{'model':12s}" + "".join(f"{f:>16s}" for f in fam_names)
          + f"{'overall':>12s}")
    for n in models:
        row = f"{n:12s}"
        for f in fam_names:
            row += f"{results[n]['per_family'][f]['accuracy']:>16.3f}"
        row += f"{results[n]['test_accuracy']:>12.3f}"
        print(row)
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
