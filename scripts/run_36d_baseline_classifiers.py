"""
Tiny classical baseline experiment on the confound-resistant dataset.

For each dataset seed: fixed pipeline (n=4000 by default) -> 36D Pauli
features -> one stratified 80/20 split (split seed fixed at 0) -> four
untuned classical models (hyperparameters fixed across all seeds):

  1. norm_only : logistic regression on the 36D feature norm (1 scalar)
  2. tree_d2   : depth-2 decision tree on raw 36D features (shallow
                 36D-derived baseline)
  3. linear_svm: LinearSVC on standardized 36D features
  4. mlp_64_32 : small MLP (64 -> 32, ReLU) on standardized 36D features

Reports overall and per-family TEST accuracy (family is diagnostic
metadata only, not a model input). With multiple dataset seeds, prints
mean +/- std and a 95% CI (t-based) per model, plus the MLP per-family
breakdown. No tuning.

Usage:
    python -m scripts.run_36d_baseline_classifiers                    # seed 2026
    python -m scripts.run_36d_baseline_classifiers \
        --seeds 2026 2027 2028 2029 2030
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
from scipy import stats
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


def run_once(n_samples: int, seed: int, test_frac: float) -> dict:
    """One full pipeline run for a single dataset seed; returns the report."""
    t0 = time.time()
    states, labels, metadata = generate_balanced_distillability_dataset(
        n_samples=n_samples, seed=seed)
    t1 = time.time()
    feats = extract_36d_features(states)
    t2 = time.time()

    norm36 = np.linalg.norm(feats, axis=1)
    families = np.array([m["family"] for m in metadata])

    # Single stratified 80/20 split on indices (keeps all arrays aligned).
    idx = np.arange(len(labels))
    idx_train, idx_test = train_test_split(
        idx, test_size=test_frac, random_state=0, stratify=labels)

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

    return {
        "n_samples": int(n_samples),
        "seed": int(seed),
        "test_frac": test_frac,
        "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)),
        "test_positive_fraction": round(float(y_test.mean()), 4),
        "runtime_s": {"generation": round(t1 - t0, 1),
                      "features": round(t2 - t1, 1)},
        "models": results,
        "note": ("single-seed, untuned classical baselines; family is "
                 "diagnostic metadata, not a model input"),
    }


def _mean_std_ci(values: List[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    out = {"mean": round(float(arr.mean()), 4), "std": round(float(arr.std(ddof=1)), 4)
           if len(arr) > 1 else 0.0}
    if len(arr) > 1:
        half = stats.t.ppf(0.975, len(arr) - 1) * arr.std(ddof=1) / np.sqrt(len(arr))
        out["ci95"] = [round(float(arr.mean() - half), 4),
                       round(float(arr.mean() + half), 4)]
    return out


def summarize(per_seed: List[dict]) -> dict:
    """Aggregate per-model overall accuracy and MLP per-family accuracy."""
    model_names = list(per_seed[0]["models"])
    fam_names = sorted(per_seed[0]["models"]["mlp_64_32"]["per_family"].keys())

    overall = {}
    for m in model_names:
        vals = [r["models"][m]["test_accuracy"] for r in per_seed]
        agg = _mean_std_ci(vals)
        agg["per_seed"] = vals
        overall[m] = agg

    mlp_fam = {}
    for f in fam_names:
        vals = [r["models"]["mlp_64_32"]["per_family"][f]["accuracy"] for r in per_seed]
        agg = _mean_std_ci(vals)
        agg["per_seed"] = vals
        mlp_fam[f] = agg

    return {"overall_accuracy": overall, "mlp_per_family_accuracy": mlp_fam}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=4000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026])
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--out-dir", type=str, default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "baseline_36d"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    per_seed = [run_once(args.n_samples, seed, args.test_frac) for seed in args.seeds]
    summary = summarize(per_seed) if len(per_seed) > 1 else None

    report = per_seed[0] if len(per_seed) == 1 else {
        "n_samples": args.n_samples,
        "seeds": args.seeds,
        "test_frac": args.test_frac,
        "per_seed": per_seed,
        "summary": summary,
        "note": ("untuned classical baselines, fixed models/hyperparameters "
                 "across seeds; family is diagnostic metadata, not a "
                 "model input"),
    }

    out_path = os.path.join(
        args.out_dir,
        "baseline_report.json" if len(per_seed) == 1 else "baseline_report_multiseed.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)

    # Console output --------------------------------------------------------
    if summary is None:
        s = per_seed[0]
        fam_names = sorted(s["models"]["norm_only"]["per_family"])
        print(f"\n36D baseline classifiers (n={s['n_samples']}, seed={s['seed']}, "
              f"test_frac={s['test_frac']})")
        print(f"{'model':12s}" + "".join(f"{f:>16s}" for f in fam_names)
              + f"{'overall':>12s}")
        for n in s["models"]:
            row = f"{n:12s}"
            for f in fam_names:
                row += f"{s['models'][n]['per_family'][f]['accuracy']:>16.3f}"
            row += f"{s['models'][n]['test_accuracy']:>12.3f}"
            print(row)
    else:
        n_seeds = len(per_seed)
        print(f"\n36D baseline classifiers, {n_seeds} dataset seeds "
              f"{args.seeds} (n={args.n_samples} each, test_frac={args.test_frac})")
        print(f"\nOverall test accuracy (mean +/- std, 95% CI):")
        for m, agg in summary["overall_accuracy"].items():
            ci = f"  95% CI [{agg['ci95'][0]:.3f}, {agg['ci95'][1]:.3f}]" \
                if "ci95" in agg else ""
            print(f"  {m:12s} {agg['mean']:.3f} +/- {agg['std']:.3f}{ci}"
                  f"   (per-seed: {agg['per_seed']})")
        print(f"\nMLP per-family test accuracy (mean +/- std, 95% CI):")
        for f, agg in summary["mlp_per_family_accuracy"].items():
            ci = f"  95% CI [{agg['ci95'][0]:.3f}, {agg['ci95'][1]:.3f}]" \
                if "ci95" in agg else ""
            print(f"  {f:15s} {agg['mean']:.3f} +/- {agg['std']:.3f}{ci}"
                  f"   (per-seed: {agg['per_seed']})")

    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
