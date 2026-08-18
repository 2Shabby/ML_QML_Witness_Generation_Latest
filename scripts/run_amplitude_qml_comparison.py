"""
Amplitude-encoded QML vs classical controls on the frozen dataset.

Frozen inputs (do not modify):
  - generator: ``generate_balanced_distillability_dataset`` (as committed)
  - raw 36D MLP reference: the frozen classical baseline
    (``results/baseline_36d/baseline_report_multiseed.json``,
    ``mlp_64_32`` overall test accuracy per seed) -- read, never re-run.

For each dataset seed (2026-2030 by default):
  - 36D Pauli features
  - ONE split, exactly the classical baseline convention:
    stratified 80/20 on indices, ``train_test_split(..., random_state=0,
    stratify=labels)``
  - controls via ``create_amplitude_encoding_controls``

Models (existing architecture/hyperparameters, unchanged, no tuning):
  1. raw_36d_mlp : frozen reference value (sklearn MLP 64-32, read from
                   the committed multi-seed baseline report)
  2. mlp_l2      : ``MLPClassifierLearner(n_features=36, n_epochs=20,
                   random_state=<dataset seed>)`` on L2-normalized 36D
                   (the existing QML-pipeline MLP control)
  3. amplitude   : ``AmplitudeQMLClassifierLearner(n_epochs=20,
                   random_state=<dataset seed>)`` (6 qubits, 2 layers,
                   existing defaults) trained on the same L2-normalized
                   36D input

Reports per-seed and mean +/- std test accuracy, and whether amplitude
QML beats the L2-normalized MLP on every seed. Direct-state QML is NOT
run here.

Usage:
    python -m scripts.run_amplitude_qml_comparison
"""

import argparse
import json
import os
import sys
import time
from typing import List

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quantum_states.balanced_dataset import generate_balanced_distillability_dataset
from src.feature_extraction.preprocessing import create_amplitude_encoding_controls
from src.ml_models.mlp_classifier import MLPClassifierLearner
from src.ml_models.amplitude_qml import AmplitudeQMLClassifierLearner
from scripts.validate_dataset_confound import extract_36d_features


def run_seed(n_samples: int, seed: int, epochs: int, test_frac: float) -> dict:
    t0 = time.time()
    states, labels, metadata = generate_balanced_distillability_dataset(
        n_samples=n_samples, seed=seed)
    t1 = time.time()
    feats = extract_36d_features(states)
    controls = create_amplitude_encoding_controls(feats)
    t2 = time.time()

    # Same split convention as the classical baseline: stratified 80/20 on
    # indices, split random_state fixed at 0.
    idx = np.arange(len(labels))
    idx_train, idx_test = train_test_split(
        idx, test_size=test_frac, random_state=0, stratify=labels)
    y_train, y_test = labels[idx_train], labels[idx_test]

    X_l2_train = controls["l2_normalized"][idx_train]
    X_l2_test = controls["l2_normalized"][idx_test]

    print(f"[seed {seed}] L2-normalized MLP control", flush=True)
    mlp = MLPClassifierLearner(n_features=36, n_epochs=epochs, random_state=seed)
    mlp.fit(X_l2_train, y_train, X_l2_test, y_test, verbose=False)
    mlp_acc = float(accuracy_score(y_test, mlp.predict(X_l2_test)))

    print(f"[seed {seed}] amplitude-encoded QML", flush=True)
    amp = AmplitudeQMLClassifierLearner(n_epochs=epochs, random_state=seed)
    amp.fit(X_l2_train, y_train, X_l2_test, y_test, verbose=False)
    amp_acc = float(accuracy_score(y_test, amp.predict(X_l2_test)))

    return {
        "seed": seed,
        "test_accuracy": {
            "mlp_l2": round(mlp_acc, 4),
            "amplitude_qml": round(amp_acc, 4),
        },
        "amplitude_beats_mlp_l2": bool(amp_acc > mlp_acc),
        "runtime_s": {"generation": round(t1 - t0, 1),
                      "features": round(t2 - t1, 1)},
    }


def load_frozen_raw_reference(path: str, seeds: List[int]) -> dict:
    with open(path) as fh:
        report = json.load(fh)
    per_seed = {r["seed"]: r["models"]["mlp_64_32"]["test_accuracy"]
                for r in report["per_seed"]}
    missing = [s for s in seeds if s not in per_seed]
    if missing:
        raise ValueError(f"frozen baseline report missing seeds {missing}")
    return per_seed


def _mean_std_ci(values: List[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    out = {"mean": round(float(arr.mean()), 4),
           "std": round(float(arr.std(ddof=1)), 4) if len(arr) > 1 else 0.0}
    if len(arr) > 1:
        from scipy import stats as sps
        half = sps.t.ppf(0.975, len(arr) - 1) * arr.std(ddof=1) / np.sqrt(len(arr))
        out["ci95"] = [round(float(arr.mean() - half), 4),
                       round(float(arr.mean() + half), 4)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=4000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029, 2030])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--frozen-baseline", type=str, default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "baseline_36d", "baseline_report_multiseed.json"))
    ap.add_argument("--out-dir", type=str, default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "amplitude_qml_36d"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    raw_ref = load_frozen_raw_reference(args.frozen_baseline, args.seeds)

    per_seed = [run_seed(args.n_samples, s, args.epochs, args.test_frac)
                for s in args.seeds]
    for r in per_seed:
        r["test_accuracy"]["raw_36d_mlp"] = raw_ref[r["seed"]]

    names = ["raw_36d_mlp", "mlp_l2", "amplitude_qml"]
    summary = {n: _mean_std_ci([r["test_accuracy"][n] for r in per_seed])
               for n in names}
    n_wins = sum(r["amplitude_beats_mlp_l2"] for r in per_seed)
    summary["amplitude_vs_mlp_l2"] = {
        "seeds_where_amplitude_wins": n_wins,
        "n_seeds": len(per_seed),
        "consistent": n_wins == len(per_seed),
    }

    report = {
        "n_samples": args.n_samples,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "test_frac": args.test_frac,
        "split_convention": ("train_test_split on indices, "
                             f"test_size={args.test_frac}, random_state=0, "
                             "stratify=labels -- identical to the classical "
                             "baseline script"),
        "models": {
            "raw_36d_mlp": "frozen reference, sklearn MLP 64-32, read from "
                           "results/baseline_36d/baseline_report_multiseed.json",
            "mlp_l2": "MLPClassifierLearner(n_features=36, n_epochs="
                      f"{args.epochs}, random_state=<dataset seed>) on "
                      "L2-normalized 36D (existing QML-pipeline control)",
            "amplitude_qml": "AmplitudeQMLClassifierLearner(n_epochs="
                             f"{args.epochs}, random_state=<dataset seed>) "
                             "on the same L2-normalized 36D input (6 qubits, "
                             "2 layers, existing defaults)",
        },
        "per_seed": per_seed,
        "summary": summary,
    }
    out_path = os.path.join(args.out_dir, "amplitude_qml_comparison.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"\nAmplitude QML vs controls ({len(per_seed)} seeds "
          f"{args.seeds}, n={args.n_samples} each, epochs={args.epochs})")
    print(f"{'seed':>8s}" + "".join(f"{n:>16s}" for n in names) + f"{'amp>mlp':>10s}")
    for r in per_seed:
        row = f"{r['seed']:>8d}"
        for n in names:
            row += f"{r['test_accuracy'][n]:>16.3f}"
        row += f"{'yes' if r['amplitude_beats_mlp_l2'] else 'no':>10s}"
        print(row)
    print("\nmean +/- std (95% CI):")
    for n in names:
        agg = summary[n]
        ci = (f"  95% CI [{agg['ci95'][0]:.3f}, {agg['ci95'][1]:.3f}]"
              if "ci95" in agg else "")
        print(f"  {n:15s} {agg['mean']:.3f} +/- {agg['std']:.3f}{ci}")
    v = summary["amplitude_vs_mlp_l2"]
    print(f"\nAmplitude QML beats L2-MLP on {v['seeds_where_amplitude_wins']}"
          f"/{v['n_seeds']} seeds (consistent: {v['consistent']})")
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
