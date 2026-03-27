#!/usr/bin/env python
"""
Run MLP classifier experiments.

Trains the MLP discriminator on 36D Pauli features and evaluates
classification performance. Includes multi-seed evaluation and
per-family analysis.

This corresponds to manuscript Section 5.1 (MLP results in Table 1).
"""

import sys
import os
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.quantum_states.state_generation import generate_distillability_dataset
from src.feature_extraction.pauli_features import (
    create_sparse_measurement_set,
    extract_features_batch,
)
from src.config import DEFAULT_SEEDS

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

try:
    from src.ml_models.mlp_classifier import MLPClassifierLearner
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def run_mlp_basic(
    n_samples: int = 5000,
    seed: int = 42,
):
    """Run basic MLP training and evaluation."""
    logger.info(f"\n{'='*60}")
    logger.info(f"MLP Basic Experiment (n_samples={n_samples}, seed={seed})")
    logger.info(f"{'='*60}")

    states, labels = generate_distillability_dataset(n_samples=n_samples, seed=seed)
    labels = np.array(labels)

    basis = create_sparse_measurement_set(3, 'two_body')
    features = extract_features_batch(states, basis, verbose=False)

    logger.info(
        f"Dataset: {features.shape[0]} samples, {features.shape[1]}D features, "
        f"distillable: {labels.sum()}/{len(labels)}"
    )

    learner = MLPClassifierLearner(
        n_features=36,
        hidden_dims=[128, 64, 32],
        dropout=0.3,
        learning_rate=1e-3,
        batch_size=64,
        n_epochs=100,
        patience=15,
        random_state=seed,
    )

    metrics = learner.train(features, labels, test_size=0.2, verbose=True)

    logger.info(f"\nMLP Results:")
    logger.info(f"  Test Accuracy:  {metrics['test_accuracy']:.4f}")
    logger.info(f"  Test Precision: {metrics['test_precision']:.4f}")
    logger.info(f"  Test Recall:    {metrics['test_recall']:.4f}")
    logger.info(f"  Test F1:        {metrics['test_f1']:.4f}")
    logger.info(f"  Parameters:     {metrics['n_parameters']}")

    return metrics, learner


def run_mlp_per_family(
    n_samples: int = 5000,
    seed: int = 42,
):
    """Analyze MLP performance per state family."""
    logger.info(f"\n{'='*60}")
    logger.info("MLP Per-Family Analysis")
    logger.info(f"{'='*60}")

    states, labels = generate_distillability_dataset(n_samples=n_samples, seed=seed)
    labels = np.array(labels)

    basis = create_sparse_measurement_set(3, 'two_body')
    features = extract_features_batch(states, basis, verbose=False)

    # Train on full dataset
    learner = MLPClassifierLearner(
        n_features=36, random_state=seed,
    )
    learner.train(features, labels, verbose=False)

    # Per-family evaluation
    n_per_family = n_samples // 5
    families = ['GHZ+noise', 'W+noise', 'Cluster+noise', 'Random mixed', 'Product']

    family_results = {}
    for i, family in enumerate(families):
        start = i * n_per_family
        end = start + n_per_family
        X_fam = features[start:end]
        y_fam = labels[start:end]

        preds = learner.predict(X_fam)
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        acc = accuracy_score(y_fam, preds)
        prec = precision_score(y_fam, preds, zero_division=0)
        rec = recall_score(y_fam, preds, zero_division=0)

        family_results[family] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'n_distillable': int(y_fam.sum()),
            'n_total': len(y_fam),
        }

        logger.info(
            f"  {family:<20} acc={acc:.4f}  prec={prec:.4f}  "
            f"rec={rec:.4f}  (dist: {y_fam.sum()}/{len(y_fam)})"
        )

    return family_results


def run_mlp_multi_seed(
    n_samples: int = 5000,
    seeds: list = None,
):
    """Run MLP experiments across multiple seeds."""
    if seeds is None:
        seeds = DEFAULT_SEEDS

    logger.info(f"\n{'='*60}")
    logger.info(f"MLP Multi-Seed Experiment ({len(seeds)} seeds)")
    logger.info(f"{'='*60}")

    all_metrics = []
    for seed in seeds:
        logger.info(f"\n--- Seed {seed} ---")
        states, labels = generate_distillability_dataset(
            n_samples=n_samples, seed=seed
        )
        labels = np.array(labels)

        basis = create_sparse_measurement_set(3, 'two_body')
        features = extract_features_batch(states, basis, verbose=False)

        learner = MLPClassifierLearner(
            n_features=36, random_state=seed,
        )
        metrics = learner.train(features, labels, verbose=False)
        all_metrics.append(metrics)
        logger.info(f"  Accuracy: {metrics['test_accuracy']:.4f}")

    # Aggregate
    accs = [m['test_accuracy'] for m in all_metrics]
    precs = [m['test_precision'] for m in all_metrics]
    recs = [m['test_recall'] for m in all_metrics]
    f1s = [m['test_f1'] for m in all_metrics]

    logger.info(f"\n{'='*60}")
    logger.info("AGGREGATE MLP RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy:  {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    logger.info(f"Precision: {np.mean(precs):.4f} +/- {np.std(precs):.4f}")
    logger.info(f"Recall:    {np.mean(recs):.4f} +/- {np.std(recs):.4f}")
    logger.info(f"F1:        {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

    return {
        'per_seed': {str(s): m for s, m in zip(seeds, all_metrics)},
        'aggregate': {
            'accuracy_mean': np.mean(accs),
            'accuracy_std': np.std(accs),
            'precision_mean': np.mean(precs),
            'recall_mean': np.mean(recs),
            'f1_mean': np.mean(f1s),
        }
    }


def run_all_mlp_experiments(output_dir: str = None):
    """Run the complete MLP experiment suite."""
    if not TORCH_AVAILABLE:
        logger.error("PyTorch is required for MLP experiments")
        return

    if output_dir is None:
        output_dir = str(project_root / 'results')
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    # 1. Basic
    metrics, _ = run_mlp_basic()
    all_results['basic'] = metrics

    # 2. Per-family
    family_results = run_mlp_per_family()
    all_results['per_family'] = family_results

    # 3. Multi-seed
    multi_seed = run_mlp_multi_seed()
    all_results['multi_seed'] = multi_seed

    # Save
    output_path = os.path.join(output_dir, 'mlp_experiments.json')
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': all_results,
        }, f, indent=2, default=str)
    logger.info(f"\nAll results saved to {output_path}")

    return all_results


if __name__ == '__main__':
    run_all_mlp_experiments()
