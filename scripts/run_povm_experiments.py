#!/usr/bin/env python
"""
Run variational POVM (quantum pipeline) experiments.

Trains the variational POVM on density matrices directly and compares
classification performance and measurement efficiency against the
classical Pauli-feature pipeline (SVM, MLP, Transformer).

This corresponds to manuscript Section 5.2 (Quantum Pipeline Results).
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

# Conditional imports
try:
    import torch
    from src.ml_models.variational_povm import VariationalPOVMLearner
    from src.ml_models.svm_witness import SVMWitnessLearner
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def states_to_numpy(states) -> np.ndarray:
    """Convert list of DensityMatrix objects to numpy array."""
    return np.array([np.array(state) for state in states])


def run_povm_basic(
    n_samples: int = 5000,
    seed: int = 42,
    n_layers: int = 4,
    n_epochs: int = 200,
    patience: int = 20,
):
    """Run basic POVM training and evaluation."""
    logger.info(f"\n{'='*60}")
    logger.info(f"POVM Basic Experiment (n_samples={n_samples}, seed={seed})")
    logger.info(f"{'='*60}")

    # Generate dataset
    states, labels = generate_distillability_dataset(n_samples=n_samples, seed=seed)
    labels = np.array(labels)
    rho_array = states_to_numpy(states)

    logger.info(
        f"Dataset: {len(states)} states, "
        f"distillable: {labels.sum()}/{len(labels)}"
    )

    # Train POVM
    learner = VariationalPOVMLearner(
        n_qubits=3,
        n_layers=n_layers,
        n_epochs=n_epochs,
        patience=patience,
        batch_size=64,
        learning_rate=1e-3,
        random_state=seed,
    )

    metrics = learner.train(rho_array, labels, test_size=0.2, verbose=True)

    logger.info(f"\nPOVM Results:")
    logger.info(f"  Test Accuracy:  {metrics['test_accuracy']:.4f}")
    logger.info(f"  Test Precision: {metrics['test_precision']:.4f}")
    logger.info(f"  Test Recall:    {metrics['test_recall']:.4f}")
    logger.info(f"  Test F1:        {metrics['test_f1']:.4f}")
    logger.info(f"  Parameters:     {metrics['n_parameters']}")
    logger.info(f"  Epochs trained: {metrics['n_epochs_trained']}")
    logger.info(f"  Measurement settings: {metrics['measurement_settings']}")

    # Verify POVM validity
    verification = learner.verify_povm()
    logger.info(f"\nPOVM Verification:")
    logger.info(f"  Completeness error: {verification['completeness_error']:.2e}")
    logger.info(f"  Min eigenvalue:     {verification['min_eigenvalue']:.2e}")
    logger.info(f"  Valid:              {verification['is_valid']}")

    return metrics, verification, learner


def run_povm_vs_classical(
    n_samples: int = 5000,
    seed: int = 42,
    n_layers: int = 4,
):
    """Compare POVM pipeline against classical Pauli pipeline."""
    logger.info(f"\n{'='*60}")
    logger.info("POVM vs Classical Pipeline Comparison")
    logger.info(f"{'='*60}")

    # Generate dataset
    states, labels = generate_distillability_dataset(n_samples=n_samples, seed=seed)
    labels = np.array(labels)
    rho_array = states_to_numpy(states)

    # Extract Pauli features for classical pipeline
    basis = create_sparse_measurement_set(3, 'two_body')
    features = extract_features_batch(states, basis, verbose=False)

    results = {}

    # 1. POVM (quantum pipeline)
    logger.info("\n--- Variational POVM (1 measurement setting) ---")
    povm_learner = VariationalPOVMLearner(
        n_qubits=3, n_layers=n_layers, n_epochs=200,
        patience=20, random_state=seed,
    )
    povm_metrics = povm_learner.train(rho_array, labels, verbose=False)
    results['povm'] = {
        **povm_metrics,
        'measurement_settings': 1,
        'input_type': 'density_matrix',
    }
    logger.info(f"  Accuracy: {povm_metrics['test_accuracy']:.4f}")

    # 2. SVM (classical pipeline)
    logger.info("\n--- Linear SVM (12 measurement settings) ---")
    svm_learner = SVMWitnessLearner(
        pauli_basis=basis, C=1.0, kernel='linear', random_state=seed
    )
    svm_metrics = svm_learner.train(features, labels, verbose=False)
    results['svm'] = {
        **svm_metrics,
        'measurement_settings': 12,
        'input_type': 'pauli_features_36d',
    }
    logger.info(f"  Accuracy: {svm_metrics['test_accuracy']:.4f}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'Model':<20} {'Accuracy':<12} {'Meas. Settings':<15}")
    logger.info(f"{'-'*47}")
    for name, r in results.items():
        logger.info(
            f"{name:<20} {r['test_accuracy']:<12.4f} "
            f"{r['measurement_settings']:<15}"
        )

    return results


def run_povm_layer_sweep(
    n_samples: int = 5000,
    seed: int = 42,
    layer_counts: list = None,
):
    """Sweep over different numbers of POVM layers."""
    if layer_counts is None:
        layer_counts = [1, 2, 4, 6, 8]

    logger.info(f"\n{'='*60}")
    logger.info("POVM Layer Count Sweep")
    logger.info(f"{'='*60}")

    states, labels = generate_distillability_dataset(n_samples=n_samples, seed=seed)
    labels = np.array(labels)
    rho_array = states_to_numpy(states)

    results = {}
    for n_layers in layer_counts:
        logger.info(f"\n--- {n_layers} layers ---")
        learner = VariationalPOVMLearner(
            n_qubits=3, n_layers=n_layers, n_epochs=200,
            patience=20, random_state=seed,
        )
        metrics = learner.train(rho_array, labels, verbose=False)
        results[n_layers] = metrics
        n_params = metrics['n_parameters']
        logger.info(
            f"  Accuracy: {metrics['test_accuracy']:.4f}, "
            f"Params: {n_params}, "
            f"Epochs: {metrics['n_epochs_trained']}"
        )

    return results


def run_povm_multi_seed(
    n_samples: int = 5000,
    seeds: list = None,
    n_layers: int = 4,
):
    """Run POVM experiments across multiple seeds for statistical significance."""
    if seeds is None:
        seeds = DEFAULT_SEEDS

    logger.info(f"\n{'='*60}")
    logger.info(f"POVM Multi-Seed Experiment ({len(seeds)} seeds)")
    logger.info(f"{'='*60}")

    all_metrics = []
    for seed in seeds:
        logger.info(f"\n--- Seed {seed} ---")
        states, labels = generate_distillability_dataset(
            n_samples=n_samples, seed=seed
        )
        labels = np.array(labels)
        rho_array = states_to_numpy(states)

        learner = VariationalPOVMLearner(
            n_qubits=3, n_layers=n_layers, n_epochs=200,
            patience=20, random_state=seed,
        )
        metrics = learner.train(rho_array, labels, verbose=False)
        all_metrics.append(metrics)
        logger.info(f"  Accuracy: {metrics['test_accuracy']:.4f}")

    # Aggregate
    accs = [m['test_accuracy'] for m in all_metrics]
    precs = [m['test_precision'] for m in all_metrics]
    recs = [m['test_recall'] for m in all_metrics]
    f1s = [m['test_f1'] for m in all_metrics]

    logger.info(f"\n{'='*60}")
    logger.info("AGGREGATE POVM RESULTS")
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


def run_all_povm_experiments(output_dir: str = None):
    """Run the complete POVM experiment suite."""
    if not TORCH_AVAILABLE:
        logger.error("PyTorch is required for POVM experiments")
        return

    if output_dir is None:
        output_dir = str(project_root / 'results')
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    # 1. Basic POVM
    metrics, verification, _ = run_povm_basic()
    all_results['basic'] = {
        'metrics': metrics,
        'verification': verification,
    }

    # 2. POVM vs Classical
    comparison = run_povm_vs_classical()
    all_results['comparison'] = comparison

    # 3. Layer sweep
    layer_results = run_povm_layer_sweep()
    all_results['layer_sweep'] = {
        str(k): v for k, v in layer_results.items()
    }

    # 4. Multi-seed
    multi_seed = run_povm_multi_seed()
    all_results['multi_seed'] = multi_seed

    # Save
    output_path = os.path.join(output_dir, 'povm_experiments.json')
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': all_results,
        }, f, indent=2, default=str)
    logger.info(f"\nAll results saved to {output_path}")

    return all_results


if __name__ == '__main__':
    run_all_povm_experiments()
