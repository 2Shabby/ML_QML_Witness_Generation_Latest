#!/usr/bin/env python3
"""
Experimental Pipeline for 3-Qubit Distillability Hypothesis

This script implements the full experimental validation pipeline to answer:
Can 36D restricted (1+2 body Pauli) features reliably distinguish
distillable from non-distillable 3-qubit states?

Usage:
    python scripts/run_experiments.py --experiment ablation
    python scripts/run_experiments.py --experiment cross_validation
    python scripts/run_experiments.py --experiment per_family
    python scripts/run_experiments.py --experiment noise_robustness
    python scripts/run_experiments.py --experiment witness_coefficients
    python scripts/run_experiments.py --experiment all
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.quantum_states.state_generation import (
    generate_distillability_dataset,
    generate_entangled_state,
    generate_3qubit_product_state,
    generate_noisy_cluster_state,
    generate_random_density_matrix,
    check_npt_any_bipartition
)
from src.feature_extraction.pauli_features import (
    create_sparse_measurement_set,
    get_pauli_basis,
    extract_features_batch
)
from src.ml_models.svm_witness import SVMWitnessLearner

# Import centralized config and utilities
from src.config import (
    DEFAULT_N_SAMPLES,
    DEFAULT_NOISE_RANGE,
    DEFAULT_CV_FOLDS,
    DEFAULT_SEEDS,
    DEFAULT_LOG_FORMAT,
    RESULTS_DIR,
)
from src.utils import convert_to_json_serializable, setup_logging

# Set up logging using centralized utility
logging.basicConfig(level=logging.INFO, format=DEFAULT_LOG_FORMAT)
logger = logging.getLogger(__name__)


def save_results(results: Dict, experiment_name: str, results_dir: Path) -> str:
    """Save results to JSON file with timestamp."""
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = results_dir / filename

    # Make a copy and add metadata
    results_copy = convert_to_json_serializable(results)
    results_copy['metadata'] = {
        'experiment': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'n_samples': results_copy.get('n_samples', 'N/A')
    }

    with open(filepath, 'w') as f:
        json.dump(results_copy, f, indent=2)

    logger.info(f"Results saved to {filepath}")
    return str(filepath)


def run_ablation_study(
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_range: Tuple[float, float] = DEFAULT_NOISE_RANGE,
    n_folds: int = DEFAULT_CV_FOLDS,
    seed: int = 42,
    results_dir: Optional[Path] = None
) -> Dict:
    """
    Compare 36D restricted features vs 63D full features using 5-fold CV.

    This is the key ablation study from GOAL.md: does restricting to 1+2 body
    Paulis hurt performance compared to using all 63 Paulis?

    Args:
        n_samples: Number of samples to generate
        noise_range: Range of noise levels
        n_folds: Number of cross-validation folds
        seed: Random seed for reproducibility
        results_dir: Directory to save results

    Returns:
        Dictionary with comparison results and statistical analysis
    """
    logger.info("="*60)
    logger.info("ABLATION STUDY: 36D Restricted vs 63D Full Features")
    logger.info("="*60)

    np.random.seed(seed)

    # Generate dataset
    logger.info(f"\n[1/4] Generating {n_samples} samples...")
    states, labels = generate_distillability_dataset(
        n_samples=n_samples,
        noise_range=noise_range,
        seed=seed
    )
    labels = np.array(labels)

    logger.info(f"  Distillable: {np.sum(labels)} ({100*np.mean(labels):.1f}%)")
    logger.info(f"  Non-distillable: {len(labels) - np.sum(labels)} ({100*(1-np.mean(labels)):.1f}%)")

    # Create feature sets
    logger.info("\n[2/4] Extracting features...")
    basis_36d = create_sparse_measurement_set(3, 'two_body')  # 36D restricted
    basis_63d = get_pauli_basis(3, include_identity=False)     # 63D full

    features_36d = extract_features_batch(states, basis_36d, verbose=False)
    features_63d = extract_features_batch(states, basis_63d, verbose=False)

    logger.info(f"  36D features: {features_36d.shape}")
    logger.info(f"  63D features: {features_63d.shape}")

    # Cross-validation
    logger.info(f"\n[3/4] Running {n_folds}-fold cross-validation...")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    results_36d = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    results_63d = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for fold, (train_idx, test_idx) in enumerate(skf.split(features_36d, labels)):
        # 36D model
        svm_36d = SVC(kernel='linear', C=1.0, random_state=seed)
        svm_36d.fit(features_36d[train_idx], labels[train_idx])
        pred_36d = svm_36d.predict(features_36d[test_idx])

        results_36d['accuracy'].append(accuracy_score(labels[test_idx], pred_36d))
        results_36d['precision'].append(precision_score(labels[test_idx], pred_36d, zero_division=0))
        results_36d['recall'].append(recall_score(labels[test_idx], pred_36d, zero_division=0))
        results_36d['f1'].append(f1_score(labels[test_idx], pred_36d, zero_division=0))

        # 63D model
        svm_63d = SVC(kernel='linear', C=1.0, random_state=seed)
        svm_63d.fit(features_63d[train_idx], labels[train_idx])
        pred_63d = svm_63d.predict(features_63d[test_idx])

        results_63d['accuracy'].append(accuracy_score(labels[test_idx], pred_63d))
        results_63d['precision'].append(precision_score(labels[test_idx], pred_63d, zero_division=0))
        results_63d['recall'].append(recall_score(labels[test_idx], pred_63d, zero_division=0))
        results_63d['f1'].append(f1_score(labels[test_idx], pred_63d, zero_division=0))

        logger.info(f"  Fold {fold+1}: 36D acc={results_36d['accuracy'][-1]:.3f}, 63D acc={results_63d['accuracy'][-1]:.3f}")

    # Statistical comparison
    logger.info("\n[4/4] Statistical analysis...")

    acc_36d = np.array(results_36d['accuracy'])
    acc_63d = np.array(results_63d['accuracy'])

    t_stat, p_value = ttest_rel(acc_63d, acc_36d)

    results = {
        'n_samples': n_samples,
        'noise_range': list(noise_range),
        'n_folds': n_folds,
        'seed': seed,
        'restricted_36d': {
            'accuracy_mean': float(np.mean(results_36d['accuracy'])),
            'accuracy_std': float(np.std(results_36d['accuracy'])),
            'precision_mean': float(np.mean(results_36d['precision'])),
            'recall_mean': float(np.mean(results_36d['recall'])),
            'f1_mean': float(np.mean(results_36d['f1'])),
            'fold_accuracies': [float(x) for x in results_36d['accuracy']]
        },
        'full_63d': {
            'accuracy_mean': float(np.mean(results_63d['accuracy'])),
            'accuracy_std': float(np.std(results_63d['accuracy'])),
            'precision_mean': float(np.mean(results_63d['precision'])),
            'recall_mean': float(np.mean(results_63d['recall'])),
            'f1_mean': float(np.mean(results_63d['f1'])),
            'fold_accuracies': [float(x) for x in results_63d['accuracy']]
        },
        'statistical_comparison': {
            'accuracy_gap': float(np.mean(acc_63d) - np.mean(acc_36d)),
            'paired_ttest_t': float(t_stat),
            'paired_ttest_p': float(p_value),
            'significant_at_0.05': p_value < 0.05
        }
    }

    # Summary
    logger.info("\n" + "="*60)
    logger.info("ABLATION STUDY RESULTS")
    logger.info("="*60)
    logger.info(f"36D Restricted: {results['restricted_36d']['accuracy_mean']:.3f} +/- {results['restricted_36d']['accuracy_std']:.3f}")
    logger.info(f"63D Full:       {results['full_63d']['accuracy_mean']:.3f} +/- {results['full_63d']['accuracy_std']:.3f}")
    logger.info(f"Accuracy Gap:   {results['statistical_comparison']['accuracy_gap']:.3f}")
    logger.info(f"Paired t-test:  t={t_stat:.3f}, p={p_value:.4f}")

    if p_value < 0.05:
        logger.info("  => Significant difference (p < 0.05)")
    else:
        logger.info("  => No significant difference (p >= 0.05)")

    if results_dir:
        save_results(results, 'ablation_study', results_dir)

    return results


def run_cross_validation(
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_range: Tuple[float, float] = DEFAULT_NOISE_RANGE,
    n_folds: int = DEFAULT_CV_FOLDS,
    seeds: List[int] = DEFAULT_SEEDS,
    results_dir: Optional[Path] = None
) -> Dict:
    """
    Statistical validation with multiple random seeds.

    Runs cross-validation with different seeds to ensure results are stable
    and not artifacts of a particular random split.

    Args:
        n_samples: Number of samples per run
        noise_range: Range of noise levels
        n_folds: Number of CV folds
        seeds: List of random seeds to test
        results_dir: Directory to save results

    Returns:
        Dictionary with cross-seed statistics
    """
    logger.info("="*60)
    logger.info("CROSS-VALIDATION: Multi-seed Statistical Validation")
    logger.info("="*60)

    all_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'seed_details': []
    }

    for seed in seeds:
        logger.info(f"\n[Seed {seed}] Running experiment...")
        np.random.seed(seed)

        # Generate dataset
        states, labels = generate_distillability_dataset(
            n_samples=n_samples,
            noise_range=noise_range,
            seed=seed
        )
        labels = np.array(labels)

        # Extract 36D features
        basis_36d = create_sparse_measurement_set(3, 'two_body')
        features = extract_features_batch(states, basis_36d, verbose=False)

        # Cross-validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []
        fold_f1s = []

        for train_idx, test_idx in skf.split(features, labels):
            svm = SVC(kernel='linear', C=1.0, random_state=seed)
            svm.fit(features[train_idx], labels[train_idx])
            pred = svm.predict(features[test_idx])

            fold_accuracies.append(accuracy_score(labels[test_idx], pred))
            fold_precisions.append(precision_score(labels[test_idx], pred, zero_division=0))
            fold_recalls.append(recall_score(labels[test_idx], pred, zero_division=0))
            fold_f1s.append(f1_score(labels[test_idx], pred, zero_division=0))

        mean_acc = np.mean(fold_accuracies)
        all_results['accuracy'].append(mean_acc)
        all_results['precision'].append(np.mean(fold_precisions))
        all_results['recall'].append(np.mean(fold_recalls))
        all_results['f1'].append(np.mean(fold_f1s))
        all_results['seed_details'].append({
            'seed': seed,
            'accuracy': float(mean_acc),
            'fold_accuracies': [float(x) for x in fold_accuracies]
        })

        logger.info(f"  Mean accuracy: {mean_acc:.3f}")

    # Aggregate statistics
    results = {
        'n_samples': n_samples,
        'noise_range': list(noise_range),
        'n_folds': n_folds,
        'seeds': seeds,
        'aggregate_statistics': {
            'accuracy_mean': float(np.mean(all_results['accuracy'])),
            'accuracy_std': float(np.std(all_results['accuracy'])),
            'accuracy_min': float(np.min(all_results['accuracy'])),
            'accuracy_max': float(np.max(all_results['accuracy'])),
            'precision_mean': float(np.mean(all_results['precision'])),
            'recall_mean': float(np.mean(all_results['recall'])),
            'f1_mean': float(np.mean(all_results['f1']))
        },
        'per_seed_results': all_results['seed_details']
    }

    # Summary
    logger.info("\n" + "="*60)
    logger.info("CROSS-VALIDATION RESULTS")
    logger.info("="*60)
    logger.info(f"Seeds tested: {seeds}")
    logger.info(f"Mean accuracy: {results['aggregate_statistics']['accuracy_mean']:.3f} +/- {results['aggregate_statistics']['accuracy_std']:.3f}")
    logger.info(f"Range: [{results['aggregate_statistics']['accuracy_min']:.3f}, {results['aggregate_statistics']['accuracy_max']:.3f}]")

    if results_dir:
        save_results(results, 'cross_validation', results_dir)

    return results


def run_per_family_analysis(
    n_samples_per_family: int = 500,
    noise_range: Tuple[float, float] = DEFAULT_NOISE_RANGE,
    seed: int = 42,
    results_dir: Optional[Path] = None
) -> Dict:
    """
    Accuracy breakdown by state family: GHZ, W, cluster, random, product.

    Tests how well the classifier performs on each type of quantum state
    to identify where the restricted features work well and where they fail.

    Args:
        n_samples_per_family: Samples per state family
        noise_range: Range of noise levels
        seed: Random seed
        results_dir: Directory to save results

    Returns:
        Dictionary with per-family accuracy metrics
    """
    logger.info("="*60)
    logger.info("PER-FAMILY ANALYSIS: Accuracy by State Type")
    logger.info("="*60)

    np.random.seed(seed)

    # Generate training data (mixed families)
    logger.info("\n[1/3] Generating training dataset...")
    train_states, train_labels = generate_distillability_dataset(
        n_samples=n_samples_per_family * 5,
        noise_range=noise_range,
        seed=seed
    )
    train_labels = np.array(train_labels)

    # Extract features and train model
    logger.info("\n[2/3] Training classifier...")
    basis = create_sparse_measurement_set(3, 'two_body')
    train_features = extract_features_batch(train_states, basis, verbose=False)

    svm = SVC(kernel='linear', C=1.0, random_state=seed)
    svm.fit(train_features, train_labels)

    # Generate test data per family
    logger.info("\n[3/3] Evaluating per family...")

    families = {
        'ghz': {'states': [], 'labels': []},
        'w': {'states': [], 'labels': []},
        'cluster': {'states': [], 'labels': []},
        'random': {'states': [], 'labels': []},
        'product': {'states': [], 'labels': []}
    }

    # GHZ states
    for i in range(n_samples_per_family):
        noise = np.random.uniform(*noise_range)
        state = generate_entangled_state(3, 'ghz', noise_level=noise, seed=seed+i)
        families['ghz']['states'].append(state)
        families['ghz']['labels'].append(1 if check_npt_any_bipartition(state) else 0)

    # W states
    for i in range(n_samples_per_family):
        noise = np.random.uniform(*noise_range)
        state = generate_entangled_state(3, 'w', noise_level=noise, seed=seed+1000+i)
        families['w']['states'].append(state)
        families['w']['labels'].append(1 if check_npt_any_bipartition(state) else 0)

    # Cluster states
    for i in range(n_samples_per_family):
        noise = np.random.uniform(*noise_range)
        state = generate_noisy_cluster_state(3, noise_level=noise, seed=seed+2000+i)
        families['cluster']['states'].append(state)
        families['cluster']['labels'].append(1 if check_npt_any_bipartition(state) else 0)

    # Random mixed states
    for i in range(n_samples_per_family):
        state = generate_random_density_matrix(3, seed=seed+3000+i)
        families['random']['states'].append(state)
        families['random']['labels'].append(1 if check_npt_any_bipartition(state) else 0)

    # Product states (always non-distillable)
    for i in range(n_samples_per_family):
        state = generate_3qubit_product_state(seed=seed+4000+i)
        families['product']['states'].append(state)
        families['product']['labels'].append(0)

    # Evaluate each family
    results = {
        'n_samples_per_family': n_samples_per_family,
        'noise_range': list(noise_range),
        'seed': seed,
        'per_family': {}
    }

    for family_name, family_data in families.items():
        features = extract_features_batch(family_data['states'], basis, verbose=False)
        labels = np.array(family_data['labels'])
        predictions = svm.predict(features)

        accuracy = accuracy_score(labels, predictions)

        # Handle edge cases where all labels are same class
        if len(np.unique(labels)) == 1:
            precision = 1.0 if labels[0] == 0 else precision_score(labels, predictions, zero_division=0)
            recall = 1.0 if labels[0] == 0 else recall_score(labels, predictions, zero_division=0)
        else:
            precision = precision_score(labels, predictions, zero_division=0)
            recall = recall_score(labels, predictions, zero_division=0)

        n_distillable = int(np.sum(labels))
        n_non_distillable = len(labels) - n_distillable

        results['per_family'][family_name] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'n_distillable': n_distillable,
            'n_non_distillable': n_non_distillable,
            'distillable_fraction': float(n_distillable / len(labels))
        }

        logger.info(f"  {family_name.upper():8s}: acc={accuracy:.3f}, prec={precision:.3f}, rec={recall:.3f} "
                   f"(dist={n_distillable}, non-dist={n_non_distillable})")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("PER-FAMILY RESULTS SUMMARY")
    logger.info("="*60)

    best_family = max(results['per_family'].keys(), key=lambda k: results['per_family'][k]['accuracy'])
    worst_family = min(results['per_family'].keys(), key=lambda k: results['per_family'][k]['accuracy'])

    logger.info(f"Best performing: {best_family} ({results['per_family'][best_family]['accuracy']:.3f})")
    logger.info(f"Worst performing: {worst_family} ({results['per_family'][worst_family]['accuracy']:.3f})")

    if results_dir:
        save_results(results, 'per_family_analysis', results_dir)

    return results


def run_noise_robustness(
    n_samples_per_level: int = 300,
    noise_levels: List[float] = None,
    seed: int = 42,
    results_dir: Optional[Path] = None
) -> Dict:
    """
    Accuracy vs noise level curves.

    Tests how classifier performance degrades as noise increases,
    identifying the noise threshold where classification becomes unreliable.

    Args:
        n_samples_per_level: Samples per noise level
        noise_levels: List of noise levels to test
        seed: Random seed
        results_dir: Directory to save results

    Returns:
        Dictionary with accuracy vs noise data
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    logger.info("="*60)
    logger.info("NOISE ROBUSTNESS: Accuracy vs Noise Level")
    logger.info("="*60)

    np.random.seed(seed)

    results = {
        'n_samples_per_level': n_samples_per_level,
        'noise_levels': noise_levels,
        'seed': seed,
        'per_noise_level': {}
    }

    for noise in noise_levels:
        logger.info(f"\n[Noise={noise:.2f}] Generating and evaluating...")

        # Generate states at this noise level (narrow range)
        noise_range = (max(0, noise - 0.05), min(1, noise + 0.05))

        states, labels = generate_distillability_dataset(
            n_samples=n_samples_per_level,
            noise_range=noise_range,
            seed=seed + int(noise * 1000)
        )
        labels = np.array(labels)

        # Extract features
        basis = create_sparse_measurement_set(3, 'two_body')
        features = extract_features_batch(states, basis, verbose=False)

        # Train and evaluate with train/test split
        n_train = int(0.8 * len(states))
        indices = np.random.permutation(len(states))
        train_idx, test_idx = indices[:n_train], indices[n_train:]

        svm = SVC(kernel='linear', C=1.0, random_state=seed)
        svm.fit(features[train_idx], labels[train_idx])
        pred = svm.predict(features[test_idx])

        accuracy = accuracy_score(labels[test_idx], pred)

        n_distillable = int(np.sum(labels))
        distillable_fraction = n_distillable / len(labels)

        results['per_noise_level'][str(noise)] = {
            'accuracy': float(accuracy),
            'distillable_fraction': float(distillable_fraction),
            'n_distillable': n_distillable,
            'n_total': len(labels)
        }

        logger.info(f"  Accuracy: {accuracy:.3f}, Distillable fraction: {distillable_fraction:.3f}")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("NOISE ROBUSTNESS SUMMARY")
    logger.info("="*60)

    for noise in noise_levels:
        acc = results['per_noise_level'][str(noise)]['accuracy']
        dist_frac = results['per_noise_level'][str(noise)]['distillable_fraction']
        logger.info(f"  Noise {noise:.2f}: acc={acc:.3f}, dist_frac={dist_frac:.3f}")

    if results_dir:
        save_results(results, 'noise_robustness', results_dir)

    return results


def analyze_witness_coefficients(
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_range: Tuple[float, float] = DEFAULT_NOISE_RANGE,
    seed: int = 42,
    results_dir: Optional[Path] = None
) -> Dict:
    """
    Rank Pauli terms by importance in the learned witness.

    Analyzes which Pauli observables contribute most to the classification,
    providing insight into what physical correlations distinguish distillable states.

    Args:
        n_samples: Number of training samples
        noise_range: Range of noise levels
        seed: Random seed
        results_dir: Directory to save results

    Returns:
        Dictionary with ranked Pauli terms and their coefficients
    """
    logger.info("="*60)
    logger.info("WITNESS COEFFICIENT ANALYSIS")
    logger.info("="*60)

    np.random.seed(seed)

    # Generate dataset
    logger.info("\n[1/3] Generating dataset...")
    states, labels = generate_distillability_dataset(
        n_samples=n_samples,
        noise_range=noise_range,
        seed=seed
    )
    labels = np.array(labels)

    # Extract features and train
    logger.info("\n[2/3] Training witness learner...")
    basis = create_sparse_measurement_set(3, 'two_body')
    features = extract_features_batch(states, basis, verbose=False)

    learner = SVMWitnessLearner(
        pauli_basis=basis,
        C=1.0,
        kernel='linear',
        random_state=seed
    )
    metrics = learner.train(features, labels, test_size=0.2, verbose=False)

    # Extract witness operator
    logger.info("\n[3/3] Analyzing witness coefficients...")
    witness = learner.get_witness_operator()

    # Get coefficients with Pauli labels
    coefficients = []
    for pauli, coeff in witness.to_list():
        # Use real part since witness coefficients should be real for Hermitian operator
        real_coeff = float(np.real(coeff))
        coefficients.append({
            'pauli': str(pauli),
            'coefficient': float(np.abs(real_coeff)),
            'sign': '+' if real_coeff >= 0 else '-',
            'raw_coefficient': real_coeff
        })

    # Sort by absolute value (importance)
    coefficients.sort(key=lambda x: x['coefficient'], reverse=True)

    # Calculate statistics
    all_coeffs = np.array([c['raw_coefficient'] for c in coefficients])

    # Categorize by Pauli type
    one_body = [c for c in coefficients if c['pauli'].count('I') == 2]
    two_body = [c for c in coefficients if c['pauli'].count('I') == 1]

    results = {
        'n_samples': n_samples,
        'noise_range': list(noise_range),
        'seed': seed,
        'training_metrics': {
            'test_accuracy': float(metrics['test_accuracy']),
            'test_precision': float(metrics['test_precision']),
            'test_recall': float(metrics['test_recall'])
        },
        'witness_statistics': {
            'n_terms': len(coefficients),
            'n_one_body': len(one_body),
            'n_two_body': len(two_body),
            'mean_abs_coefficient': float(np.mean(np.abs(all_coeffs))),
            'max_abs_coefficient': float(np.max(np.abs(all_coeffs))),
            'min_abs_coefficient': float(np.min(np.abs(all_coeffs)))
        },
        'ranked_coefficients': coefficients[:20],  # Top 20
        'all_coefficients': coefficients,
        'one_body_terms': one_body,
        'two_body_terms': two_body
    }

    # Summary
    logger.info("\n" + "="*60)
    logger.info("WITNESS COEFFICIENT ANALYSIS RESULTS")
    logger.info("="*60)
    logger.info(f"Total terms: {len(coefficients)}")
    logger.info(f"One-body terms: {len(one_body)}")
    logger.info(f"Two-body terms: {len(two_body)}")
    logger.info("\nTop 10 most important Pauli terms:")
    for i, c in enumerate(coefficients[:10]):
        logger.info(f"  {i+1}. {c['pauli']}: {c['sign']}{c['coefficient']:.4f}")

    # One-body vs two-body importance
    one_body_total = sum(c['coefficient'] for c in one_body)
    two_body_total = sum(c['coefficient'] for c in two_body)
    total = one_body_total + two_body_total

    logger.info(f"\nImportance by type:")
    logger.info(f"  One-body: {100*one_body_total/total:.1f}%")
    logger.info(f"  Two-body: {100*two_body_total/total:.1f}%")

    results['importance_by_type'] = {
        'one_body_fraction': float(one_body_total / total),
        'two_body_fraction': float(two_body_total / total)
    }

    if results_dir:
        save_results(results, 'witness_coefficients', results_dir)

    return results


def run_all_experiments(
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_range: Tuple[float, float] = DEFAULT_NOISE_RANGE,
    seed: int = 42,
    results_dir: Optional[Path] = None
) -> Dict:
    """Run all experiments and compile results."""
    logger.info("="*70)
    logger.info("RUNNING ALL EXPERIMENTS")
    logger.info("="*70)

    all_results = {}

    all_results['ablation'] = run_ablation_study(
        n_samples=n_samples, noise_range=noise_range, seed=seed, results_dir=results_dir
    )

    seeds = [seed + i * 100 for i in range(5)]
    all_results['cross_validation'] = run_cross_validation(
        n_samples=n_samples, noise_range=noise_range, seeds=seeds, results_dir=results_dir
    )

    all_results['per_family'] = run_per_family_analysis(
        n_samples_per_family=n_samples//5, noise_range=noise_range, seed=seed, results_dir=results_dir
    )

    all_results['noise_robustness'] = run_noise_robustness(
        n_samples_per_level=n_samples//5, seed=seed, results_dir=results_dir
    )

    all_results['witness_coefficients'] = analyze_witness_coefficients(
        n_samples=n_samples, noise_range=noise_range, seed=seed, results_dir=results_dir
    )

    # Save combined results
    if results_dir:
        save_results(all_results, 'all_experiments', results_dir)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Run experimental pipeline for 3-qubit distillability hypothesis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_experiments.py --experiment ablation
  python scripts/run_experiments.py --experiment cross_validation --n-samples 5000
  python scripts/run_experiments.py --experiment all --seed 42
        """
    )

    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        choices=['ablation', 'cross_validation', 'per_family', 'noise_robustness',
                 'witness_coefficients', 'all'],
        help='Which experiment to run'
    )

    parser.add_argument(
        '--n-samples',
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f'Number of samples (default: {DEFAULT_N_SAMPLES})'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    parser.add_argument(
        '--noise-min',
        type=float,
        default=0.0,
        help='Minimum noise level (default: 0.0)'
    )

    parser.add_argument(
        '--noise-max',
        type=float,
        default=0.5,
        help='Maximum noise level (default: 0.5)'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Directory to save results (default: results/)'
    )

    args = parser.parse_args()

    # Set up results directory using centralized default
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = RESULTS_DIR

    noise_range = (args.noise_min, args.noise_max)

    # Run selected experiment
    if args.experiment == 'ablation':
        run_ablation_study(
            n_samples=args.n_samples,
            noise_range=noise_range,
            seed=args.seed,
            results_dir=results_dir
        )
    elif args.experiment == 'cross_validation':
        # Use provided seed as the base for multiple seeds
        seeds = [args.seed + i * 100 for i in range(5)]
        run_cross_validation(
            n_samples=args.n_samples,
            noise_range=noise_range,
            seeds=seeds,
            results_dir=results_dir
        )
    elif args.experiment == 'per_family':
        run_per_family_analysis(
            n_samples_per_family=args.n_samples // 5,
            noise_range=noise_range,
            seed=args.seed,
            results_dir=results_dir
        )
    elif args.experiment == 'noise_robustness':
        run_noise_robustness(
            n_samples_per_level=args.n_samples // 5,
            seed=args.seed,
            results_dir=results_dir
        )
    elif args.experiment == 'witness_coefficients':
        analyze_witness_coefficients(
            n_samples=args.n_samples,
            noise_range=noise_range,
            seed=args.seed,
            results_dir=results_dir
        )
    elif args.experiment == 'all':
        run_all_experiments(
            n_samples=args.n_samples,
            noise_range=noise_range,
            seed=args.seed,
            results_dir=results_dir
        )

    logger.info("\nExperiment complete!")


if __name__ == '__main__':
    main()
