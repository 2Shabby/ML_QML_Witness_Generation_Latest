#!/usr/bin/env python3
"""
Transformer vs SVM Experimental Pipeline for 3-Qubit Distillability

This script runs comparative experiments between:
1. Linear SVM (baseline)
2. Transformer Classifier (pure classification)
3. Hybrid Transformer (with witness extraction)

Supports large datasets and provides comprehensive comparison metrics.

Usage:
    python scripts/run_transformer_experiments.py --experiment comparison
    python scripts/run_transformer_experiments.py --experiment transformer_only --n-samples 10000
    python scripts/run_transformer_experiments.py --experiment all --n-samples 5000
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

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
    DEFAULT_SEED,
    DEFAULT_TRANSFORMER_CONFIG,
    DEFAULT_LOG_FORMAT,
    RESULTS_DIR,
)
from src.utils import convert_to_json_serializable, TORCH_AVAILABLE

# Try to import transformer (may fail if torch not installed)
if TORCH_AVAILABLE:
    from src.ml_models.transformer_witness import TransformerWitnessLearner
else:
    print("Warning: PyTorch not installed. Install with: pip install torch")
    print("Transformer experiments will be skipped.")

# Set up logging using centralized utility
logging.basicConfig(level=logging.INFO, format=DEFAULT_LOG_FORMAT)
logger = logging.getLogger(__name__)


def save_results(results: Dict, experiment_name: str, results_dir: Path) -> str:
    """Save results to JSON file with timestamp."""
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = results_dir / filename

    results_copy = convert_to_json_serializable(results)
    results_copy['metadata'] = {
        'experiment': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'n_samples': results_copy.get('n_samples', 'N/A'),
        'torch_available': TORCH_AVAILABLE
    }

    with open(filepath, 'w') as f:
        json.dump(results_copy, f, indent=2)

    logger.info(f"Results saved to {filepath}")
    return str(filepath)


def generate_dataset(
    n_samples: int,
    noise_range: Tuple[float, float],
    seed: int
) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Generate dataset with features and labels.

    Returns:
        features: (n_samples, 36) array of Pauli expectations
        labels: (n_samples,) array of binary labels
        states: List of density matrices (for witness evaluation)
    """
    logger.info(f"Generating {n_samples} samples with noise range {noise_range}...")

    states, labels = generate_distillability_dataset(
        n_samples=n_samples,
        noise_range=noise_range,
        seed=seed
    )
    labels = np.array(labels)

    # Extract 36D features
    basis = create_sparse_measurement_set(3, 'two_body')
    features = extract_features_batch(states, basis, verbose=False)

    logger.info(f"Dataset: {n_samples} samples, {features.shape[1]} features")
    logger.info(f"Class balance: {np.mean(labels):.2%} distillable")

    return features, labels, states, basis


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    basis,
    seed: int
) -> Dict:
    """Train SVM and return metrics."""
    logger.info("Training SVM...")

    learner = SVMWitnessLearner(
        pauli_basis=basis,
        C=1.0,
        kernel='linear',
        random_state=seed
    )

    # Manual train/test since we have pre-split data
    from sklearn.svm import SVC
    svm = SVC(kernel='linear', C=1.0, random_state=seed, probability=True)
    svm.fit(X_train, y_train)

    y_pred_train = svm.predict(X_train)
    y_pred_test = svm.predict(X_test)

    metrics = {
        'model': 'SVM',
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'train_precision': precision_score(y_train, y_pred_train, zero_division=0),
        'test_precision': precision_score(y_test, y_pred_test, zero_division=0),
        'train_recall': recall_score(y_train, y_pred_train, zero_division=0),
        'test_recall': recall_score(y_test, y_pred_test, zero_division=0),
        'test_f1': f1_score(y_test, y_pred_test, zero_division=0),
        'n_support_vectors': len(svm.support_),
        'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist()
    }

    logger.info(f"SVM Test Accuracy: {metrics['test_accuracy']:.4f}")

    return metrics


def train_transformer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    basis,
    mode: str,
    config: Dict,
    seed: int
) -> Dict:
    """Train transformer and return metrics."""
    if not TORCH_AVAILABLE:
        return {'error': 'PyTorch not available'}

    logger.info(f"Training Transformer ({mode} mode)...")

    learner = TransformerWitnessLearner(
        pauli_basis=basis,
        mode=mode,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        patience=config['patience'],
        random_state=seed
    )

    # Train on pre-split data (no data leakage)
    train_metrics = learner.fit(X_train, y_train, X_test, y_test, verbose=True)

    # Evaluate on test set
    y_pred_test = learner.predict(X_test)

    metrics = {
        'model': f'Transformer_{mode}',
        'train_accuracy': train_metrics['train_accuracy'],
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'train_precision': train_metrics['train_precision'],
        'test_precision': precision_score(y_test, y_pred_test, zero_division=0),
        'train_recall': train_metrics['train_recall'],
        'test_recall': recall_score(y_test, y_pred_test, zero_division=0),
        'test_f1': f1_score(y_test, y_pred_test, zero_division=0),
        'n_parameters': train_metrics['n_parameters'],
        'n_epochs_trained': train_metrics['n_epochs_trained'],
        'best_val_loss': train_metrics['best_val_loss'],
        'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist(),
        'config': config
    }

    # Add witness info for hybrid mode
    if mode == 'hybrid':
        try:
            witness = learner.get_witness_operator()
            metrics['n_witness_terms'] = len(witness.to_list())
            metrics['measurement_cost'] = learner.get_measurement_cost()
        except Exception as e:
            metrics['witness_error'] = str(e)

    logger.info(f"Transformer ({mode}) Test Accuracy: {metrics['test_accuracy']:.4f}")

    return metrics, learner


def run_model_comparison(
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_range: Tuple[float, float] = DEFAULT_NOISE_RANGE,
    seed: int = DEFAULT_SEED,
    transformer_config: Dict = None,
    results_dir: Optional[Path] = None
) -> Dict:
    """
    Compare SVM, Transformer Classifier, and Hybrid Transformer.

    Args:
        n_samples: Number of samples to generate
        noise_range: Range of noise levels for state generation
        seed: Random seed for reproducibility
        transformer_config: Configuration for transformer models
        results_dir: Directory to save results

    Returns:
        Dictionary with comparison results
    """
    logger.info("="*70)
    logger.info("MODEL COMPARISON: SVM vs Transformer")
    logger.info("="*70)

    if transformer_config is None:
        transformer_config = DEFAULT_TRANSFORMER_CONFIG.to_dict()

    np.random.seed(seed)

    # Generate dataset
    features, labels, states, basis = generate_dataset(n_samples, noise_range, seed)

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=seed, stratify=labels
    )

    results = {
        'n_samples': n_samples,
        'noise_range': list(noise_range),
        'seed': seed,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'class_balance': float(np.mean(labels)),
        'models': {}
    }

    # Train SVM
    svm_metrics = train_svm(X_train, y_train, X_test, y_test, basis, seed)
    results['models']['svm'] = svm_metrics

    # Train Transformer Classifier
    if TORCH_AVAILABLE:
        tf_class_metrics, _ = train_transformer(
            X_train, y_train, X_test, y_test, basis,
            mode='classifier', config=transformer_config, seed=seed
        )
        results['models']['transformer_classifier'] = tf_class_metrics

        # Train Hybrid Transformer
        tf_hybrid_metrics, hybrid_learner = train_transformer(
            X_train, y_train, X_test, y_test, basis,
            mode='hybrid', config=transformer_config, seed=seed
        )
        results['models']['transformer_hybrid'] = tf_hybrid_metrics

    # Summary comparison
    logger.info("\n" + "="*70)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*70)

    for model_name, metrics in results['models'].items():
        if 'error' not in metrics:
            logger.info(f"{model_name:25s}: acc={metrics['test_accuracy']:.4f}, "
                       f"prec={metrics['test_precision']:.4f}, "
                       f"rec={metrics['test_recall']:.4f}, "
                       f"f1={metrics['test_f1']:.4f}")

    if results_dir:
        save_results(results, 'model_comparison', results_dir)

    return results


def run_cross_validation_comparison(
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_range: Tuple[float, float] = DEFAULT_NOISE_RANGE,
    n_folds: int = DEFAULT_CV_FOLDS,
    seed: int = DEFAULT_SEED,
    transformer_config: Dict = None,
    results_dir: Optional[Path] = None
) -> Dict:
    """
    Run cross-validation comparison between SVM and Transformer.

    Args:
        n_samples: Number of samples to generate
        noise_range: Range of noise levels
        n_folds: Number of CV folds
        seed: Random seed
        transformer_config: Configuration for transformer
        results_dir: Directory to save results

    Returns:
        Dictionary with CV results
    """
    logger.info("="*70)
    logger.info(f"CROSS-VALIDATION COMPARISON ({n_folds}-fold)")
    logger.info("="*70)

    if transformer_config is None:
        transformer_config = DEFAULT_TRANSFORMER_CONFIG.to_dict()

    np.random.seed(seed)

    # Generate dataset
    features, labels, states, basis = generate_dataset(n_samples, noise_range, seed)

    # Initialize result containers
    model_results = {
        'svm': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
    }
    if TORCH_AVAILABLE:
        model_results['transformer_classifier'] = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        model_results['transformer_hybrid'] = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    # Cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
        logger.info(f"\n--- Fold {fold + 1}/{n_folds} ---")

        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # SVM
        svm_metrics = train_svm(X_train, y_train, X_test, y_test, basis, seed)
        model_results['svm']['accuracy'].append(svm_metrics['test_accuracy'])
        model_results['svm']['precision'].append(svm_metrics['test_precision'])
        model_results['svm']['recall'].append(svm_metrics['test_recall'])
        model_results['svm']['f1'].append(svm_metrics['test_f1'])

        # Transformers
        if TORCH_AVAILABLE:
            for mode in ['classifier', 'hybrid']:
                tf_metrics, _ = train_transformer(
                    X_train, y_train, X_test, y_test, basis,
                    mode=mode, config=transformer_config, seed=seed + fold
                )
                key = f'transformer_{mode}'
                model_results[key]['accuracy'].append(tf_metrics['test_accuracy'])
                model_results[key]['precision'].append(tf_metrics['test_precision'])
                model_results[key]['recall'].append(tf_metrics['test_recall'])
                model_results[key]['f1'].append(tf_metrics['test_f1'])

    # Aggregate results
    results = {
        'n_samples': n_samples,
        'noise_range': list(noise_range),
        'n_folds': n_folds,
        'seed': seed,
        'models': {}
    }

    for model_name, metrics in model_results.items():
        results['models'][model_name] = {
            'accuracy_mean': float(np.mean(metrics['accuracy'])),
            'accuracy_std': float(np.std(metrics['accuracy'])),
            'precision_mean': float(np.mean(metrics['precision'])),
            'recall_mean': float(np.mean(metrics['recall'])),
            'f1_mean': float(np.mean(metrics['f1'])),
            'fold_accuracies': [float(x) for x in metrics['accuracy']]
        }

    # Summary
    logger.info("\n" + "="*70)
    logger.info("CROSS-VALIDATION RESULTS")
    logger.info("="*70)

    for model_name, model_stats in results['models'].items():
        logger.info(f"{model_name:25s}: acc={model_stats['accuracy_mean']:.4f} +/- {model_stats['accuracy_std']:.4f}")

    if results_dir:
        save_results(results, 'cv_comparison', results_dir)

    return results


def run_scaling_experiment(
    sample_sizes: List[int] = None,
    noise_range: Tuple[float, float] = DEFAULT_NOISE_RANGE,
    seed: int = DEFAULT_SEED,
    transformer_config: Dict = None,
    results_dir: Optional[Path] = None
) -> Dict:
    """
    Test how model performance scales with dataset size.

    Args:
        sample_sizes: List of sample sizes to test
        noise_range: Range of noise levels
        seed: Random seed
        transformer_config: Configuration for transformer
        results_dir: Directory to save results

    Returns:
        Dictionary with scaling results
    """
    if sample_sizes is None:
        sample_sizes = [500, 1000, 2000, 5000, 10000]

    logger.info("="*70)
    logger.info("SCALING EXPERIMENT: Performance vs Dataset Size")
    logger.info("="*70)

    if transformer_config is None:
        transformer_config = DEFAULT_TRANSFORMER_CONFIG.to_dict()

    results = {
        'sample_sizes': sample_sizes,
        'noise_range': list(noise_range),
        'seed': seed,
        'per_size': {}
    }

    for n_samples in sample_sizes:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing with {n_samples} samples")
        logger.info(f"{'='*50}")

        comparison = run_model_comparison(
            n_samples=n_samples,
            noise_range=noise_range,
            seed=seed,
            transformer_config=transformer_config,
            results_dir=None  # Don't save intermediate results
        )

        results['per_size'][str(n_samples)] = {
            'n_samples': n_samples,
            'models': comparison['models']
        }

    # Summary
    logger.info("\n" + "="*70)
    logger.info("SCALING EXPERIMENT SUMMARY")
    logger.info("="*70)
    logger.info(f"{'N samples':>10} | {'SVM':>10} | {'TF-Class':>10} | {'TF-Hybrid':>10}")
    logger.info("-" * 50)

    for n_samples in sample_sizes:
        size_results = results['per_size'][str(n_samples)]['models']
        svm_acc = size_results['svm']['test_accuracy']
        tf_class_acc = size_results.get('transformer_classifier', {}).get('test_accuracy', 'N/A')
        tf_hybrid_acc = size_results.get('transformer_hybrid', {}).get('test_accuracy', 'N/A')

        tf_class_str = f"{tf_class_acc:.4f}" if isinstance(tf_class_acc, float) else tf_class_acc
        tf_hybrid_str = f"{tf_hybrid_acc:.4f}" if isinstance(tf_hybrid_acc, float) else tf_hybrid_acc

        logger.info(f"{n_samples:>10} | {svm_acc:>10.4f} | {tf_class_str:>10} | {tf_hybrid_str:>10}")

    if results_dir:
        save_results(results, 'scaling_experiment', results_dir)

    return results


def run_hyperparameter_search(
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_range: Tuple[float, float] = DEFAULT_NOISE_RANGE,
    seed: int = DEFAULT_SEED,
    results_dir: Optional[Path] = None
) -> Dict:
    """
    Search for best transformer hyperparameters.

    Args:
        n_samples: Number of samples
        noise_range: Range of noise levels
        seed: Random seed
        results_dir: Directory to save results

    Returns:
        Dictionary with search results
    """
    if not TORCH_AVAILABLE:
        logger.error("PyTorch required for hyperparameter search")
        return {'error': 'PyTorch not available'}

    logger.info("="*70)
    logger.info("HYPERPARAMETER SEARCH")
    logger.info("="*70)

    np.random.seed(seed)

    # Generate dataset
    features, labels, states, basis = generate_dataset(n_samples, noise_range, seed)

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=seed, stratify=labels
    )

    # Hyperparameter grid
    configs = [
        {'d_model': 32, 'n_heads': 2, 'n_layers': 1, 'd_ff': 64},
        {'d_model': 64, 'n_heads': 4, 'n_layers': 2, 'd_ff': 128},
        {'d_model': 128, 'n_heads': 4, 'n_layers': 2, 'd_ff': 256},
        {'d_model': 64, 'n_heads': 4, 'n_layers': 3, 'd_ff': 128},
        {'d_model': 64, 'n_heads': 8, 'n_layers': 2, 'd_ff': 128},
    ]

    results = {
        'n_samples': n_samples,
        'seed': seed,
        'search_results': []
    }

    best_accuracy = 0
    best_config = None

    for i, config in enumerate(configs):
        logger.info(f"\n[{i+1}/{len(configs)}] Testing config: {config}")

        full_config = DEFAULT_TRANSFORMER_CONFIG.copy()
        full_config.update(config)

        try:
            metrics, _ = train_transformer(
                X_train, y_train, X_test, y_test, basis,
                mode='hybrid', config=full_config, seed=seed
            )

            result = {
                'config': config,
                'test_accuracy': metrics['test_accuracy'],
                'n_parameters': metrics['n_parameters'],
                'n_epochs_trained': metrics['n_epochs_trained']
            }

            if metrics['test_accuracy'] > best_accuracy:
                best_accuracy = metrics['test_accuracy']
                best_config = config

        except Exception as e:
            result = {'config': config, 'error': str(e)}

        results['search_results'].append(result)

    results['best_config'] = best_config
    results['best_accuracy'] = best_accuracy

    # Summary
    logger.info("\n" + "="*70)
    logger.info("HYPERPARAMETER SEARCH RESULTS")
    logger.info("="*70)
    logger.info(f"Best config: {best_config}")
    logger.info(f"Best accuracy: {best_accuracy:.4f}")

    if results_dir:
        save_results(results, 'hyperparameter_search', results_dir)

    return results


def run_per_family_comparison(
    n_samples_per_family: int = 400,
    noise_range: Tuple[float, float] = DEFAULT_NOISE_RANGE,
    seed: int = DEFAULT_SEED,
    transformer_config: Dict = None,
    results_dir: Optional[Path] = None
) -> Dict:
    """
    Compare SVM, Transformer Classifier, and Hybrid on each state family.

    Evaluates per-family accuracy for GHZ, W, Cluster, Random, Product states.

    Args:
        n_samples_per_family: Samples per state family for testing
        noise_range: Range of noise levels
        seed: Random seed
        transformer_config: Configuration for transformer
        results_dir: Directory to save results

    Returns:
        Dictionary with per-family comparison results
    """
    logger.info("="*70)
    logger.info("PER-FAMILY COMPARISON: All Models by State Type")
    logger.info("="*70)

    if transformer_config is None:
        transformer_config = DEFAULT_TRANSFORMER_CONFIG.to_dict()

    np.random.seed(seed)

    # Generate training data (mixed families)
    logger.info("\n[1/4] Generating training dataset...")
    train_n_samples = n_samples_per_family * 5
    features, labels, states, basis = generate_dataset(train_n_samples, noise_range, seed)

    # Split for training
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=seed, stratify=labels
    )

    # Train all models
    logger.info("\n[2/4] Training models...")

    # SVM
    from sklearn.svm import SVC
    svm = SVC(kernel='linear', C=1.0, random_state=seed, probability=True)
    svm.fit(X_train, y_train)

    # Transformers
    tf_classifier = None
    tf_hybrid = None
    if TORCH_AVAILABLE:
        tf_classifier = TransformerWitnessLearner(
            pauli_basis=basis, mode='classifier', **transformer_config, random_state=seed
        )
        tf_classifier.fit(X_train, y_train, X_val, y_val, verbose=False)

        tf_hybrid = TransformerWitnessLearner(
            pauli_basis=basis, mode='hybrid', **transformer_config, random_state=seed
        )
        tf_hybrid.fit(X_train, y_train, X_val, y_val, verbose=False)

    # Generate test data per family
    logger.info("\n[3/4] Generating per-family test sets...")

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
    logger.info("\n[4/4] Evaluating per family...")

    results = {
        'n_samples_per_family': n_samples_per_family,
        'noise_range': list(noise_range),
        'seed': seed,
        'per_family': {}
    }

    for family_name, family_data in families.items():
        family_features = extract_features_batch(family_data['states'], basis, verbose=False)
        family_labels = np.array(family_data['labels'])

        family_results = {
            'n_samples': len(family_labels),
            'n_distillable': int(np.sum(family_labels)),
            'distillable_fraction': float(np.mean(family_labels)),
            'models': {}
        }

        # SVM predictions
        svm_pred = svm.predict(family_features)
        family_results['models']['svm'] = {
            'accuracy': float(accuracy_score(family_labels, svm_pred)),
            'precision': float(precision_score(family_labels, svm_pred, zero_division=0)),
            'recall': float(recall_score(family_labels, svm_pred, zero_division=0)),
            'f1': float(f1_score(family_labels, svm_pred, zero_division=0))
        }

        # Transformer predictions
        if TORCH_AVAILABLE and tf_classifier is not None:
            tf_class_pred = tf_classifier.predict(family_features)
            family_results['models']['transformer_classifier'] = {
                'accuracy': float(accuracy_score(family_labels, tf_class_pred)),
                'precision': float(precision_score(family_labels, tf_class_pred, zero_division=0)),
                'recall': float(recall_score(family_labels, tf_class_pred, zero_division=0)),
                'f1': float(f1_score(family_labels, tf_class_pred, zero_division=0))
            }

            tf_hybrid_pred = tf_hybrid.predict(family_features)
            family_results['models']['transformer_hybrid'] = {
                'accuracy': float(accuracy_score(family_labels, tf_hybrid_pred)),
                'precision': float(precision_score(family_labels, tf_hybrid_pred, zero_division=0)),
                'recall': float(recall_score(family_labels, tf_hybrid_pred, zero_division=0)),
                'f1': float(f1_score(family_labels, tf_hybrid_pred, zero_division=0))
            }

        results['per_family'][family_name] = family_results

        # Log results
        logger.info(f"\n{family_name.upper()}:")
        for model_name, metrics in family_results['models'].items():
            logger.info(f"  {model_name:25s}: acc={metrics['accuracy']:.3f}, f1={metrics['f1']:.3f}")

    if results_dir:
        save_results(results, 'per_family_comparison', results_dir)

    return results


def run_ablation_comparison(
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_range: Tuple[float, float] = DEFAULT_NOISE_RANGE,
    n_folds: int = DEFAULT_CV_FOLDS,
    seed: int = DEFAULT_SEED,
    transformer_config: Dict = None,
    results_dir: Optional[Path] = None
) -> Dict:
    """
    Compare 36D restricted vs 63D full features for all models.

    Args:
        n_samples: Number of samples
        noise_range: Range of noise levels
        n_folds: Number of CV folds
        seed: Random seed
        transformer_config: Configuration for transformer
        results_dir: Directory to save results

    Returns:
        Dictionary with ablation comparison results
    """
    logger.info("="*70)
    logger.info("ABLATION COMPARISON: 36D vs 63D for All Models")
    logger.info("="*70)

    if transformer_config is None:
        transformer_config = DEFAULT_TRANSFORMER_CONFIG.to_dict()

    np.random.seed(seed)
    from scipy.stats import ttest_rel

    # Generate dataset
    logger.info("\n[1/3] Generating dataset...")
    states, labels = generate_distillability_dataset(
        n_samples=n_samples,
        noise_range=noise_range,
        seed=seed
    )
    labels = np.array(labels)

    # Create feature sets
    logger.info("\n[2/3] Extracting features...")
    basis_36d = create_sparse_measurement_set(3, 'two_body')
    basis_63d = get_pauli_basis(3, include_identity=False)

    features_36d = extract_features_batch(states, basis_36d, verbose=False)
    features_63d = extract_features_batch(states, basis_63d, verbose=False)

    logger.info(f"  36D features: {features_36d.shape}")
    logger.info(f"  63D features: {features_63d.shape}")

    # Cross-validation
    logger.info(f"\n[3/3] Running {n_folds}-fold cross-validation...")

    from sklearn.svm import SVC
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # Results containers
    model_configs = ['svm']
    if TORCH_AVAILABLE:
        model_configs.extend(['transformer_classifier', 'transformer_hybrid'])

    results_by_model = {}
    for model in model_configs:
        results_by_model[model] = {
            '36d': {'accuracy': [], 'f1': []},
            '63d': {'accuracy': [], 'f1': []}
        }

    for fold, (train_idx, test_idx) in enumerate(skf.split(features_36d, labels)):
        logger.info(f"\n--- Fold {fold + 1}/{n_folds} ---")

        # Train and evaluate each model on both feature sets
        for dim, features in [('36d', features_36d), ('63d', features_63d)]:
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            basis = basis_36d if dim == '36d' else basis_63d

            # SVM
            svm = SVC(kernel='linear', C=1.0, random_state=seed)
            svm.fit(X_train, y_train)
            svm_pred = svm.predict(X_test)
            results_by_model['svm'][dim]['accuracy'].append(accuracy_score(y_test, svm_pred))
            results_by_model['svm'][dim]['f1'].append(f1_score(y_test, svm_pred, zero_division=0))

            # Transformers
            if TORCH_AVAILABLE:
                for mode in ['classifier', 'hybrid']:
                    key = f'transformer_{mode}'
                    learner = TransformerWitnessLearner(
                        pauli_basis=basis, mode=mode, **transformer_config, random_state=seed+fold
                    )
                    learner.fit(X_train, y_train, X_test, y_test, verbose=False)
                    pred = learner.predict(X_test)
                    results_by_model[key][dim]['accuracy'].append(accuracy_score(y_test, pred))
                    results_by_model[key][dim]['f1'].append(f1_score(y_test, pred, zero_division=0))

    # Compile results
    results = {
        'n_samples': n_samples,
        'noise_range': list(noise_range),
        'n_folds': n_folds,
        'seed': seed,
        'models': {}
    }

    for model_name, model_data in results_by_model.items():
        acc_36d = np.array(model_data['36d']['accuracy'])
        acc_63d = np.array(model_data['63d']['accuracy'])

        t_stat, p_value = ttest_rel(acc_63d, acc_36d)

        results['models'][model_name] = {
            'restricted_36d': {
                'accuracy_mean': float(np.mean(acc_36d)),
                'accuracy_std': float(np.std(acc_36d)),
                'f1_mean': float(np.mean(model_data['36d']['f1'])),
                'fold_accuracies': [float(x) for x in acc_36d]
            },
            'full_63d': {
                'accuracy_mean': float(np.mean(acc_63d)),
                'accuracy_std': float(np.std(acc_63d)),
                'f1_mean': float(np.mean(model_data['63d']['f1'])),
                'fold_accuracies': [float(x) for x in acc_63d]
            },
            'statistical_comparison': {
                'accuracy_gap': float(np.mean(acc_63d) - np.mean(acc_36d)),
                'paired_ttest_t': float(t_stat),
                'paired_ttest_p': float(p_value),
                'significant_at_0.05': p_value < 0.05
            }
        }

    # Summary
    logger.info("\n" + "="*70)
    logger.info("ABLATION COMPARISON RESULTS")
    logger.info("="*70)
    logger.info(f"{'Model':25s} | {'36D Acc':>10} | {'63D Acc':>10} | {'Gap':>8} | {'p-value':>8}")
    logger.info("-" * 70)

    for model_name, model_results in results['models'].items():
        acc_36 = model_results['restricted_36d']['accuracy_mean']
        acc_63 = model_results['full_63d']['accuracy_mean']
        gap = model_results['statistical_comparison']['accuracy_gap']
        p_val = model_results['statistical_comparison']['paired_ttest_p']
        sig = "*" if model_results['statistical_comparison']['significant_at_0.05'] else ""
        logger.info(f"{model_name:25s} | {acc_36:>10.4f} | {acc_63:>10.4f} | {gap:>+8.4f} | {p_val:>7.4f}{sig}")

    if results_dir:
        save_results(results, 'ablation_comparison', results_dir)

    return results


def run_witness_analysis(
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_range: Tuple[float, float] = DEFAULT_NOISE_RANGE,
    seed: int = DEFAULT_SEED,
    transformer_config: Dict = None,
    results_dir: Optional[Path] = None
) -> Dict:
    """
    Compare witness operators from SVM and Hybrid Transformer.

    Args:
        n_samples: Number of samples
        noise_range: Range of noise levels
        seed: Random seed
        transformer_config: Configuration for transformer
        results_dir: Directory to save results

    Returns:
        Dictionary with witness comparison
    """
    logger.info("="*70)
    logger.info("WITNESS ANALYSIS: SVM vs Hybrid Transformer")
    logger.info("="*70)

    if transformer_config is None:
        transformer_config = DEFAULT_TRANSFORMER_CONFIG.to_dict()

    np.random.seed(seed)

    # Generate dataset
    features, labels, states, basis = generate_dataset(n_samples, noise_range, seed)

    results = {
        'n_samples': n_samples,
        'noise_range': list(noise_range),
        'seed': seed,
        'witnesses': {}
    }

    # Train SVM and extract witness
    logger.info("\nTraining SVM...")
    svm_learner = SVMWitnessLearner(
        pauli_basis=basis,
        C=1.0,
        kernel='linear',
        random_state=seed
    )
    svm_metrics = svm_learner.train(features, labels, test_size=0.2, verbose=True)

    svm_witness = svm_learner.get_witness_operator()
    svm_coeffs = {str(p): float(np.real(c)) for p, c in svm_witness.to_list()}

    results['witnesses']['svm'] = {
        'test_accuracy': svm_metrics['test_accuracy'],
        'n_terms': len(svm_coeffs),
        'measurement_cost': svm_learner.get_measurement_cost(),
        'top_coefficients': sorted(
            [(k, abs(v)) for k, v in svm_coeffs.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
    }

    # Train Hybrid Transformer and extract witness
    if TORCH_AVAILABLE:
        logger.info("\nTraining Hybrid Transformer...")
        tf_learner = TransformerWitnessLearner(
            pauli_basis=basis,
            mode='hybrid',
            **transformer_config,
            random_state=seed
        )
        tf_metrics = tf_learner.train(features, labels, test_size=0.2, verbose=True)

        tf_witness = tf_learner.get_witness_operator()
        tf_coeffs = {str(p): float(np.real(c)) for p, c in tf_witness.to_list()}

        results['witnesses']['transformer_hybrid'] = {
            'test_accuracy': tf_metrics['test_accuracy'],
            'n_terms': len(tf_coeffs),
            'measurement_cost': tf_learner.get_measurement_cost(),
            'top_coefficients': sorted(
                [(k, abs(v)) for k, v in tf_coeffs.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

        # Compare coefficient correlations
        common_paulis = set(svm_coeffs.keys()) & set(tf_coeffs.keys())
        if len(common_paulis) > 0:
            svm_vec = np.array([svm_coeffs.get(p, 0) for p in sorted(common_paulis)])
            tf_vec = np.array([tf_coeffs.get(p, 0) for p in sorted(common_paulis)])
            correlation = np.corrcoef(svm_vec, tf_vec)[0, 1]
            results['coefficient_correlation'] = float(correlation)

    # Summary
    logger.info("\n" + "="*70)
    logger.info("WITNESS ANALYSIS SUMMARY")
    logger.info("="*70)

    for model, data in results['witnesses'].items():
        logger.info(f"\n{model.upper()}:")
        logger.info(f"  Accuracy: {data['test_accuracy']:.4f}")
        logger.info(f"  Terms: {data['n_terms']}")
        logger.info(f"  Measurement cost: {data['measurement_cost']}")
        logger.info("  Top 5 coefficients:")
        for pauli, coeff in data['top_coefficients'][:5]:
            logger.info(f"    {pauli}: {coeff:.4f}")

    if 'coefficient_correlation' in results:
        logger.info(f"\nCoefficient correlation: {results['coefficient_correlation']:.4f}")

    if results_dir:
        save_results(results, 'witness_analysis', results_dir)

    return results


def run_all_experiments(
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_range: Tuple[float, float] = DEFAULT_NOISE_RANGE,
    seed: int = DEFAULT_SEED,
    results_dir: Optional[Path] = None
) -> Dict:
    """Run all experiments and compile results."""
    logger.info("="*70)
    logger.info("RUNNING ALL TRANSFORMER EXPERIMENTS")
    logger.info("="*70)

    transformer_config = DEFAULT_TRANSFORMER_CONFIG.to_dict()

    all_results = {}

    # Model comparison
    all_results['comparison'] = run_model_comparison(
        n_samples=n_samples, noise_range=noise_range, seed=seed,
        transformer_config=transformer_config, results_dir=results_dir
    )

    # Cross-validation
    all_results['cross_validation'] = run_cross_validation_comparison(
        n_samples=n_samples, noise_range=noise_range, n_folds=5, seed=seed,
        transformer_config=transformer_config, results_dir=results_dir
    )

    # Per-family comparison
    all_results['per_family'] = run_per_family_comparison(
        n_samples_per_family=n_samples // 5, noise_range=noise_range, seed=seed,
        transformer_config=transformer_config, results_dir=results_dir
    )

    # Ablation study (36D vs 63D)
    all_results['ablation'] = run_ablation_comparison(
        n_samples=n_samples, noise_range=noise_range, seed=seed,
        transformer_config=transformer_config, results_dir=results_dir
    )

    # Witness analysis
    all_results['witness_analysis'] = run_witness_analysis(
        n_samples=n_samples, noise_range=noise_range, seed=seed,
        transformer_config=transformer_config, results_dir=results_dir
    )

    if results_dir:
        save_results(all_results, 'all_transformer_experiments', results_dir)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Run transformer vs SVM comparison experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_transformer_experiments.py --experiment comparison
  python scripts/run_transformer_experiments.py --experiment cv --n-samples 5000
  python scripts/run_transformer_experiments.py --experiment scaling
  python scripts/run_transformer_experiments.py --experiment witness
  python scripts/run_transformer_experiments.py --experiment hyperparam
  python scripts/run_transformer_experiments.py --experiment all --n-samples 5000
        """
    )

    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        choices=['comparison', 'cv', 'scaling', 'witness', 'hyperparam', 'family', 'ablation', 'all'],
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
        default=DEFAULT_SEED,
        help=f'Random seed (default: {DEFAULT_SEED})'
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

    # Transformer hyperparameters (use centralized defaults from config dataclass)
    parser.add_argument('--d-model', type=int, default=DEFAULT_TRANSFORMER_CONFIG.d_model,
                        help=f'Transformer hidden dimension (default: {DEFAULT_TRANSFORMER_CONFIG.d_model})')
    parser.add_argument('--n-heads', type=int, default=DEFAULT_TRANSFORMER_CONFIG.n_heads,
                        help=f'Number of attention heads (default: {DEFAULT_TRANSFORMER_CONFIG.n_heads})')
    parser.add_argument('--n-layers', type=int, default=DEFAULT_TRANSFORMER_CONFIG.n_layers,
                        help=f'Number of transformer layers (default: {DEFAULT_TRANSFORMER_CONFIG.n_layers})')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_TRANSFORMER_CONFIG.batch_size,
                        help=f'Training batch size (default: {DEFAULT_TRANSFORMER_CONFIG.batch_size})')
    parser.add_argument('--n-epochs', type=int, default=DEFAULT_TRANSFORMER_CONFIG.n_epochs,
                        help=f'Maximum training epochs (default: {DEFAULT_TRANSFORMER_CONFIG.n_epochs})')

    args = parser.parse_args()

    # Set up results directory using centralized default
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = RESULTS_DIR

    noise_range = (args.noise_min, args.noise_max)

    # Build transformer config from args
    transformer_config = {
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'd_ff': args.d_model * 2,
        'dropout': 0.1,
        'learning_rate': 1e-3,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'patience': 15
    }

    # Run selected experiment
    if args.experiment == 'comparison':
        run_model_comparison(
            n_samples=args.n_samples,
            noise_range=noise_range,
            seed=args.seed,
            transformer_config=transformer_config,
            results_dir=results_dir
        )
    elif args.experiment == 'cv':
        run_cross_validation_comparison(
            n_samples=args.n_samples,
            noise_range=noise_range,
            seed=args.seed,
            transformer_config=transformer_config,
            results_dir=results_dir
        )
    elif args.experiment == 'scaling':
        run_scaling_experiment(
            noise_range=noise_range,
            seed=args.seed,
            transformer_config=transformer_config,
            results_dir=results_dir
        )
    elif args.experiment == 'witness':
        run_witness_analysis(
            n_samples=args.n_samples,
            noise_range=noise_range,
            seed=args.seed,
            transformer_config=transformer_config,
            results_dir=results_dir
        )
    elif args.experiment == 'hyperparam':
        run_hyperparameter_search(
            n_samples=args.n_samples,
            noise_range=noise_range,
            seed=args.seed,
            results_dir=results_dir
        )
    elif args.experiment == 'family':
        run_per_family_comparison(
            n_samples_per_family=args.n_samples // 5,
            noise_range=noise_range,
            seed=args.seed,
            transformer_config=transformer_config,
            results_dir=results_dir
        )
    elif args.experiment == 'ablation':
        run_ablation_comparison(
            n_samples=args.n_samples,
            noise_range=noise_range,
            seed=args.seed,
            transformer_config=transformer_config,
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
