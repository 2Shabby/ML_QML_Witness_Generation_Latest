#!/usr/bin/env python
"""
Run noise robustness experiments across multiple noise models.

Tests all classical pipeline models (SVM, MLP, Transformer) under:
- Depolarizing noise (existing)
- Dephasing noise
- Amplitude damping noise

This supports manuscript Section 5.5 (Noise Robustness) and the abstract claim
about robustness under dephasing and amplitude-damping noise.
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

from qiskit.quantum_info import DensityMatrix
from sklearn.metrics import accuracy_score

from src.quantum_states.state_generation import (
    generate_entangled_state,
    generate_3qubit_product_state,
    generate_noisy_cluster_state,
    generate_random_density_matrix,
    check_npt_any_bipartition,
)
from src.feature_extraction.pauli_features import (
    create_sparse_measurement_set,
    extract_features_batch,
)
from src.ml_models.svm_witness import SVMWitnessLearner

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Import noise utilities from investigate_negative_results
from investigate_negative_results import (
    apply_dephasing_noise,
    apply_amplitude_damping,
)

try:
    from src.ml_models.mlp_classifier import MLPClassifierLearner
    from src.ml_models.transformer_witness import TransformerWitnessLearner
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def generate_noisy_dataset(
    n_samples: int = 1000,
    noise_type: str = 'depolarizing',
    noise_level: float = 0.2,
    seed: int = 42,
):
    """
    Generate a dataset with a specific noise model applied.

    Args:
        n_samples: Number of states
        noise_type: 'depolarizing', 'dephasing', or 'amplitude_damping'
        noise_level: Noise strength parameter
        seed: Random seed

    Returns:
        (states, labels) tuple
    """
    np.random.seed(seed)
    states = []
    labels = []

    n_per_family = n_samples // 5
    families = [
        ('ghz', generate_entangled_state),
        ('w', generate_entangled_state),
        ('cluster', None),
        ('random', None),
        ('product', None),
    ]

    for fam_idx, (fam_name, _) in enumerate(families):
        for i in range(n_per_family):
            state_seed = seed + fam_idx * n_per_family + i

            if fam_name in ('ghz', 'w'):
                # Generate pure state first, then apply noise
                state = generate_entangled_state(
                    n_qubits=3, entanglement_type=fam_name,
                    noise_level=0.0, seed=state_seed,
                )
                rho = np.array(state)

                if noise_type == 'depolarizing':
                    dim = rho.shape[0]
                    rho_noisy = (1 - noise_level) * rho + noise_level * np.eye(dim) / dim
                elif noise_type == 'dephasing':
                    rho_noisy = apply_dephasing_noise(rho, noise_level)
                elif noise_type == 'amplitude_damping':
                    rho_noisy = apply_amplitude_damping(rho, noise_level)
                else:
                    raise ValueError(f"Unknown noise type: {noise_type}")

                state = DensityMatrix(rho_noisy)

            elif fam_name == 'cluster':
                state = generate_noisy_cluster_state(
                    n_qubits=3, noise_level=0.0, seed=state_seed,
                )
                rho = np.array(state)

                if noise_type == 'depolarizing':
                    dim = rho.shape[0]
                    rho_noisy = (1 - noise_level) * rho + noise_level * np.eye(dim) / dim
                elif noise_type == 'dephasing':
                    rho_noisy = apply_dephasing_noise(rho, noise_level)
                elif noise_type == 'amplitude_damping':
                    rho_noisy = apply_amplitude_damping(rho, noise_level)
                else:
                    raise ValueError(f"Unknown noise type: {noise_type}")

                state = DensityMatrix(rho_noisy)

            elif fam_name == 'random':
                state = generate_random_density_matrix(
                    n_qubits=3, seed=state_seed,
                )

            elif fam_name == 'product':
                state = generate_3qubit_product_state(seed=state_seed)

            is_distillable = check_npt_any_bipartition(state)
            states.append(state)
            labels.append(1 if is_distillable else 0)

    return states, np.array(labels)


def train_and_evaluate_models(
    features_train, y_train, features_test, y_test,
    basis, seed=42,
):
    """Train SVM, MLP, Transformer and return accuracy dict."""
    results = {}

    # SVM
    svm = SVMWitnessLearner(
        pauli_basis=basis, C=1.0, kernel='linear', random_state=seed,
    )
    svm.svm.fit(features_train, y_train)
    svm_pred = svm.svm.predict(features_test)
    results['svm'] = accuracy_score(y_test, svm_pred)

    if TORCH_AVAILABLE:
        # MLP
        mlp = MLPClassifierLearner(n_features=36, random_state=seed)
        mlp.fit(features_train, y_train, features_test, y_test, verbose=False)
        mlp_pred = mlp.predict(features_test)
        results['mlp'] = accuracy_score(y_test, mlp_pred)

        # Transformer
        tf = TransformerWitnessLearner(
            pauli_basis=basis, mode='hybrid',
            d_model=16, n_heads=2, n_layers=1,
            n_epochs=50, patience=10, random_state=seed,
        )
        tf.train(
            np.vstack([features_train, features_test]),
            np.concatenate([y_train, y_test]),
            test_size=len(y_test) / (len(y_train) + len(y_test)),
            verbose=False,
        )
        tf_pred = tf.predict(features_test)
        results['transformer'] = accuracy_score(y_test, tf_pred)

    return results


def run_noise_sweep(
    noise_type: str = 'depolarizing',
    noise_levels: list = None,
    n_samples: int = 1000,
    seed: int = 42,
):
    """Sweep noise levels for a given noise model."""
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]

    logger.info(f"\n{'='*60}")
    logger.info(f"Noise Sweep: {noise_type}")
    logger.info(f"{'='*60}")

    basis = create_sparse_measurement_set(3, 'two_body')
    sweep_results = {}

    for p in noise_levels:
        logger.info(f"\n--- noise_level = {p:.2f} ---")

        # Generate noisy dataset
        states, labels = generate_noisy_dataset(
            n_samples=n_samples, noise_type=noise_type,
            noise_level=p, seed=seed,
        )

        features = extract_features_batch(states, basis, verbose=False)

        # Train/test split
        from sklearn.model_selection import train_test_split
        split_seed = seed + 1000
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2,
            random_state=split_seed, stratify=labels,
        )

        # Train and evaluate
        results = train_and_evaluate_models(
            X_train, y_train, X_test, y_test, basis, seed,
        )

        sweep_results[str(p)] = results

        for model, acc in results.items():
            logger.info(f"  {model}: {acc:.4f}")

    return sweep_results


def run_all_noise_experiments(output_dir: str = None):
    """Run noise robustness experiments across all noise models."""
    if output_dir is None:
        output_dir = str(project_root / 'results')
    os.makedirs(output_dir, exist_ok=True)

    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
    all_results = {}

    for noise_type in ['depolarizing', 'dephasing', 'amplitude_damping']:
        results = run_noise_sweep(
            noise_type=noise_type,
            noise_levels=noise_levels,
            n_samples=1000,
            seed=42,
        )
        all_results[noise_type] = results

    # Summary table
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY: Model accuracy across noise types")
    logger.info(f"{'='*60}")

    models = ['svm']
    if TORCH_AVAILABLE:
        models.extend(['mlp', 'transformer'])

    for model in models:
        logger.info(f"\n{model.upper()}:")
        header = f"  {'p':<6}"
        for nt in ['depolarizing', 'dephasing', 'amplitude_damping']:
            header += f" {nt:<18}"
        logger.info(header)

        for p in noise_levels:
            row = f"  {p:<6.1f}"
            for nt in ['depolarizing', 'dephasing', 'amplitude_damping']:
                acc = all_results[nt][str(p)].get(model, float('nan'))
                row += f" {acc:<18.4f}"
            logger.info(row)

    # Save
    output_path = os.path.join(output_dir, 'noise_experiments.json')
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'noise_levels': noise_levels,
            'n_samples': 1000,
            'results': all_results,
        }, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    return all_results


if __name__ == '__main__':
    run_all_noise_experiments()
