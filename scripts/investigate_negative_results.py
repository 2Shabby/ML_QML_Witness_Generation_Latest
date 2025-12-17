#!/usr/bin/env python3
"""
Investigate Potential Negative Results

This script systematically searches for scenarios where:
1. 36D restricted features fail to distinguish distillable from non-distillable
2. 3-body correlations are essential
3. Adversarial noise models break the classifier

Goal: Find a "strong negative result" demonstrating limitations of restricted features.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.quantum_states.state_generation import (
    generate_entangled_state,
    generate_3qubit_product_state,
    generate_noisy_cluster_state,
    generate_random_density_matrix,
    check_npt_any_bipartition
)
from src.feature_extraction.pauli_features import (
    create_sparse_measurement_set,
    get_pauli_basis,
    extract_features_batch,
    extract_pauli_features
)
from qiskit.quantum_info import DensityMatrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def apply_dephasing_noise(rho: np.ndarray, gamma: float) -> np.ndarray:
    """
    Apply dephasing (phase damping) noise to a density matrix.

    Dephasing destroys off-diagonal coherences without affecting populations.
    This is a fundamentally different noise model from depolarizing.

    Args:
        rho: Density matrix
        gamma: Dephasing rate (0 = no noise, 1 = complete dephasing)

    Returns:
        Dephased density matrix
    """
    dim = rho.shape[0]
    rho_dephased = rho.copy()

    for i in range(dim):
        for j in range(dim):
            if i != j:
                # Decay rate depends on number of differing qubits
                n_diff = bin(i ^ j).count('1')
                rho_dephased[i, j] *= (1 - gamma) ** n_diff

    return rho_dephased


def apply_amplitude_damping(rho: np.ndarray, gamma: float) -> np.ndarray:
    """
    Apply amplitude damping (T1 decay) noise.

    Drives excited states toward ground state |000⟩.

    Args:
        rho: Density matrix
        gamma: Damping rate

    Returns:
        Damped density matrix
    """
    dim = rho.shape[0]
    ground = np.zeros((dim, dim))
    ground[0, 0] = 1.0

    # Simplified: convex combination toward ground state
    return (1 - gamma) * rho + gamma * ground


def apply_asymmetric_noise(rho: np.ndarray, noise_levels: List[float]) -> np.ndarray:
    """
    Apply different noise levels to different qubits.

    Args:
        rho: 3-qubit density matrix
        noise_levels: [noise_q0, noise_q1, noise_q2]

    Returns:
        Asymmetrically noisy state
    """
    dim = 8
    identity = np.eye(dim) / dim

    # Average noise effect (simplified model)
    avg_noise = np.mean(noise_levels)
    return (1 - avg_noise) * rho + avg_noise * identity


def generate_dephased_ghz(gamma: float) -> DensityMatrix:
    """Generate GHZ state with dephasing noise."""
    psi = np.zeros(8, dtype=complex)
    psi[0] = 1/np.sqrt(2)
    psi[7] = 1/np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    rho_dephased = apply_dephasing_noise(rho, gamma)
    return DensityMatrix(rho_dephased)


def generate_amplitude_damped_ghz(gamma: float) -> DensityMatrix:
    """Generate GHZ state with amplitude damping."""
    psi = np.zeros(8, dtype=complex)
    psi[0] = 1/np.sqrt(2)
    psi[7] = 1/np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    rho_damped = apply_amplitude_damping(rho, gamma)
    return DensityMatrix(rho_damped)


def generate_asymmetric_noisy_ghz(noise_levels: List[float]) -> DensityMatrix:
    """Generate GHZ with asymmetric noise on each qubit."""
    psi = np.zeros(8, dtype=complex)
    psi[0] = 1/np.sqrt(2)
    psi[7] = 1/np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    rho_noisy = apply_asymmetric_noise(rho, noise_levels)
    return DensityMatrix(rho_noisy)


def investigate_dephasing_threshold():
    """
    Find NPT/PPT threshold under dephasing noise.

    Dephasing affects coherences differently than depolarizing,
    potentially creating a different threshold.
    """
    logger.info("="*60)
    logger.info("INVESTIGATING: Dephasing noise threshold")
    logger.info("="*60)

    results = {'gamma': [], 'is_npt': [], 'min_eigenvalue': []}

    for gamma in np.linspace(0, 1, 51):
        state = generate_dephased_ghz(gamma)
        is_npt = check_npt_any_bipartition(state)

        results['gamma'].append(gamma)
        results['is_npt'].append(is_npt)

        logger.info(f"  gamma={gamma:.2f}: NPT={is_npt}")

    # Find threshold
    npt_values = np.array(results['is_npt'])
    if np.any(~npt_values):
        threshold_idx = np.where(~npt_values)[0][0]
        threshold = results['gamma'][threshold_idx]
        logger.info(f"\nDephasing NPT threshold: gamma ≈ {threshold:.2f}")
    else:
        logger.info("\nNo PPT transition found (always NPT)")
        threshold = 1.0

    return {'threshold': threshold, 'details': results}


def investigate_amplitude_damping_threshold():
    """
    Find NPT/PPT threshold under amplitude damping.
    """
    logger.info("="*60)
    logger.info("INVESTIGATING: Amplitude damping threshold")
    logger.info("="*60)

    results = {'gamma': [], 'is_npt': []}

    for gamma in np.linspace(0, 1, 51):
        state = generate_amplitude_damped_ghz(gamma)
        is_npt = check_npt_any_bipartition(state)

        results['gamma'].append(gamma)
        results['is_npt'].append(is_npt)

        logger.info(f"  gamma={gamma:.2f}: NPT={is_npt}")

    # Find threshold
    npt_values = np.array(results['is_npt'])
    if np.any(~npt_values):
        threshold_idx = np.where(~npt_values)[0][0]
        threshold = results['gamma'][threshold_idx]
        logger.info(f"\nAmplitude damping NPT threshold: gamma ≈ {threshold:.2f}")
    else:
        logger.info("\nNo PPT transition found")
        threshold = 1.0

    return {'threshold': threshold, 'details': results}


def investigate_classifier_on_adversarial_noise(n_samples: int = 500, seed: int = 42):
    """
    Test classifier performance on adversarial noise models.

    Trains on standard depolarizing noise, tests on:
    1. Dephasing noise
    2. Amplitude damping
    3. Asymmetric noise

    This tests generalization to unseen noise types.
    """
    logger.info("="*60)
    logger.info("INVESTIGATING: Classifier generalization to adversarial noise")
    logger.info("="*60)

    np.random.seed(seed)

    # Generate training data with standard depolarizing noise
    logger.info("\n[1/4] Generating training data (depolarizing noise)...")

    train_states = []
    train_labels = []

    for _ in range(n_samples // 2):
        noise = np.random.uniform(0, 0.5)
        state = generate_entangled_state(3, 'ghz', noise_level=noise)
        train_states.append(state)
        train_labels.append(1 if check_npt_any_bipartition(state) else 0)

    for _ in range(n_samples // 2):
        state = generate_3qubit_product_state()
        train_states.append(state)
        train_labels.append(0)

    train_labels = np.array(train_labels)

    # Extract features
    basis_36d = create_sparse_measurement_set(3, 'two_body')
    train_features = extract_features_batch(train_states, basis_36d, verbose=False)

    # Train classifier
    logger.info("\n[2/4] Training classifier...")
    clf = SVC(kernel='linear', C=1.0, random_state=seed)
    clf.fit(train_features, train_labels)

    train_acc = accuracy_score(train_labels, clf.predict(train_features))
    logger.info(f"  Training accuracy: {train_acc:.3f}")

    results = {'train_accuracy': train_acc, 'adversarial_tests': {}}

    # Test on adversarial noise models
    logger.info("\n[3/4] Testing on adversarial noise models...")

    # Test 1: Dephasing noise
    logger.info("\n  Testing dephasing noise...")
    dephasing_states = []
    dephasing_labels = []
    for _ in range(100):
        gamma = np.random.uniform(0, 0.8)
        state = generate_dephased_ghz(gamma)
        dephasing_states.append(state)
        dephasing_labels.append(1 if check_npt_any_bipartition(state) else 0)

    dephasing_features = extract_features_batch(dephasing_states, basis_36d, verbose=False)
    dephasing_pred = clf.predict(dephasing_features)
    dephasing_acc = accuracy_score(dephasing_labels, dephasing_pred)
    logger.info(f"    Dephasing accuracy: {dephasing_acc:.3f}")
    results['adversarial_tests']['dephasing'] = {
        'accuracy': float(dephasing_acc),
        'n_samples': len(dephasing_labels)
    }

    # Test 2: Amplitude damping
    logger.info("\n  Testing amplitude damping...")
    damping_states = []
    damping_labels = []
    for _ in range(100):
        gamma = np.random.uniform(0, 0.8)
        state = generate_amplitude_damped_ghz(gamma)
        damping_states.append(state)
        damping_labels.append(1 if check_npt_any_bipartition(state) else 0)

    damping_features = extract_features_batch(damping_states, basis_36d, verbose=False)
    damping_pred = clf.predict(damping_features)
    damping_acc = accuracy_score(damping_labels, damping_pred)
    logger.info(f"    Amplitude damping accuracy: {damping_acc:.3f}")
    results['adversarial_tests']['amplitude_damping'] = {
        'accuracy': float(damping_acc),
        'n_samples': len(damping_labels)
    }

    # Test 3: Asymmetric noise
    logger.info("\n  Testing asymmetric noise...")
    asymmetric_states = []
    asymmetric_labels = []
    for _ in range(100):
        noise_levels = [np.random.uniform(0, 0.6) for _ in range(3)]
        state = generate_asymmetric_noisy_ghz(noise_levels)
        asymmetric_states.append(state)
        asymmetric_labels.append(1 if check_npt_any_bipartition(state) else 0)

    asymmetric_features = extract_features_batch(asymmetric_states, basis_36d, verbose=False)
    asymmetric_pred = clf.predict(asymmetric_features)
    asymmetric_acc = accuracy_score(asymmetric_labels, asymmetric_pred)
    logger.info(f"    Asymmetric noise accuracy: {asymmetric_acc:.3f}")
    results['adversarial_tests']['asymmetric'] = {
        'accuracy': float(asymmetric_acc),
        'n_samples': len(asymmetric_labels)
    }

    # Summary
    logger.info("\n[4/4] Summary:")
    logger.info(f"  Training (depolarizing): {train_acc:.3f}")
    logger.info(f"  Dephasing:               {dephasing_acc:.3f}")
    logger.info(f"  Amplitude damping:       {damping_acc:.3f}")
    logger.info(f"  Asymmetric:              {asymmetric_acc:.3f}")

    # Check for negative result
    min_adversarial = min(dephasing_acc, damping_acc, asymmetric_acc)
    if min_adversarial < 0.7:
        logger.info("\n  *** POTENTIAL NEGATIVE RESULT: Classifier fails on adversarial noise ***")
        results['negative_result_found'] = True
    else:
        logger.info("\n  Classifier generalizes reasonably to adversarial noise")
        results['negative_result_found'] = False

    return results


def investigate_36d_vs_63d_boundary(n_samples: int = 200, seed: int = 42):
    """
    Compare 36D vs 63D features specifically on boundary states.

    If 63D significantly outperforms 36D near the boundary,
    this suggests 3-body correlations are important for hard cases.
    """
    logger.info("="*60)
    logger.info("INVESTIGATING: 36D vs 63D on boundary states")
    logger.info("="*60)

    np.random.seed(seed)

    # Generate states near the NPT/PPT boundary (noise 0.75-0.85)
    logger.info("\n[1/3] Generating boundary states...")

    states = []
    labels = []

    # GHZ near boundary
    for _ in range(n_samples // 3):
        noise = np.random.uniform(0.75, 0.85)
        state = generate_entangled_state(3, 'ghz', noise_level=noise)
        states.append(state)
        labels.append(1 if check_npt_any_bipartition(state) else 0)

    # W near boundary
    for _ in range(n_samples // 3):
        noise = np.random.uniform(0.75, 0.85)
        state = generate_entangled_state(3, 'w', noise_level=noise)
        states.append(state)
        labels.append(1 if check_npt_any_bipartition(state) else 0)

    # Cluster near boundary
    for _ in range(n_samples // 3):
        noise = np.random.uniform(0.75, 0.85)
        state = generate_noisy_cluster_state(3, noise_level=noise)
        states.append(state)
        labels.append(1 if check_npt_any_bipartition(state) else 0)

    labels = np.array(labels)

    logger.info(f"  Generated {len(states)} boundary states")
    logger.info(f"  Distillable: {np.sum(labels)} ({100*np.mean(labels):.1f}%)")

    # Extract features
    logger.info("\n[2/3] Extracting features...")
    basis_36d = create_sparse_measurement_set(3, 'two_body')
    basis_63d = get_pauli_basis(3, include_identity=False)

    features_36d = extract_features_batch(states, basis_36d, verbose=False)
    features_63d = extract_features_batch(states, basis_63d, verbose=False)

    # Train and evaluate
    logger.info("\n[3/3] Training and evaluating...")

    X_train_36d, X_test_36d, y_train, y_test = train_test_split(
        features_36d, labels, test_size=0.3, random_state=seed, stratify=labels
    )
    X_train_63d, X_test_63d, _, _ = train_test_split(
        features_63d, labels, test_size=0.3, random_state=seed, stratify=labels
    )

    clf_36d = SVC(kernel='linear', C=1.0, random_state=seed)
    clf_36d.fit(X_train_36d, y_train)
    acc_36d = accuracy_score(y_test, clf_36d.predict(X_test_36d))

    clf_63d = SVC(kernel='linear', C=1.0, random_state=seed)
    clf_63d.fit(X_train_63d, y_train)
    acc_63d = accuracy_score(y_test, clf_63d.predict(X_test_63d))

    logger.info(f"\n  36D boundary accuracy: {acc_36d:.3f}")
    logger.info(f"  63D boundary accuracy: {acc_63d:.3f}")
    logger.info(f"  Gap: {acc_63d - acc_36d:.3f}")

    results = {
        'accuracy_36d': float(acc_36d),
        'accuracy_63d': float(acc_63d),
        'gap': float(acc_63d - acc_36d),
        'n_samples': len(states),
        'distillable_fraction': float(np.mean(labels))
    }

    if acc_63d - acc_36d > 0.05:
        logger.info("\n  *** POTENTIAL NEGATIVE RESULT: 63D outperforms 36D on boundary ***")
        logger.info("  This suggests 3-body correlations help near the boundary.")
        results['negative_result_found'] = True
    else:
        logger.info("\n  No significant advantage from 3-body correlations on boundary")
        results['negative_result_found'] = False

    return results


def main():
    parser = argparse.ArgumentParser(description='Investigate negative results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    all_results = {}

    # Investigation 1: Dephasing threshold
    all_results['dephasing_threshold'] = investigate_dephasing_threshold()

    # Investigation 2: Amplitude damping threshold
    all_results['amplitude_damping_threshold'] = investigate_amplitude_damping_threshold()

    # Investigation 3: Adversarial noise generalization
    all_results['adversarial_noise'] = investigate_classifier_on_adversarial_noise(seed=args.seed)

    # Investigation 4: Boundary state comparison
    all_results['boundary_comparison'] = investigate_36d_vs_63d_boundary(seed=args.seed)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("INVESTIGATION SUMMARY")
    logger.info("="*60)

    negative_results = []

    if all_results['adversarial_noise'].get('negative_result_found'):
        negative_results.append("Classifier fails on adversarial noise")

    if all_results['boundary_comparison'].get('negative_result_found'):
        negative_results.append("63D outperforms 36D on boundary states")

    if negative_results:
        logger.info("\nPOTENTIAL NEGATIVE RESULTS FOUND:")
        for nr in negative_results:
            logger.info(f"  - {nr}")
    else:
        logger.info("\nNo strong negative results found.")
        logger.info("36D restricted features appear robust to tested scenarios.")

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2, default=float)
        logger.info(f"\nResults saved to {args.output}")

    return all_results


if __name__ == '__main__':
    main()
