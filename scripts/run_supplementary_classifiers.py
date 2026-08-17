#!/usr/bin/env python
"""
Run supplementary classifier experiments (Random Forest, XGBoost/GradientBoosting).

Validates the manuscript footnote claiming RF: 99.8%, XGBoost: 99.9% accuracy
under identical conditions to the main pipeline.
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

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

from src.quantum_states.state_generation import generate_distillability_dataset
from src.feature_extraction.pauli_features import (
    create_sparse_measurement_set,
    extract_features_batch,
)
from src.config import DEFAULT_SEEDS

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def run_supplementary_experiments(
    n_samples: int = 5000,
    seeds: list = None,
    test_size: float = 0.2,
    output_dir: str = None,
):
    """Run RF and GradientBoosting experiments matching manuscript conditions."""
    if seeds is None:
        seeds = DEFAULT_SEEDS
    if output_dir is None:
        output_dir = str(project_root / 'results')

    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for seed in seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"Seed: {seed}")
        logger.info(f"{'='*60}")

        # Generate dataset (same as main pipeline)
        states, labels = generate_distillability_dataset(
            n_samples=n_samples, seed=seed
        )
        labels = np.array(labels)

        # Extract 36D Pauli features (same as SVM/MLP/Transformer)
        basis = create_sparse_measurement_set(3, 'two_body')
        features = extract_features_batch(states, basis, verbose=False)

        # Split (same seed derivation as other learners)
        split_seed = seed + 1000 if seed else None
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size,
            random_state=split_seed, stratify=labels
        )

        logger.info(
            f"Dataset: {features.shape[0]} samples, "
            f"{features.shape[1]}D features, "
            f"distillable: {labels.sum()}/{len(labels)}"
        )

        seed_results = {}

        # Random Forest
        logger.info("\n--- Random Forest ---")
        rf = RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)

        rf_metrics = {
            'accuracy': accuracy_score(y_test, rf_pred),
            'precision': precision_score(y_test, rf_pred, zero_division=0),
            'recall': recall_score(y_test, rf_pred, zero_division=0),
            'f1': f1_score(y_test, rf_pred, zero_division=0),
        }
        seed_results['random_forest'] = rf_metrics
        logger.info(f"  Accuracy:  {rf_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {rf_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {rf_metrics['recall']:.4f}")
        logger.info(f"  F1:        {rf_metrics['f1']:.4f}")

        # Cross-validation for RF
        rf_cv = cross_val_score(
            RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
            features, labels, cv=5, scoring='accuracy'
        )
        seed_results['random_forest']['cv_mean'] = rf_cv.mean()
        seed_results['random_forest']['cv_std'] = rf_cv.std()
        logger.info(f"  CV accuracy: {rf_cv.mean():.4f} +/- {rf_cv.std():.4f}")

        # Gradient Boosting (XGBoost-equivalent in sklearn)
        logger.info("\n--- Gradient Boosting (XGBoost-equivalent) ---")
        gb = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3,
            random_state=seed
        )
        gb.fit(X_train, y_train)
        gb_pred = gb.predict(X_test)

        gb_metrics = {
            'accuracy': accuracy_score(y_test, gb_pred),
            'precision': precision_score(y_test, gb_pred, zero_division=0),
            'recall': recall_score(y_test, gb_pred, zero_division=0),
            'f1': f1_score(y_test, gb_pred, zero_division=0),
        }
        seed_results['gradient_boosting'] = gb_metrics
        logger.info(f"  Accuracy:  {gb_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {gb_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {gb_metrics['recall']:.4f}")
        logger.info(f"  F1:        {gb_metrics['f1']:.4f}")

        # Cross-validation for GB
        gb_cv = cross_val_score(
            GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=3,
                random_state=seed
            ),
            features, labels, cv=5, scoring='accuracy'
        )
        seed_results['gradient_boosting']['cv_mean'] = gb_cv.mean()
        seed_results['gradient_boosting']['cv_std'] = gb_cv.std()
        logger.info(f"  CV accuracy: {gb_cv.mean():.4f} +/- {gb_cv.std():.4f}")

        all_results[str(seed)] = seed_results

    # Aggregate across seeds
    logger.info(f"\n{'='*60}")
    logger.info("AGGREGATE RESULTS (mean +/- std across seeds)")
    logger.info(f"{'='*60}")

    for model_name in ['random_forest', 'gradient_boosting']:
        accs = [all_results[s][model_name]['accuracy'] for s in all_results]
        logger.info(
            f"{model_name}: {np.mean(accs):.4f} +/- {np.std(accs):.4f}"
        )

    # Save results
    output_path = os.path.join(output_dir, 'supplementary_classifiers.json')
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'n_samples': n_samples,
            'seeds': seeds,
            'results': all_results,
        }, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    return all_results


if __name__ == '__main__':
    run_supplementary_experiments()
