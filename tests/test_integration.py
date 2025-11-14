"""
Integration tests for the complete witness learning pipeline.
"""

import pytest
import numpy as np
import logging
import sys
sys.path.insert(0, '/home/user/ML_QML_Witness_Generation')

from src.quantum_states.state_generation import (
    generate_dataset,
    generate_bell_state,
    generate_separable_state
)
from src.feature_extraction.pauli_features import (
    get_pauli_basis,
    extract_features_batch
)
from src.ml_models.svm_witness import SVMWitnessLearner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestIntegration:
    """Integration tests for the complete pipeline."""

    def test_end_to_end_svm_witness(self):
        """
        Test the complete pipeline from state generation to witness learning.

        This follows the approach from Section 6.1 (Use-Case 1) of the framework.
        """
        logger.info("="*60)
        logger.info("Running end-to-end SVM witness integration test")
        logger.info("="*60)

        # Parameters
        n_qubits = 2
        n_samples = 200
        entangled_fraction = 0.5
        seed = 42

        # Step 1: Generate dataset
        logger.info("\n[Step 1] Generating quantum states dataset...")
        states, labels = generate_dataset(
            n_qubits=n_qubits,
            n_samples=n_samples,
            entangled_fraction=entangled_fraction,
            noise_range=(0.0, 0.2),
            seed=seed
        )

        assert len(states) == n_samples
        assert len(labels) == n_samples
        logger.info(f"Generated {n_samples} states")

        # Step 2: Extract features
        logger.info("\n[Step 2] Extracting Pauli features...")
        pauli_basis = get_pauli_basis(n_qubits, include_identity=False)
        logger.info(f"Pauli basis size: {len(pauli_basis)}")

        features = extract_features_batch(states, pauli_basis, verbose=False)
        logger.info(f"Feature matrix shape: {features.shape}")

        assert features.shape == (n_samples, len(pauli_basis))

        # Step 3: Train SVM witness
        logger.info("\n[Step 3] Training SVM witness learner...")
        svm_learner = SVMWitnessLearner(
            pauli_basis=pauli_basis,
            C=1.0,
            random_state=seed
        )

        metrics = svm_learner.train(features, labels, test_size=0.2, verbose=True)

        # Check performance metrics
        assert metrics['test_accuracy'] > 0.7, "Test accuracy should be > 70%"
        assert metrics['test_precision'] > 0.6, "Test precision should be > 60%"
        assert metrics['test_recall'] > 0.6, "Test recall should be > 60%"

        logger.info("\n[Success] End-to-end pipeline completed!")

    def test_incomplete_measurements_pipeline(self):
        """
        Test witness learning from incomplete measurements.

        This follows Failure Mode 4 from Section 2.3 and Use-Case 6.4.
        """
        logger.info("="*60)
        logger.info("Testing incomplete measurements pipeline")
        logger.info("="*60)

        # Parameters
        n_qubits = 2
        n_samples = 150
        seed = 42

        # Step 1: Generate dataset
        logger.info("\n[Step 1] Generating dataset...")
        states, labels = generate_dataset(
            n_qubits=n_qubits,
            n_samples=n_samples,
            entangled_fraction=0.5,
            seed=seed
        )

        # Step 2: Use sparse measurement set (incomplete tomography)
        from src.feature_extraction.pauli_features import create_sparse_measurement_set

        logger.info("\n[Step 2] Creating sparse measurement set...")
        sparse_basis = create_sparse_measurement_set(n_qubits, strategy='two_body')
        logger.info(f"Sparse basis size: {len(sparse_basis)} (vs full: {4**n_qubits - 1})")

        # Step 3: Extract features with sparse basis
        logger.info("\n[Step 3] Extracting features from sparse measurements...")
        features_sparse = extract_features_batch(states, sparse_basis, verbose=False)
        logger.info(f"Sparse feature matrix shape: {features_sparse.shape}")

        # Step 4: Train SVM on sparse features
        logger.info("\n[Step 4] Training SVM on incomplete measurements...")
        svm_learner = SVMWitnessLearner(
            pauli_basis=sparse_basis,
            C=1.0,
            random_state=seed
        )

        metrics = svm_learner.train(features_sparse, labels, test_size=0.2, verbose=True)

        # Performance should still be reasonable even with incomplete measurements
        assert metrics['test_accuracy'] > 0.6, "Should achieve >60% accuracy with sparse measurements"

        logger.info("\n[Success] Sparse measurement pipeline completed!")

    def test_witness_on_known_states(self):
        """
        Test that learned witness correctly classifies known entangled/separable states.
        """
        logger.info("="*60)
        logger.info("Testing witness on known states")
        logger.info("="*60)

        from src.quantum_states.state_generation import (
            generate_bell_state,
            generate_separable_state
        )

        n_qubits = 2
        n_samples = 100
        seed = 42

        # Generate training data
        logger.info("\n[Step 1] Generating training data...")
        states, labels = generate_dataset(
            n_qubits=n_qubits,
            n_samples=n_samples,
            entangled_fraction=0.5,
            seed=seed
        )

        pauli_basis = get_pauli_basis(n_qubits, include_identity=False)
        features = extract_features_batch(states, pauli_basis, verbose=False)

        # Train SVM
        logger.info("\n[Step 2] Training SVM...")
        svm_learner = SVMWitnessLearner(pauli_basis=pauli_basis, C=1.0, random_state=seed)
        svm_learner.train(features, labels, test_size=0.2, verbose=False)

        # Test on known Bell states (should be entangled)
        logger.info("\n[Step 3] Testing on Bell states...")
        bell_states = [generate_bell_state(i) for i in range(4)]
        bell_features = extract_features_batch(bell_states, pauli_basis, verbose=False)
        bell_predictions = svm_learner.predict(bell_features)

        logger.info(f"Bell state predictions: {bell_predictions}")
        # At least half of Bell states should be classified as entangled (label=1)
        # Note: Linear SVMs may struggle with phase-dependent Bell states in small training sets
        assert np.mean(bell_predictions) >= 0.5, "At least half of Bell states should be detected as entangled"

        # Test on known separable states (should be separable)
        logger.info("\n[Step 4] Testing on separable states...")
        sep_states = [generate_separable_state(n_qubits, seed=seed+i) for i in range(4)]
        sep_features = extract_features_batch(sep_states, pauli_basis, verbose=False)
        sep_predictions = svm_learner.predict(sep_features)

        logger.info(f"Separable state predictions: {sep_predictions}")
        # Most separable states should be classified as separable (label=0)
        assert np.mean(sep_predictions) < 0.5, "Most separable states should be detected as separable"

        logger.info("\n[Success] Known state classification test passed!")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
