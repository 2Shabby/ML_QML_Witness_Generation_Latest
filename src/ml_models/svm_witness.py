"""
SVM-based witness learning.

This module implements the linear SVM approach for learning entanglement witnesses
from Pauli feature vectors.
"""

import numpy as np
from typing import Dict, Optional
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from qiskit.quantum_info import PauliList, DensityMatrix
import logging

logger = logging.getLogger(__name__)


class SVMWitnessLearner:
    """
    SVM-based entanglement witness learner.

    This class implements the linear SVM approach for learning witnesses,
    where the hyperplane coefficients directly correspond to the witness operator.
    """

    def __init__(self, pauli_basis: PauliList, C: float = 1.0, random_state: int = None):
        """
        Initialize the SVM witness learner.

        Args:
            pauli_basis: PauliList of operators used for features
            C: SVM regularization parameter
            random_state: Random seed for reproducibility
        """
        self.pauli_basis = pauli_basis
        self.C = C
        self.random_state = random_state
        self.model = None
        self.witness_coefficients = None

    def train(self, features: np.ndarray, labels: np.ndarray,
              test_size: float = 0.2, verbose: bool = False) -> Dict:
        """
        Train the SVM witness learner.

        Args:
            features: Feature matrix of shape (n_samples, n_features)
            labels: Binary labels (1 for entangled, 0 for separable)
            test_size: Fraction of data for testing
            verbose: Whether to print training info

        Returns:
            Dictionary of training metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=self.random_state
        )

        if verbose:
            logger.info(f"Training SVM on {len(X_train)} samples...")
            logger.info(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")

        # Train linear SVM
        self.model = SVC(kernel='linear', C=self.C, random_state=self.random_state)
        self.model.fit(X_train, y_train)

        # Extract witness coefficients
        self.witness_coefficients = self.model.coef_[0]

        # Evaluate
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'train_precision': precision_score(y_train, y_train_pred, zero_division=0),
            'test_precision': precision_score(y_test, y_test_pred, zero_division=0),
            'train_recall': recall_score(y_train, y_train_pred, zero_division=0),
            'test_recall': recall_score(y_test, y_test_pred, zero_division=0),
            'train_f1': f1_score(y_train, y_train_pred, zero_division=0),
            'test_f1': f1_score(y_test, y_test_pred, zero_division=0),
        }

        if verbose:
            logger.info(f"Training accuracy: {metrics['train_accuracy']:.3f}")
            logger.info(f"Test accuracy: {metrics['test_accuracy']:.3f}")
            logger.info(f"Test precision: {metrics['test_precision']:.3f}")
            logger.info(f"Test recall: {metrics['test_recall']:.3f}")
            logger.info(f"Test F1: {metrics['test_f1']:.3f}")

        return metrics

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict labels for new states.

        Args:
            features: Feature matrix

        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        return self.model.predict(features)

    def get_witness_operator(self) -> Dict:
        """
        Get the learned witness operator.

        Returns:
            Dictionary with witness coefficients and Pauli operators
        """
        if self.witness_coefficients is None:
            raise ValueError("Model not trained yet!")

        return {
            'coefficients': self.witness_coefficients,
            'paulis': self.pauli_basis,
            'bias': self.model.intercept_[0]
        }

    def evaluate_witness(self, state: DensityMatrix) -> float:
        """
        Evaluate the witness operator on a state: Tr(W ρ).

        Args:
            state: Density matrix

        Returns:
            Witness expectation value
        """
        from src.feature_extraction.pauli_features import extract_features_batch

        if self.witness_coefficients is None:
            raise ValueError("Model not trained yet!")

        # Extract features
        features = extract_features_batch([state], self.pauli_basis, verbose=False)

        # Compute Tr(W ρ) = w · x + b
        return np.dot(self.witness_coefficients, features[0]) + self.model.intercept_[0]
