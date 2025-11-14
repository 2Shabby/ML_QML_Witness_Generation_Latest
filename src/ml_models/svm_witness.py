"""
SVM Witness Learner

Implements linear SVM-based witness learning following Section 4.1 and 3.3.1
of the framework document.

The learned hyperplane w·x + b = 0 directly maps to a witness operator
W = Σₖ wₖ Pₖ where Pₖ are Pauli basis operators.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from qiskit.quantum_info import PauliList, SparsePauliOp
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SVMWitnessLearner:
    """
    Linear SVM for learning entanglement witnesses.

    Following the framework's approach where:
    - Decision function: f(x) = w·x + b
    - Witness operator: W = Σₖ wₖ Pₖ
    - Witness condition: Tr(Wρ) < -b indicates entanglement
    """

    def __init__(
        self,
        pauli_basis: PauliList,
        C: float = 1.0,
        l1_ratio: float = 0.0,
        kernel: str = 'linear',
        random_state: Optional[int] = None
    ):
        """
        Initialize SVM witness learner.

        Args:
            pauli_basis: Pauli basis used for feature extraction
            C: Regularization parameter (smaller = more regularization)
            l1_ratio: L1 regularization ratio for sparse witnesses (0.0 = L2 only)
            kernel: SVM kernel ('linear' for interpretable witnesses)
            random_state: Random seed for reproducibility
        """
        self.pauli_basis = pauli_basis
        self.C = C
        self.l1_ratio = l1_ratio
        self.kernel = kernel
        self.random_state = random_state

        # Initialize SVM
        if kernel == 'linear':
            self.svm = SVC(
                kernel='linear',
                C=C,
                random_state=random_state,
                probability=True
            )
        else:
            self.svm = SVC(
                kernel=kernel,
                C=C,
                random_state=random_state,
                probability=True
            )

        self.is_trained = False
        self.witness_operator = None
        self.bias = None
        self.metrics = {}

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train the SVM on labeled feature data.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Labels (0=separable, 1=entangled)
            test_size: Fraction of data for testing
            verbose: Whether to log progress

        Returns:
            Dictionary of performance metrics
        """
        if verbose:
            logger.info("Training SVM witness learner...")
            logger.info(f"Training samples: {len(X)}, Features: {X.shape[1]}")

        # Split data
        # Use a different seed for split to avoid bad luck with specific random states
        split_seed = self.random_state + 1000 if self.random_state is not None else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=split_seed, stratify=y
        )

        # Train SVM
        self.svm.fit(X_train, y_train)
        self.is_trained = True

        # Extract witness operator (for linear kernel)
        if self.kernel == 'linear':
            self._extract_witness_operator()

        # Evaluate
        y_pred_train = self.svm.predict(X_train)
        y_pred_test = self.svm.predict(X_test)

        self.metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'train_precision': precision_score(y_train, y_pred_train, zero_division=0),
            'test_precision': precision_score(y_test, y_pred_test, zero_division=0),
            'train_recall': recall_score(y_train, y_pred_train, zero_division=0),
            'test_recall': recall_score(y_test, y_pred_test, zero_division=0),
            'n_support_vectors': len(self.svm.support_),
        }

        if verbose:
            logger.info("Training complete!")
            logger.info(f"Test Accuracy: {self.metrics['test_accuracy']:.4f}")
            logger.info(f"Test Precision: {self.metrics['test_precision']:.4f}")
            logger.info(f"Test Recall: {self.metrics['test_recall']:.4f}")
            logger.info(f"Support Vectors: {self.metrics['n_support_vectors']}")

        return self.metrics

    def _extract_witness_operator(self):
        """
        Extract the witness operator W from the trained linear SVM.

        Following Section 3.3.1:
        W = Σₖ wₖ Pₖ where w is the SVM hyperplane vector
        """
        if not self.is_trained:
            raise RuntimeError("SVM must be trained before extracting witness")

        if self.kernel != 'linear':
            raise ValueError("Can only extract witness from linear kernel SVM")

        # Get hyperplane coefficients
        w = self.svm.coef_[0]  # Shape: (n_features,)
        self.bias = self.svm.intercept_[0]

        # Build witness operator as weighted sum of Pauli operators
        # W = Σₖ wₖ Pₖ
        pauli_strings = []
        coefficients = []

        for i, (pauli, weight) in enumerate(zip(self.pauli_basis, w)):
            if abs(weight) > 1e-10:  # Only include non-zero terms
                pauli_strings.append(str(pauli))
                coefficients.append(weight)

        # Create SparsePauliOp
        if len(pauli_strings) > 0:
            self.witness_operator = SparsePauliOp(
                pauli_strings,
                coeffs=coefficients
            )
        else:
            logger.warning("All witness coefficients are zero!")

        logger.info(f"Witness operator extracted: {len(pauli_strings)} non-zero terms")

        return self.witness_operator

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for feature vectors.

        Args:
            X: Feature matrix

        Returns:
            Predicted labels (0=separable, 1=entangled)
        """
        if not self.is_trained:
            raise RuntimeError("SVM must be trained before prediction")

        return self.svm.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix

        Returns:
            Probability matrix of shape (n_samples, 2)
        """
        if not self.is_trained:
            raise RuntimeError("SVM must be trained before prediction")

        return self.svm.predict_proba(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function f(x) = w·x + b.

        For witness interpretation: Tr(Wρ) + b where b is the bias.

        Args:
            X: Feature matrix

        Returns:
            Decision function values
        """
        if not self.is_trained:
            raise RuntimeError("SVM must be trained before computing decision function")

        return self.svm.decision_function(X)

    def get_witness_operator(self) -> SparsePauliOp:
        """
        Get the extracted witness operator.

        Returns:
            Witness operator as SparsePauliOp
        """
        if self.witness_operator is None:
            if not self.is_trained:
                raise RuntimeError("SVM must be trained first")
            self._extract_witness_operator()

        return self.witness_operator

    def get_sparse_witness(self, threshold: float = 0.01) -> SparsePauliOp:
        """
        Get a sparse version of the witness by thresholding small coefficients.

        Following Section 8.2 for minimal measurement sets.

        Args:
            threshold: Coefficients with |w_k| < threshold are set to zero

        Returns:
            Sparse witness operator
        """
        if self.witness_operator is None:
            self._extract_witness_operator()

        # Filter terms by coefficient magnitude
        pauli_strings = []
        coefficients = []

        for pauli, coeff in self.witness_operator.to_list():
            if abs(coeff) >= threshold:
                pauli_strings.append(pauli)
                coefficients.append(coeff)

        logger.info(f"Sparse witness: {len(pauli_strings)} terms (threshold={threshold})")

        return SparsePauliOp(pauli_strings, coeffs=coefficients)

    def evaluate_witness(
        self,
        rho_data: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate witness performance on density matrices.

        Following Section 7.1 and 7.2 evaluation metrics.

        Args:
            rho_data: Array of density matrices
            labels: True labels (0=separable, 1=entangled)

        Returns:
            Dictionary of witness-specific metrics
        """
        if self.witness_operator is None:
            raise RuntimeError("Witness operator not available")

        # Compute witness expectation values
        witness_values = []
        for rho in rho_data:
            exp_val = np.trace(rho @ self.witness_operator.to_matrix()).real
            witness_values.append(exp_val)

        witness_values = np.array(witness_values)

        # Classify based on witness: Tr(Wρ) + bias < 0 => entangled
        predicted = (witness_values + self.bias) < 0

        # Compute metrics
        metrics = {
            'witness_accuracy': accuracy_score(labels, predicted),
            'witness_precision': precision_score(labels, predicted, zero_division=0),
            'witness_recall': recall_score(labels, predicted, zero_division=0),
            'mean_violation_entangled': witness_values[labels == 1].mean(),
            'mean_violation_separable': witness_values[labels == 0].mean(),
        }

        return metrics

    def get_measurement_cost(self) -> int:
        """
        Estimate the number of measurement settings required.

        Following Section 7.3 efficiency metrics.

        Returns:
            Number of measurement settings
        """
        from ..feature_extraction.pauli_features import estimate_measurement_cost

        if self.witness_operator is None:
            raise RuntimeError("Witness operator not available")

        # Convert to PauliList
        pauli_list = PauliList([pauli for pauli, _ in self.witness_operator.to_list()])

        return estimate_measurement_cost(pauli_list)
