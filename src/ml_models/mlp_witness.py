"""
MLP Witness Learner

Implements Multi-Layer Perceptron for nonlinear witness learning.
Following Section 4.3 of the framework document.

The MLP learns a nonlinear witness functional W[ρ] ≡ f_θ(x_ρ)
where x_ρ is the Bloch vector representation.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# TensorFlow imports (with graceful degradation)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks, optimizers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. MLP witness learner will not work.")


class MLPWitnessLearner:
    """
    Multi-Layer Perceptron for learning nonlinear entanglement witnesses.

    Unlike linear SVM, MLP can learn complex decision boundaries
    for tasks like incomplete measurements (Failure Mode 4).

    Attributes:
        model: Keras model
        history: Training history
        config: Model configuration
    """

    def __init__(
        self,
        n_features: int,
        hidden_layers: List[int] = [128, 64, 32],
        activation: str = 'relu',
        dropout_rate: float = 0.3,
        l2_reg: float = 0.01,
        learning_rate: float = 0.001,
        optimizer: str = 'adam',
        random_state: Optional[int] = None
    ):
        """
        Initialize MLP witness learner.

        Args:
            n_features: Number of input features (Pauli expectations)
            hidden_layers: List of hidden layer sizes
            activation: Activation function ('relu', 'tanh', 'elu')
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization strength
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer type ('adam', 'sgd', 'rmsprop')
            random_state: Random seed for reproducibility
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for MLP witness learner")

        self.n_features = n_features
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.random_state = random_state

        # Set random seeds
        if random_state is not None:
            tf.random.set_seed(random_state)
            np.random.seed(random_state)

        # Build model
        self.model = self._build_model()
        self.history = None

        logger.info(
            f"MLPWitnessLearner initialized: "
            f"layers={hidden_layers}, activation={activation}"
        )

    def _build_model(self) -> keras.Model:
        """
        Build Keras MLP model.

        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=(self.n_features,), name='pauli_features')

        # Hidden layers
        x = inputs
        for i, units in enumerate(self.hidden_layers):
            x = layers.Dense(
                units,
                activation=self.activation,
                kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                name=f'hidden_{i+1}'
            )(x)

            if self.dropout_rate > 0:
                x = layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}')(x)

        # Output layer (binary classification)
        outputs = layers.Dense(
            1,
            activation='sigmoid',
            name='entanglement_prob'
        )(x)

        # Create model
        model = models.Model(inputs=inputs, outputs=outputs, name='MLP_Witness')

        # Optimizer
        if self.optimizer_name == 'adam':
            opt = optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'sgd':
            opt = optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9)
        elif self.optimizer_name == 'rmsprop':
            opt = optimizers.RMSprop(learning_rate=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        # Compile model
        model.compile(
            optimizer=opt,
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )

        return model

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping: bool = True,
        patience: int = 15,
        verbose: int = 1,
        callbacks_list: Optional[List[callbacks.Callback]] = None
    ) -> Dict[str, Any]:
        """
        Train MLP on labeled feature data.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Labels (0=separable, 1=entangled)
            validation_split: Fraction of data for validation
            batch_size: Training batch size
            epochs: Maximum number of epochs
            early_stopping: Whether to use early stopping
            patience: Early stopping patience
            verbose: Verbosity level (0=silent, 1=progress, 2=epoch)
            callbacks_list: Optional list of additional callbacks

        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training MLP on {len(X)} samples with {X.shape[1]} features")

        # Prepare callbacks
        cb = callbacks_list or []

        if early_stopping:
            cb.append(callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ))

        # Learning rate reduction
        cb.append(callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=max(5, patience // 3),
            min_lr=1e-6,
            verbose=1
        ))

        # Train model
        self.history = self.model.fit(
            X, y,
            validation_split=validation_split,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=cb,
            verbose=verbose
        )

        # Extract final metrics
        final_metrics = {
            'train_loss': float(self.history.history['loss'][-1]),
            'train_accuracy': float(self.history.history['accuracy'][-1]),
            'val_loss': float(self.history.history['val_loss'][-1]),
            'val_accuracy': float(self.history.history['val_accuracy'][-1]),
            'epochs_trained': len(self.history.history['loss'])
        }

        logger.info(
            f"Training complete: "
            f"val_accuracy={final_metrics['val_accuracy']:.4f}, "
            f"epochs={final_metrics['epochs_trained']}"
        )

        return final_metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix

        Returns:
            Binary predictions (0 or 1)
        """
        probs = self.predict_proba(X)
        return (probs > 0.5).astype(int).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities.

        Args:
            X: Feature matrix

        Returns:
            Probabilities of entanglement
        """
        return self.model.predict(X, verbose=0)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of metrics
        """
        # Get predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        # Compute metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score
        )

        metrics = {
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_precision': precision_score(y_test, y_pred, zero_division=0),
            'test_recall': recall_score(y_test, y_pred, zero_division=0),
            'test_f1': f1_score(y_test, y_pred, zero_division=0),
            'test_auc': roc_auc_score(y_test, y_proba)
        }

        logger.info(f"Test evaluation: accuracy={metrics['test_accuracy']:.4f}")

        return metrics

    def get_witness_functional(self) -> keras.Model:
        """
        Get the witness functional (the trained model itself).

        For MLP, the witness is the nonlinear function W[ρ] ≡ f_θ(x_ρ).
        There is no single operator W; the entire network is the witness.

        Returns:
            Trained Keras model
        """
        return self.model

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function values.

        For MLP: logit(p) where p is predicted probability.

        Args:
            X: Feature matrix

        Returns:
            Decision function values
        """
        probs = self.predict_proba(X)
        # Convert probability to logit
        eps = 1e-7
        probs = np.clip(probs, eps, 1 - eps)
        return np.log(probs / (1 - probs))

    def save_model(self, save_path: Path) -> None:
        """
        Save trained model.

        Args:
            save_path: Path to save model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(save_path)
        logger.info(f"Saved MLP model to {save_path}")

    @classmethod
    def load_model(cls, model_path: Path) -> 'MLPWitnessLearner':
        """
        Load saved model.

        Args:
            model_path: Path to saved model

        Returns:
            Loaded MLPWitnessLearner
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required to load MLP model")

        loaded_model = keras.models.load_model(model_path)

        # Create instance
        instance = cls.__new__(cls)
        instance.model = loaded_model
        instance.n_features = loaded_model.input_shape[1]

        logger.info(f"Loaded MLP model from {model_path}")

        return instance

    def summary(self) -> None:
        """Print model summary."""
        self.model.summary()
