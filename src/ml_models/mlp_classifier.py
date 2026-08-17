"""
MLP Discriminator Classifier for Quantum State Classification

A discriminator-style deep neural network for binary classification
of quantum states based on Pauli expectation value features.

This is a pure classifier - no witness extraction.
"""

import numpy as np
import logging
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

# Conditional PyTorch import
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
    optim = None
    DataLoader = None
    TensorDataset = None

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class SingletonSafeBatchNorm1d(nn.BatchNorm1d if TORCH_AVAILABLE else object):
    """Batch normalization that falls back to running statistics for singletons."""

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        if self.training and x.shape[0] == 1:
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )
        return super().forward(x)


class MLPDiscriminator(nn.Module if TORCH_AVAILABLE else object):
    """
    Discriminator-style MLP for binary classification.

    Architecture follows GAN discriminator conventions:
    - LeakyReLU activations
    - BatchNorm for stability
    - Dropout for regularization
    """

    def __init__(
        self,
        input_dim: int = 36,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
        leaky_slope: float = 0.2
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for MLPDiscriminator")

        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                SingletonSafeBatchNorm1d(dim),
                nn.LeakyReLU(leaky_slope),
                nn.Dropout(dropout),
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, 2))  # Binary classification

        self.network = nn.Sequential(*layers)
        self.hidden_dims = hidden_dims

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """Forward pass through the network."""
        return self.network(x)

    def get_features(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """Get intermediate features (before final layer)."""
        for layer in list(self.network.children())[:-1]:
            x = layer(x)
        return x


class MLPClassifierLearner:
    """
    Wrapper class providing consistent API with other learners.

    Note: This is a pure classifier. Witness-related methods
    return None or raise NotImplementedError.

    Example:
        learner = MLPClassifierLearner(n_features=36)
        metrics = learner.train(X, y)
        predictions = learner.predict(X_test)
    """

    def __init__(
        self,
        n_features: int = 36,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        n_epochs: int = 100,
        patience: int = 15,
        random_state: Optional[int] = None,
        device: Optional[str] = None
    ):
        """
        Initialize MLP classifier.

        Args:
            n_features: Input dimension (36 for two-body Paulis)
            hidden_dims: Hidden layer sizes [default: [128, 64, 32]]
            dropout: Dropout rate
            learning_rate: Adam learning rate
            batch_size: Training batch size
            n_epochs: Maximum training epochs
            patience: Early stopping patience
            random_state: Random seed for reproducibility
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for MLPClassifierLearner")

        self.n_features = n_features
        self.hidden_dims = hidden_dims or [128, 64, 32]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.patience = patience
        self.random_state = random_state

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Set seed
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        # Initialize model
        self.model = MLPDiscriminator(
            input_dim=n_features,
            hidden_dims=self.hidden_dims,
            dropout=dropout
        ).to(self.device)

        self.is_trained = False
        self.training_history = []
        self.metrics = {}

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train the classifier with train/test split.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            test_size: Fraction for test set
            verbose: Print training progress

        Returns:
            Dictionary of metrics
        """
        # Split data
        split_seed = self.random_state + 1000 if self.random_state else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=split_seed, stratify=y
        )

        return self.fit(X_train, y_train, X_test, y_test, verbose)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train on pre-split data.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            verbose: Print progress

        Returns:
            Dictionary of metrics
        """
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Validation data
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.LongTensor(y_val).to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        self.training_history = []

        for epoch in range(self.n_epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_t)
                    val_loss = criterion(val_outputs, y_val_t).item()
                    val_pred = val_outputs.argmax(dim=1).cpu().numpy()
                    val_acc = accuracy_score(y_val, val_pred)

                self.training_history.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc
                })

                if verbose and epoch % 10 == 0:
                    logger.info(
                        f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                        f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                    )

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if verbose:
                            logger.info(f"Early stopping at epoch {epoch}")
                        break
            else:
                self.training_history.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                })

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.is_trained = True

        # Compute final metrics
        self.model.eval()
        with torch.no_grad():
            train_pred = self.model(X_train_t).argmax(dim=1).cpu().numpy()

        self.metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'train_precision': precision_score(y_train, train_pred, zero_division=0),
            'train_recall': recall_score(y_train, train_pred, zero_division=0),
            'train_f1': f1_score(y_train, train_pred, zero_division=0),
        }

        if has_val:
            with torch.no_grad():
                val_pred = self.model(X_val_t).argmax(dim=1).cpu().numpy()

            self.metrics.update({
                'test_accuracy': accuracy_score(y_val, val_pred),
                'test_precision': precision_score(y_val, val_pred, zero_division=0),
                'test_recall': recall_score(y_val, val_pred, zero_division=0),
                'test_f1': f1_score(y_val, val_pred, zero_division=0),
            })

        self.metrics['n_parameters'] = sum(
            p.numel() for p in self.model.parameters()
        )

        return self.metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1)
            return probs.cpu().numpy()

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Get raw logit difference (class 1 - class 0)."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before computing decision function")

        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            return (logits[:, 1] - logits[:, 0]).cpu().numpy()

    def get_witness_operator(self):
        """Not implemented - this is a pure classifier."""
        return None

    def get_sparse_witness(self, threshold: float = 0.01):
        """Not implemented - this is a pure classifier."""
        return None

    def get_measurement_cost(self) -> int:
        """Not applicable - returns -1."""
        return -1

    def save(self, path: str):
        """Save model to file."""
        torch.save({
            'model_state': self.model.state_dict(),
            'config': {
                'n_features': self.n_features,
                'hidden_dims': self.hidden_dims,
                'dropout': self.dropout,
            },
            'metrics': self.metrics,
            'is_trained': self.is_trained,
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        self.metrics = checkpoint['metrics']
        self.is_trained = checkpoint['is_trained']
        logger.info(f"Model loaded from {path}")
