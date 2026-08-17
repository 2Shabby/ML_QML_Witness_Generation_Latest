"""Shared training wrapper for differentiable PennyLane classifiers."""

import logging
from collections.abc import Callable
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from ..utils import set_seed, stratified_split_indices

logger = logging.getLogger(__name__)


class TorchQMLClassifierLearner:
    """Reusable optimization, evaluation, and persistence for QML classifiers."""

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        checkpoint_config: dict,
        learning_rate: float,
        batch_size: int,
        n_epochs: int,
        patience: int,
        random_state: Optional[int],
        device: Optional[str],
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.patience = patience
        self.random_state = random_state
        self.checkpoint_config = checkpoint_config
        if device is None:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "ROCm device unavailable; this classifier requires the local GPU"
                )
            device = "cuda"
        self.device = torch.device(device)

        if random_state is not None:
            set_seed(random_state)
        self.model = model_factory().to(self.device)
        self.is_trained = False
        self.training_history = []
        self.metrics = {}
        self.split_indices = None

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Split data reproducibly and train the classifier."""
        X = np.asarray(X)
        y = np.asarray(y)
        self.split_indices = stratified_split_indices(
            y,
            test_size=test_size,
            random_state=self.random_state,
        )
        train_indices = self.split_indices["train"]
        test_indices = self.split_indices["test"]
        return self.fit(
            X[train_indices],
            y[train_indices],
            X[test_indices],
            y[test_indices],
            verbose=verbose,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Train on an explicit split, suitable for controlled comparisons."""
        X_train_t = self._input_tensor(X_train)
        y_train_t = torch.as_tensor(y_train, dtype=torch.long, device=self.device)
        loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=self.batch_size,
            shuffle=True,
        )
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_t = self._input_tensor(X_val)
            y_val_t = torch.as_tensor(y_val, dtype=torch.long, device=self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        best_loss = float("inf")
        best_state = None
        stale_epochs = 0
        self.training_history = []

        for epoch in range(self.n_epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(loader)
            record = {"epoch": epoch, "train_loss": epoch_loss}

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(X_val_t)
                    val_loss = criterion(val_logits, y_val_t).item()
                record["val_loss"] = val_loss
                if val_loss < best_loss:
                    best_loss = val_loss
                    stale_epochs = 0
                    best_state = {
                        key: value.detach().clone()
                        for key, value in self.model.state_dict().items()
                    }
                else:
                    stale_epochs += 1
            self.training_history.append(record)

            if verbose and epoch % 10 == 0:
                logger.info("QML epoch %d: train_loss=%.4f", epoch, epoch_loss)
            if has_val and stale_epochs >= self.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.is_trained = True

        train_predictions = self._predict_tensor(X_train_t)
        self.metrics = self._classification_metrics(
            y_train, train_predictions, prefix="train"
        )
        if has_val:
            val_predictions = self._predict_tensor(X_val_t)
            self.metrics.update(
                self._classification_metrics(y_val, val_predictions, prefix="test")
            )
        self.metrics["n_parameters"] = sum(
            parameter.numel() for parameter in self.model.parameters()
        )
        return self.metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels."""
        self._require_trained()
        return self._predict_tensor(self._input_tensor(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax probabilities derived from circuit outputs."""
        self._require_trained()
        self.model.eval()
        with torch.no_grad():
            probabilities = torch.softmax(self.model(self._input_tensor(X)), dim=1)
        return probabilities.cpu().numpy()

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return the class-one minus class-zero circuit output."""
        self._require_trained()
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self._input_tensor(X))
        return (logits[:, 1] - logits[:, 0]).cpu().numpy()

    def save(self, path: str) -> None:
        """Persist model state and training metadata."""
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "config": self.checkpoint_config,
                "metrics": self.metrics,
                "is_trained": self.is_trained,
                "split_indices": self.split_indices,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Restore model state and training metadata."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.metrics = checkpoint["metrics"]
        self.is_trained = checkpoint["is_trained"]
        self.split_indices = checkpoint.get("split_indices")

    def _input_tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(X, dtype=torch.float64, device=self.device)

    def _predict_tensor(self, X: torch.Tensor) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return self.model(X).argmax(dim=1).cpu().numpy()

    def _require_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

    @staticmethod
    def _classification_metrics(y_true, y_pred, prefix: str) -> Dict[str, float]:
        return {
            f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
            f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
            f"{prefix}_recall": recall_score(y_true, y_pred, zero_division=0),
            f"{prefix}_f1": f1_score(y_true, y_pred, zero_division=0),
        }
