"""Amplitude-encoded variational classifier for restricted Pauli features."""

from typing import Optional

import pennylane as qml
import torch
import torch.nn as nn

from .qml_training import TorchQMLClassifierLearner


class AmplitudeEncodedQMLClassifier(nn.Module):
    """Six-qubit variational circuit mapping 36 amplitudes to two logits."""

    def __init__(
        self,
        n_features: int = 36,
        n_qubits: int = 6,
        n_layers: int = 2,
    ):
        super().__init__()
        if n_features > 2**n_qubits:
            raise ValueError(
                f"{n_qubits} qubits encode at most {2**n_qubits} features, "
                f"received {n_features}"
            )
        if n_qubits < 2:
            raise ValueError("At least two qubits are required for two output logits")

        self.n_features = n_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        weight_shape = qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)
        self.weights = nn.Parameter(0.01 * torch.randn(weight_shape, dtype=torch.float64))

        qml_device = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(qml_device, interface="torch", diff_method="backprop")
        def circuit(features, weights):
            qml.AmplitudeEmbedding(
                features,
                wires=range(n_qubits),
                pad_with=0.0,
                normalize=True,
            )
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        self.circuit = circuit

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return two circuit expectation values as classification logits."""
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.ndim != 2 or features.shape[1] != self.n_features:
            raise ValueError(
                f"Expected features with shape (batch, {self.n_features}), "
                f"received {tuple(features.shape)}"
            )
        if torch.any(torch.linalg.vector_norm(features, dim=1) == 0):
            raise ValueError("Zero-norm feature vectors cannot be amplitude encoded")

        outputs = self.circuit(features.to(dtype=torch.float64), self.weights)
        return torch.stack(outputs, dim=-1)


class AmplitudeQMLClassifierLearner(TorchQMLClassifierLearner):
    """Training and inference wrapper for the amplitude-encoded classifier."""

    def __init__(
        self,
        n_features: int = 36,
        n_qubits: int = 6,
        n_layers: int = 2,
        learning_rate: float = 1e-2,
        batch_size: int = 16,
        n_epochs: int = 50,
        patience: int = 10,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.n_features = n_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        super().__init__(
            model_factory=lambda: AmplitudeEncodedQMLClassifier(
                n_features=n_features,
                n_qubits=n_qubits,
                n_layers=n_layers,
            ),
            checkpoint_config={
                "n_features": n_features,
                "n_qubits": n_qubits,
                "n_layers": n_layers,
            },
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_epochs=n_epochs,
            patience=patience,
            random_state=random_state,
            device=device,
        )
