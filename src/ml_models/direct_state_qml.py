"""PennyLane classifier operating directly on three-qubit density matrices."""

from typing import Optional

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from .qml_training import TorchQMLClassifierLearner


class DirectStateQMLClassifier(nn.Module):
    """Variational circuit applied directly to mixed-state density matrices."""

    def __init__(self, n_qubits: int = 3, n_layers: int = 2):
        super().__init__()
        if n_qubits < 2:
            raise ValueError("At least two qubits are required for two output logits")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dimension = 2**n_qubits
        weight_shape = qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)
        self.weights = nn.Parameter(0.01 * torch.randn(weight_shape, dtype=torch.float64))

        wires = range(n_qubits)

        def circuit(weights):
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

        self.circuit_matrix = qml.matrix(circuit, wire_order=wires)
        identity = torch.eye(2, dtype=torch.complex128)
        pauli_z = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.complex128))

        def embedded_z(target: int) -> torch.Tensor:
            observable = pauli_z if target == 0 else identity
            for wire in range(1, n_qubits):
                observable = torch.kron(
                    observable,
                    pauli_z if wire == target else identity,
                )
            return observable

        self.register_buffer("z0", embedded_z(0))
        self.register_buffer("z1", embedded_z(1))

    def forward(self, density_matrices: torch.Tensor) -> torch.Tensor:
        """Return two circuit expectation values for each density matrix."""
        if density_matrices.ndim == 2:
            density_matrices = density_matrices.unsqueeze(0)
        expected = (self.dimension, self.dimension)
        if density_matrices.ndim != 3 or tuple(density_matrices.shape[1:]) != expected:
            raise ValueError(
                f"Expected density matrices with shape (batch, {expected[0]}, "
                f"{expected[1]}), received {tuple(density_matrices.shape)}"
            )

        density_matrices = density_matrices.to(dtype=torch.complex128)
        if not torch.allclose(
            density_matrices,
            density_matrices.mH,
            rtol=1e-7,
            atol=1e-8,
        ):
            raise ValueError("Density matrices must be Hermitian")
        traces = density_matrices.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        if not torch.allclose(
            traces,
            torch.ones_like(traces),
            rtol=1e-7,
            atol=1e-8,
        ):
            raise ValueError("Density matrices must have unit trace")

        unitary = self.circuit_matrix(self.weights)
        rotated = unitary @ density_matrices @ unitary.mH
        return torch.stack(
            (
                torch.einsum("bij,ji->b", rotated, self.z0).real,
                torch.einsum("bij,ji->b", rotated, self.z1).real,
            ),
            dim=-1,
        )


class DirectStateQMLClassifierLearner(TorchQMLClassifierLearner):
    """Training wrapper for direct density-matrix classification."""

    def __init__(
        self,
        n_qubits: int = 3,
        n_layers: int = 2,
        learning_rate: float = 1e-2,
        batch_size: int = 16,
        n_epochs: int = 50,
        patience: int = 10,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        super().__init__(
            model_factory=lambda: DirectStateQMLClassifier(
                n_qubits=n_qubits,
                n_layers=n_layers,
            ),
            checkpoint_config={"n_qubits": n_qubits, "n_layers": n_layers},
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_epochs=n_epochs,
            patience=patience,
            random_state=random_state,
            device=device,
        )

    def _input_tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(X, dtype=torch.complex128, device=self.device)
