"""
Variational POVM Learner

Implements a variational positive operator-valued measure (POVM) for quantum state
classification. Instead of fixed Pauli measurements, this approach learns an optimal
measurement basis by parameterizing a unitary circuit U(theta) applied before
computational basis measurement.

For a 3-qubit system:
- U(theta) is an 8x8 parameterized unitary (hardware-efficient ansatz)
- POVM elements: M_k(theta) = U(theta)^dag |k><k| U(theta), k = 0,...,7
- Outcome probabilities: p_k(rho) = Tr[M_k(theta) rho]
- Features for downstream classifier: p = (p_0, ..., p_7)

This is fully classically simulable since 8x8 matrix operations are trivial.

Reference: Manuscript Section 2.3 (Variational POVMs) and Section 3.3 (Quantum Pipeline).
"""

import numpy as np
import logging
from typing import Dict, Optional, List, Tuple

logger = logging.getLogger(__name__)

# Conditional PyTorch import
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ..utils import set_seed, get_split_seed


class ParameterizedUnitary(nn.Module if TORCH_AVAILABLE else object):
    """
    Hardware-efficient parameterized unitary for n qubits.

    Each layer consists of:
    1. Single-qubit R_Y(theta) and R_Z(theta) rotations on each qubit
    2. CNOT entangling gates with linear connectivity (0-1, 1-2, ...)

    The full unitary is the product of all layers:
    U(theta) = L_d . L_{d-1} . ... . L_1
    where each layer L_i = CNOT_layer . (tensor_j R_Z(theta_j) R_Y(theta_j))
    """

    def __init__(self, n_qubits: int = 3, n_layers: int = 4):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ParameterizedUnitary")

        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dim = 2 ** n_qubits  # 8 for 3 qubits

        # Parameters: 2 angles (R_Y, R_Z) per qubit per layer
        self.angles = nn.Parameter(
            torch.randn(n_layers, n_qubits, 2) * 0.1
        )

        # Precompute fixed CNOT matrices
        self._precompute_cnots()

    def _precompute_cnots(self):
        """Precompute CNOT gate matrices embedded in full Hilbert space."""
        cnot_2q = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=torch.complex64)

        # Linear connectivity: CNOT(0,1), CNOT(1,2), ...
        cnot_matrices = []
        for ctrl in range(self.n_qubits - 1):
            tgt = ctrl + 1
            cnot_full = self._embed_two_qubit_gate(cnot_2q, ctrl, tgt)
            cnot_matrices.append(cnot_full)

        # Compose all CNOTs into a single layer matrix
        composed = cnot_matrices[0]
        for g in cnot_matrices[1:]:
            composed = g @ composed

        self.register_buffer('cnot_layer', composed)

    def _embed_two_qubit_gate(
        self, gate_2q: 'torch.Tensor', q1: int, q2: int
    ) -> 'torch.Tensor':
        """Embed a 2-qubit gate into the full n-qubit Hilbert space."""
        dim = self.dim
        result = torch.zeros(dim, dim, dtype=torch.complex64)

        for i in range(dim):
            for j in range(dim):
                # Extract bits for q1, q2
                i_q1 = (i >> (self.n_qubits - 1 - q1)) & 1
                i_q2 = (i >> (self.n_qubits - 1 - q2)) & 1
                j_q1 = (j >> (self.n_qubits - 1 - q1)) & 1
                j_q2 = (j >> (self.n_qubits - 1 - q2)) & 1

                # All other qubits must match (identity on them)
                mask = ~((1 << (self.n_qubits - 1 - q1)) |
                         (1 << (self.n_qubits - 1 - q2)))
                if (i & mask) == (j & mask):
                    idx_i = i_q1 * 2 + i_q2
                    idx_j = j_q1 * 2 + j_q2
                    result[i, j] = gate_2q[idx_i, idx_j]

        return result

    def _single_qubit_rotation(
        self,
        angles_y: 'torch.Tensor',
        angles_z: 'torch.Tensor'
    ) -> 'torch.Tensor':
        """
        Build tensor product of R_Z(theta_z) R_Y(theta_y) for all qubits.

        Uses explicit real/imaginary construction for differentiability.
        """
        gates = []
        for q in range(self.n_qubits):
            cy = torch.cos(angles_y[q] / 2)
            sy = torch.sin(angles_y[q] / 2)
            cz = torch.cos(angles_z[q] / 2)
            sz = torch.sin(angles_z[q] / 2)

            # R_Z(theta) @ R_Y(phi) computed analytically:
            # [[e^{-iz/2} cos(y/2),  -e^{-iz/2} sin(y/2)],
            #  [e^{iz/2}  sin(y/2),   e^{iz/2}  cos(y/2)]]
            real_part = torch.stack([
                torch.stack([cz * cy, -cz * sy]),
                torch.stack([cz * sy, cz * cy])
            ])
            imag_part = torch.stack([
                torch.stack([-sz * cy, sz * sy]),
                torch.stack([sz * sy, sz * cy])
            ])
            gate = torch.complex(real_part, imag_part)
            gates.append(gate)

        # Tensor product: gate_0 kron gate_1 kron gate_2
        result = gates[0]
        for g in gates[1:]:
            result = torch.kron(result, g)

        return result

    def build_unitary(self) -> 'torch.Tensor':
        """
        Construct the full parameterized unitary U(theta).

        Returns:
            Unitary matrix of shape (dim, dim), complex64
        """
        U = torch.eye(self.dim, dtype=torch.complex64, device=self.angles.device)

        for layer in range(self.n_layers):
            rot = self._single_qubit_rotation(
                self.angles[layer, :, 0],  # R_Y angles
                self.angles[layer, :, 1],  # R_Z angles
            )
            # Move rotation to correct device
            rot = rot.to(self.angles.device)
            # Apply rotations then CNOTs
            U = self.cnot_layer @ rot @ U

        return U

    def forward(self, rho_batch: 'torch.Tensor') -> 'torch.Tensor':
        """
        Compute measurement probabilities for a batch of density matrices.

        Args:
            rho_batch: shape (batch_size, dim, dim), complex64

        Returns:
            Outcome probabilities, shape (batch_size, dim)
        """
        U = self.build_unitary()
        U_dag = U.conj().T

        # rho' = U rho U^dag (batched)
        rho_rotated = U @ rho_batch @ U_dag

        # p_k = <k| rho' |k> = diagonal elements
        probs = torch.diagonal(rho_rotated, dim1=-2, dim2=-1).real

        # Numerical safety: clamp and renormalize
        probs = torch.clamp(probs, min=0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        return probs


class VariationalPOVMClassifier(nn.Module if TORCH_AVAILABLE else object):
    """
    End-to-end variational POVM classifier.

    Combines a parameterized unitary (measurement) with a linear classifier
    on the outcome probabilities.
    """

    def __init__(self, n_qubits: int = 3, n_layers: int = 4):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for VariationalPOVMClassifier")

        super().__init__()
        dim = 2 ** n_qubits

        self.povm = ParameterizedUnitary(n_qubits=n_qubits, n_layers=n_layers)
        self.classifier = nn.Linear(dim, 2)

    def forward(self, rho_batch: 'torch.Tensor') -> 'torch.Tensor':
        """
        Args:
            rho_batch: shape (batch_size, dim, dim), complex64

        Returns:
            Logits, shape (batch_size, 2)
        """
        probs = self.povm(rho_batch)
        logits = self.classifier(probs)
        return logits


class VariationalPOVMLearner:
    """
    Wrapper for training and evaluating the variational POVM classifier.

    Follows the same API pattern as SVMWitnessLearner and TransformerWitnessLearner.

    Key difference from classical pipeline: takes density matrices directly,
    not Pauli feature vectors. The measurement itself is learned end-to-end.
    """

    def __init__(
        self,
        n_qubits: int = 3,
        n_layers: int = 4,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        n_epochs: int = 200,
        patience: int = 20,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for VariationalPOVMLearner")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.patience = patience
        self.random_state = random_state

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = None
        self.is_trained = False
        self.metrics = {}
        self.training_history = []

    def _build_model(self) -> 'VariationalPOVMClassifier':
        """Build and return the model."""
        model = VariationalPOVMClassifier(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
        )
        return model.to(self.device)

    def train(
        self,
        X_rho: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Train the variational POVM classifier.

        Args:
            X_rho: Density matrices, shape (n_samples, dim, dim), complex
            y: Labels, shape (n_samples,), 0 or 1
            test_size: Fraction held out for testing
            verbose: Whether to log progress

        Returns:
            Dictionary of performance metrics
        """
        if self.random_state is not None:
            set_seed(self.random_state)

        # Split data
        split_seed = get_split_seed(self.random_state)
        X_train, X_test, y_train, y_test = train_test_split(
            X_rho, y, test_size=test_size, random_state=split_seed, stratify=y
        )

        if verbose:
            logger.info(
                f"Training POVM learner: {len(X_train)} train, {len(X_test)} test"
            )
            logger.info(
                f"Architecture: {self.n_layers} layers, {self.n_qubits} qubits"
            )

        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.complex64).to(self.device)
        X_test_t = torch.tensor(X_test, dtype=torch.complex64).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(self.device)
        y_test_t = torch.tensor(y_test, dtype=torch.long).to(self.device)

        # Build model
        self.model = self._build_model()

        n_params = sum(p.numel() for p in self.model.parameters())
        if verbose:
            logger.info(f"Parameters: {n_params}")

        # Training setup
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Training loop with early stopping
        best_val_acc = 0.0
        best_state = None
        patience_counter = 0
        self.training_history = []

        for epoch in range(self.n_epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_rho, batch_y in train_loader:
                optimizer.zero_grad()
                logits = self.model(batch_rho)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(batch_y)
                train_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                train_total += len(batch_y)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_test_t)
                val_loss = criterion(val_logits, y_test_t).item()
                val_preds = val_logits.argmax(dim=1).cpu().numpy()
                val_acc = accuracy_score(y_test, val_preds)

            epoch_train_loss = train_loss / train_total
            epoch_train_acc = train_correct / train_total

            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': epoch_train_loss,
                'val_loss': val_loss,
                'train_acc': epoch_train_acc,
                'val_acc': val_acc,
            })

            if verbose and (epoch + 1) % 20 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.n_epochs}: "
                    f"train_loss={epoch_train_loss:.4f}, "
                    f"val_acc={val_acc:.4f}"
                )

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if verbose:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            test_logits = self.model(X_test_t)
            test_preds = test_logits.argmax(dim=1).cpu().numpy()

            train_logits = self.model(X_train_t)
            train_preds = train_logits.argmax(dim=1).cpu().numpy()

        self.metrics = {
            'train_accuracy': accuracy_score(y_train, train_preds),
            'test_accuracy': accuracy_score(y_test, test_preds),
            'train_precision': precision_score(y_train, train_preds, zero_division=0),
            'test_precision': precision_score(y_test, test_preds, zero_division=0),
            'train_recall': recall_score(y_train, train_preds, zero_division=0),
            'test_recall': recall_score(y_test, test_preds, zero_division=0),
            'train_f1': f1_score(y_train, train_preds, zero_division=0),
            'test_f1': f1_score(y_test, test_preds, zero_division=0),
            'n_parameters': n_params,
            'n_epochs_trained': len(self.training_history),
            'measurement_settings': 1,
        }

        self.is_trained = True

        if verbose:
            logger.info(
                f"Training complete. Test accuracy: {self.metrics['test_accuracy']:.4f}"
            )

        return self.metrics

    def predict(self, X_rho: np.ndarray) -> np.ndarray:
        """Predict labels for density matrices."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_rho, dtype=torch.complex64).to(self.device)
            logits = self.model(X_t)
            return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X_rho: np.ndarray) -> np.ndarray:
        """Predict class probabilities for density matrices."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_rho, dtype=torch.complex64).to(self.device)
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1)
            return probs.cpu().numpy()

    def get_povm_elements(self) -> List[np.ndarray]:
        """
        Extract the learned POVM elements M_k = U^dag |k><k| U.

        Returns:
            List of 2^n POVM element matrices (each dim x dim, complex)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before extracting POVM")

        self.model.eval()
        with torch.no_grad():
            U = self.model.povm.build_unitary().cpu().numpy()
            U_dag = U.conj().T

            dim = U.shape[0]
            povm_elements = []
            for k in range(dim):
                proj_k = np.zeros((dim, dim), dtype=complex)
                proj_k[k, k] = 1.0
                M_k = U_dag @ proj_k @ U
                povm_elements.append(M_k)

            return povm_elements

    def verify_povm(self) -> Dict[str, float]:
        """
        Verify POVM validity: all M_k >= 0 and sum_k M_k = I.

        Returns:
            Dictionary with verification metrics
        """
        elements = self.get_povm_elements()
        dim = elements[0].shape[0]

        # Check completeness: sum M_k = I
        total = sum(elements)
        completeness_error = np.linalg.norm(total - np.eye(dim))

        # Check positivity: all eigenvalues of M_k >= 0
        min_eigenvalue = float('inf')
        for M_k in elements:
            eigvals = np.linalg.eigvalsh(M_k)
            min_eigenvalue = min(min_eigenvalue, eigvals.min())

        return {
            'completeness_error': completeness_error,
            'min_eigenvalue': min_eigenvalue,
            'is_valid': completeness_error < 1e-5 and min_eigenvalue > -1e-10,
        }

    def get_measurement_cost(self) -> int:
        """
        Return the number of measurement settings.

        The variational POVM uses a single measurement setting
        (one unitary circuit + computational basis measurement).
        """
        return 1

    def get_classifier_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the linear classifier weights and bias.

        Returns:
            (weights, bias) where weights shape is (2, dim) and bias is (2,)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")

        with torch.no_grad():
            weights = self.model.classifier.weight.cpu().numpy()
            bias = self.model.classifier.bias.cpu().numpy()
        return weights, bias

    def get_witness_operator(self):
        """Not directly applicable for POVM pipeline."""
        return None

    def get_sparse_witness(self, threshold: float = 0.01):
        """Not directly applicable for POVM pipeline."""
        return None

    def save(self, path: str):
        """Save model to file."""
        torch.save({
            'model_state': self.model.state_dict(),
            'config': {
                'n_qubits': self.n_qubits,
                'n_layers': self.n_layers,
            },
            'metrics': self.metrics,
            'is_trained': self.is_trained,
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        config = checkpoint['config']
        self.n_qubits = config['n_qubits']
        self.n_layers = config['n_layers']
        self.model = self._build_model()
        self.model.load_state_dict(checkpoint['model_state'])
        self.metrics = checkpoint['metrics']
        self.is_trained = checkpoint['is_trained']
        logger.info(f"Model loaded from {path}")
