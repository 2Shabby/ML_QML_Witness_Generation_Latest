"""
Transformer Witness Learner

Implements transformer-based witness learning for quantum state classification.
Provides both a standard transformer classifier and a hybrid architecture that
maintains interpretability by outputting witness coefficients.

Two architectures are provided:
1. TransformerClassifier: Standard transformer for pure classification
2. HybridTransformerWitness: Constrained architecture that outputs
   interpretable witness coefficients W = Σ wₖPₖ

The hybrid approach treats each Pauli expectation value as a "token" and uses
attention mechanisms to learn which correlators are most important for
distinguishing distillable from non-distillable states.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from qiskit.quantum_info import PauliList, SparsePauliOp
from typing import Tuple, Dict, Optional, List
import logging
import math

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer input.

    For Pauli features, we use learnable positional embeddings that can
    capture the structure of 1-body vs 2-body terms.
    """

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        return self.dropout(x)


class PauliTypeEncoding(nn.Module):
    """
    Encode Pauli operator type (1-body vs 2-body) as additional features.

    This helps the transformer understand the physical structure of the
    measurement basis.
    """

    def __init__(self, d_model: int, n_types: int = 2):
        super().__init__()
        self.type_embedding = nn.Embedding(n_types, d_model)

    def forward(self, x: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (batch_size, seq_len, d_model)
            type_ids: Pauli type indices (seq_len,) - 0 for 1-body, 1 for 2-body
        """
        type_emb = self.type_embedding(type_ids)  # (seq_len, d_model)
        return x + type_emb.unsqueeze(0)


class TransformerBlock(nn.Module):
    """
    Single transformer block with multi-head self-attention and feed-forward.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.n_heads = n_heads

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            Output tensor and attention weights
        """
        # Self-attention with residual
        # Pass average_attn_weights=False to get per-head attention (batch, n_heads, seq, seq)
        attn_out, attn_weights = self.attention(x, x, x, average_attn_weights=False)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        return x, attn_weights


class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier for quantum state classification.

    Treats the 36D Pauli feature vector as a sequence, with each feature
    being a "token". Uses self-attention to learn correlations between
    different Pauli expectation values.

    Architecture:
    - Input embedding: Linear projection of scalar features to d_model
    - Positional encoding: Learnable position embeddings
    - Transformer blocks: Self-attention + feed-forward
    - Classification head: Pool + MLP for binary classification
    """

    def __init__(
        self,
        n_features: int = 36,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        use_pauli_type: bool = True
    ):
        super().__init__()

        self.n_features = n_features
        self.d_model = d_model
        self.use_pauli_type = use_pauli_type

        # Input embedding: project scalar to d_model
        self.input_embedding = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=n_features, dropout=dropout)

        # Pauli type encoding (optional)
        if use_pauli_type:
            self.type_encoding = PauliTypeEncoding(d_model, n_types=2)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)  # Binary classification
        )

        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Store attention weights for analysis
        self.attention_weights = None

    def forward(
        self,
        x: torch.Tensor,
        pauli_types: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Feature tensor (batch_size, n_features)
            pauli_types: Optional tensor of Pauli types (n_features,)

        Returns:
            Logits tensor (batch_size, 2)
        """
        batch_size = x.size(0)

        # Reshape to sequence: (batch, n_features) -> (batch, n_features, 1)
        x = x.unsqueeze(-1)

        # Embed each feature
        x = self.input_embedding(x)  # (batch, n_features, d_model)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Add Pauli type encoding if provided
        if self.use_pauli_type and pauli_types is not None:
            x = self.type_encoding(x, pauli_types)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, 1 + n_features, d_model)

        # Apply transformer blocks
        all_attention = []
        for block in self.transformer_blocks:
            x, attn = block(x)
            all_attention.append(attn)

        self.attention_weights = all_attention

        # Use CLS token for classification
        cls_output = x[:, 0, :]  # (batch, d_model)

        # Classification
        logits = self.classifier(cls_output)  # (batch, 2)

        return logits

    def get_attention_weights(self) -> List[torch.Tensor]:
        """Return attention weights from last forward pass."""
        return self.attention_weights


class HybridTransformerWitness(nn.Module):
    """
    Hybrid transformer that outputs interpretable witness coefficients.

    This architecture is constrained to output a linear combination of
    Pauli operators, maintaining the interpretability of SVM witnesses
    while potentially learning non-linear feature interactions.

    Architecture:
    - Transformer encoder processes the features
    - Output layer produces 36 coefficients (one per Pauli)
    - Classification is: sign(Σ wₖ xₖ + bias)

    This allows direct extraction of witness operator W = Σ wₖPₖ
    """

    def __init__(
        self,
        n_features: int = 36,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_features = n_features
        self.d_model = d_model

        # Input embedding
        self.input_embedding = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=n_features, dropout=dropout)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Coefficient generator: produces one coefficient per Pauli term
        # This layer processes each position and outputs a scalar weight
        self.coeff_generator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        # Learnable bias term
        self.bias = nn.Parameter(torch.zeros(1))

        # Store coefficients for witness extraction
        self.witness_coefficients = None
        self.attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that maintains witness interpretability.

        The output is computed as: logit = Σ wₖ(x) · xₖ + bias
        where wₖ(x) are the learned coefficients.

        Args:
            x: Feature tensor (batch_size, n_features)

        Returns:
            Logits tensor (batch_size, 2) for binary classification
        """
        batch_size = x.size(0)
        original_x = x  # Keep original features for final computation

        # Reshape to sequence
        x = x.unsqueeze(-1)  # (batch, n_features, 1)

        # Embed
        x = self.input_embedding(x)  # (batch, n_features, d_model)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Apply transformer blocks
        all_attention = []
        for block in self.transformer_blocks:
            x, attn = block(x)
            all_attention.append(attn)

        self.attention_weights = all_attention

        # Generate coefficient for each position
        coeffs = self.coeff_generator(x).squeeze(-1)  # (batch, n_features)
        self.witness_coefficients = coeffs.detach()

        # Compute witness value: Σ wₖ · xₖ
        witness_value = (coeffs * original_x).sum(dim=1, keepdim=True)  # (batch, 1)

        # Add bias and convert to logits
        decision = witness_value + self.bias

        # Convert to binary logits: [logit_class0, logit_class1]
        # Positive decision -> class 1 (distillable)
        logits = torch.cat([-decision, decision], dim=1)

        return logits

    def get_witness_coefficients(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the witness coefficients for given inputs.

        Args:
            x: Feature tensor (batch_size, n_features)

        Returns:
            Coefficient tensor (batch_size, n_features)
        """
        _ = self.forward(x)
        return self.witness_coefficients

    def get_mean_coefficients(self, x: torch.Tensor) -> np.ndarray:
        """
        Get mean witness coefficients across a batch.

        This is useful for extracting a single witness operator that
        works well on average across the training distribution.

        Args:
            x: Feature tensor (batch_size, n_features)

        Returns:
            Mean coefficients as numpy array (n_features,)
        """
        coeffs = self.get_witness_coefficients(x)
        return coeffs.mean(dim=0).cpu().numpy()


class TransformerWitnessLearner:
    """
    Wrapper class providing same API as SVMWitnessLearner for transformers.

    Supports both standard transformer classification and hybrid witness
    extraction mode.

    Example usage:
        learner = TransformerWitnessLearner(
            pauli_basis=basis,
            mode='hybrid',  # or 'classifier'
            d_model=64,
            n_layers=2
        )
        metrics = learner.train(X, y)
        witness = learner.get_witness_operator()
    """

    def __init__(
        self,
        pauli_basis: PauliList,
        mode: str = 'hybrid',
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        n_epochs: int = 100,
        patience: int = 10,
        random_state: Optional[int] = None,
        device: Optional[str] = None
    ):
        """
        Initialize transformer witness learner.

        Args:
            pauli_basis: Pauli basis used for feature extraction
            mode: 'classifier' for standard transformer, 'hybrid' for witness extraction
            d_model: Transformer hidden dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
            learning_rate: Learning rate for Adam optimizer
            weight_decay: L2 regularization strength
            batch_size: Training batch size
            n_epochs: Maximum training epochs
            patience: Early stopping patience
            random_state: Random seed for reproducibility
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.pauli_basis = pauli_basis
        self.mode = mode
        self.n_features = len(pauli_basis)

        # Architecture hyperparameters
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout

        # Training hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.patience = patience
        self.random_state = random_state

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Set random seed
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        # Initialize model
        self._init_model()

        self.is_trained = False
        self.witness_operator = None
        self.bias = None
        self.metrics = {}
        self.training_history = []

        # Create Pauli type tensor (0 for 1-body, 1 for 2-body)
        self.pauli_types = self._compute_pauli_types()

    def _init_model(self):
        """Initialize the transformer model based on mode."""
        if self.mode == 'classifier':
            self.model = TransformerClassifier(
                n_features=self.n_features,
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=self.n_layers,
                d_ff=self.d_ff,
                dropout=self.dropout
            )
        elif self.mode == 'hybrid':
            self.model = HybridTransformerWitness(
                n_features=self.n_features,
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=self.n_layers,
                d_ff=self.d_ff,
                dropout=self.dropout
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'classifier' or 'hybrid'")

        self.model = self.model.to(self.device)

    def _compute_pauli_types(self) -> torch.Tensor:
        """Compute Pauli type (1-body vs 2-body) for each basis element."""
        types = []
        for pauli in self.pauli_basis:
            pauli_str = str(pauli)
            n_non_identity = sum(1 for c in pauli_str if c != 'I')
            # 0 for 1-body, 1 for 2-body (or higher)
            types.append(0 if n_non_identity == 1 else 1)
        return torch.tensor(types, dtype=torch.long, device=self.device)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train the transformer on labeled feature data.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Labels (0=separable, 1=entangled/distillable)
            test_size: Fraction of data for testing
            verbose: Whether to log progress

        Returns:
            Dictionary of performance metrics
        """
        if verbose:
            logger.info(f"Training Transformer Witness Learner ({self.mode} mode)...")
            logger.info(f"Device: {self.device}")
            logger.info(f"Training samples: {len(X)}, Features: {X.shape[1]}")
            logger.info(f"Architecture: d_model={self.d_model}, n_layers={self.n_layers}, n_heads={self.n_heads}")

        # Split data
        split_seed = self.random_state + 1000 if self.random_state is not None else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=split_seed, stratify=y
        )

        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.long, device=self.device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=self.device)
        y_test_t = torch.tensor(y_test, dtype=torch.long, device=self.device)

        # Create data loaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizer and loss
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Learning rate scheduler (verbose removed in newer PyTorch versions)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        criterion = nn.CrossEntropyLoss()

        # Training loop with early stopping
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        self.training_history = []

        for epoch in range(self.n_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()

                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(logits, 1)
                train_correct += (predicted == batch_y).sum().item()
                train_total += batch_y.size(0)

            train_loss /= train_total
            train_acc = train_correct / train_total

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_test_t)
                val_loss = criterion(val_logits, y_test_t).item()
                _, val_predicted = torch.max(val_logits, 1)
                val_acc = (val_predicted == y_test_t).float().mean().item()

            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.n_epochs}: "
                          f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                          f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

            if patience_counter >= self.patience:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})

        self.is_trained = True

        # Extract witness operator (for hybrid mode)
        if self.mode == 'hybrid':
            self._extract_witness_operator(X_train_t)

        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            train_logits = self.model(X_train_t)
            test_logits = self.model(X_test_t)

            _, train_pred = torch.max(train_logits, 1)
            _, test_pred = torch.max(test_logits, 1)

            train_pred = train_pred.cpu().numpy()
            test_pred = test_pred.cpu().numpy()

        self.metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'test_accuracy': accuracy_score(y_test, test_pred),
            'train_precision': precision_score(y_train, train_pred, zero_division=0),
            'test_precision': precision_score(y_test, test_pred, zero_division=0),
            'train_recall': recall_score(y_train, train_pred, zero_division=0),
            'test_recall': recall_score(y_test, test_pred, zero_division=0),
            'n_parameters': sum(p.numel() for p in self.model.parameters()),
            'n_epochs_trained': len(self.training_history),
            'best_val_loss': best_val_loss
        }

        if verbose:
            logger.info("Training complete!")
            logger.info(f"Test Accuracy: {self.metrics['test_accuracy']:.4f}")
            logger.info(f"Test Precision: {self.metrics['test_precision']:.4f}")
            logger.info(f"Test Recall: {self.metrics['test_recall']:.4f}")
            logger.info(f"Parameters: {self.metrics['n_parameters']}")

        return self.metrics

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train the transformer on pre-split data (no internal splitting).

        Use this method when you have already split your data to avoid
        data leakage issues.

        Args:
            X_train: Training feature matrix
            y_train: Training labels
            X_val: Validation/test feature matrix
            y_val: Validation/test labels
            verbose: Whether to log progress

        Returns:
            Dictionary of performance metrics
        """
        if verbose:
            logger.info(f"Training Transformer Witness Learner ({self.mode} mode)...")
            logger.info(f"Device: {self.device}")
            logger.info(f"Training samples: {len(X_train)}, Val samples: {len(X_val)}, Features: {X_train.shape[1]}")
            logger.info(f"Architecture: d_model={self.d_model}, n_layers={self.n_layers}, n_heads={self.n_heads}")

        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.long, device=self.device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.long, device=self.device)

        # Create data loaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizer and loss
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        criterion = nn.CrossEntropyLoss()

        # Training loop with early stopping
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        self.training_history = []

        for epoch in range(self.n_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()

                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(logits, 1)
                train_correct += (predicted == batch_y).sum().item()
                train_total += batch_y.size(0)

            train_loss /= train_total
            train_acc = train_correct / train_total

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val_t)
                val_loss = criterion(val_logits, y_val_t).item()
                _, val_predicted = torch.max(val_logits, 1)
                val_acc = (val_predicted == y_val_t).float().mean().item()

            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.n_epochs}: "
                          f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                          f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

            if patience_counter >= self.patience:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})

        self.is_trained = True

        # Extract witness operator (for hybrid mode)
        if self.mode == 'hybrid':
            self._extract_witness_operator(X_train_t)

        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            train_logits = self.model(X_train_t)
            val_logits = self.model(X_val_t)

            _, train_pred = torch.max(train_logits, 1)
            _, val_pred = torch.max(val_logits, 1)

            train_pred = train_pred.cpu().numpy()
            val_pred = val_pred.cpu().numpy()

        self.metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'test_accuracy': accuracy_score(y_val, val_pred),
            'train_precision': precision_score(y_train, train_pred, zero_division=0),
            'test_precision': precision_score(y_val, val_pred, zero_division=0),
            'train_recall': recall_score(y_train, train_pred, zero_division=0),
            'test_recall': recall_score(y_val, val_pred, zero_division=0),
            'n_parameters': sum(p.numel() for p in self.model.parameters()),
            'n_epochs_trained': len(self.training_history),
            'best_val_loss': best_val_loss
        }

        if verbose:
            logger.info("Training complete!")
            logger.info(f"Test Accuracy: {self.metrics['test_accuracy']:.4f}")
            logger.info(f"Test Precision: {self.metrics['test_precision']:.4f}")
            logger.info(f"Test Recall: {self.metrics['test_recall']:.4f}")
            logger.info(f"Parameters: {self.metrics['n_parameters']}")

        return self.metrics

    def _extract_witness_operator(self, X: torch.Tensor):
        """
        Extract witness operator from hybrid model.

        Uses mean coefficients across training data as the witness.
        """
        if self.mode != 'hybrid':
            raise ValueError("Witness extraction only available in hybrid mode")

        self.model.eval()
        with torch.no_grad():
            # Get coefficients for all training data
            coeffs = self.model.get_mean_coefficients(X)
            self.bias = self.model.bias.item()

        # Build witness operator
        pauli_strings = []
        coefficients = []

        for i, (pauli, weight) in enumerate(zip(self.pauli_basis, coeffs)):
            if abs(weight) > 1e-10:
                pauli_strings.append(str(pauli))
                coefficients.append(float(weight))

        if len(pauli_strings) > 0:
            self.witness_operator = SparsePauliOp(
                pauli_strings,
                coeffs=coefficients
            )

        logger.info(f"Witness operator extracted: {len(pauli_strings)} non-zero terms")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for feature vectors.

        Args:
            X: Feature matrix

        Returns:
            Predicted labels (0=separable, 1=entangled)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = self.model(X_t)
            _, predicted = torch.max(logits, 1)

        return predicted.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix

        Returns:
            Probability matrix of shape (n_samples, 2)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1)

        return probs.cpu().numpy()

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function (raw scores before sigmoid).

        For hybrid mode, this is Tr(Wρ) + bias.

        Args:
            X: Feature matrix

        Returns:
            Decision function values
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before computing decision function")

        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = self.model(X_t)
            # Return difference: logit[1] - logit[0]
            decision = logits[:, 1] - logits[:, 0]

        return decision.cpu().numpy()

    def get_witness_operator(self) -> SparsePauliOp:
        """
        Get the extracted witness operator.

        Only available in hybrid mode.

        Returns:
            Witness operator as SparsePauliOp
        """
        if self.mode != 'hybrid':
            raise ValueError("Witness extraction only available in hybrid mode. "
                           "Use mode='hybrid' for interpretable witnesses.")

        if self.witness_operator is None:
            raise RuntimeError("Model must be trained first")

        return self.witness_operator

    def get_sparse_witness(self, threshold: float = 0.01) -> SparsePauliOp:
        """
        Get a sparse version of the witness by thresholding small coefficients.

        Args:
            threshold: Coefficients with |w_k| < threshold are set to zero

        Returns:
            Sparse witness operator
        """
        if self.witness_operator is None:
            raise RuntimeError("Witness operator not available")

        pauli_strings = []
        coefficients = []

        for pauli, coeff in self.witness_operator.to_list():
            if abs(coeff) >= threshold:
                pauli_strings.append(pauli)
                coefficients.append(coeff)

        logger.info(f"Sparse witness: {len(pauli_strings)} terms (threshold={threshold})")

        # Handle case where no coefficients exceed threshold
        if len(pauli_strings) == 0:
            logger.warning(
                f"No coefficients exceed threshold {threshold}. "
                f"Returning full witness operator instead."
            )
            return self.witness_operator

        return SparsePauliOp(pauli_strings, coeffs=coefficients)

    def evaluate_witness(
        self,
        rho_data: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate witness performance on density matrices.

        Args:
            rho_data: Array of density matrices
            labels: True labels (0=separable, 1=entangled)

        Returns:
            Dictionary of witness-specific metrics
        """
        if self.witness_operator is None:
            raise RuntimeError("Witness operator not available")

        witness_values = []
        for rho in rho_data:
            exp_val = np.trace(rho @ self.witness_operator.to_matrix()).real
            witness_values.append(exp_val)

        witness_values = np.array(witness_values)

        # Classify based on witness
        predicted = (witness_values + self.bias) > 0

        metrics = {
            'witness_accuracy': accuracy_score(labels, predicted),
            'witness_precision': precision_score(labels, predicted, zero_division=0),
            'witness_recall': recall_score(labels, predicted, zero_division=0),
            'mean_violation_entangled': float(witness_values[labels == 1].mean()) if sum(labels == 1) > 0 else 0.0,
            'mean_violation_separable': float(witness_values[labels == 0].mean()) if sum(labels == 0) > 0 else 0.0,
        }

        return metrics

    def get_measurement_cost(self) -> int:
        """
        Estimate the number of measurement settings required.

        Returns:
            Number of measurement settings
        """
        from ..feature_extraction.pauli_features import estimate_measurement_cost

        if self.witness_operator is None:
            raise RuntimeError("Witness operator not available")

        pauli_list = PauliList([pauli for pauli, _ in self.witness_operator.to_list()])

        return estimate_measurement_cost(pauli_list)

    def get_attention_analysis(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Analyze attention patterns to understand feature importance.

        Args:
            X: Feature matrix

        Returns:
            Dictionary with attention statistics
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")

        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _ = self.model(X_t)
            attention_weights = self.model.attention_weights

        # Average attention across batch and heads
        # Shape: (batch, heads, seq+1, seq+1) -> (n_features,)
        mean_attention = []
        for layer_attn in attention_weights:
            # Get attention from CLS token to feature tokens
            cls_attention = layer_attn[:, :, 0, 1:].mean(dim=(0, 1))
            mean_attention.append(cls_attention.cpu().numpy())

        return {
            'per_layer_attention': mean_attention,
            'mean_attention': np.mean(mean_attention, axis=0),
            'pauli_labels': [str(p) for p in self.pauli_basis]
        }

    def save(self, path: str):
        """Save model to file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'mode': self.mode,
                'n_features': self.n_features,
                'd_model': self.d_model,
                'n_heads': self.n_heads,
                'n_layers': self.n_layers,
                'd_ff': self.d_ff,
                'dropout': self.dropout,
            },
            'metrics': self.metrics,
            'training_history': self.training_history,
            'bias': self.bias
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.metrics = checkpoint.get('metrics', {})
        self.training_history = checkpoint.get('training_history', [])
        self.bias = checkpoint.get('bias')
        self.is_trained = True
        logger.info(f"Model loaded from {path}")
