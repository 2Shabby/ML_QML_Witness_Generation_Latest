# MLP Discriminator Classifier Implementation Plan

## Overview

Add a discriminator-style MLP (Multi-Layer Perceptron) as a **pure classifier** for 3-qubit distillability classification. No witness extraction - focused solely on classification performance.

## Rationale

- **SVM**: Linear classifier with interpretable witness (W = Σ wₖPₖ)
- **Transformer**: Attention-based classifier with optional witness
- **MLP Discriminator**: Deep non-linear classifier, pure classification focus

The MLP discriminator provides:
1. Non-linear decision boundaries (like Transformer)
2. Simpler architecture than Transformer (no attention mechanism)
3. Fast training and inference
4. Standard deep learning baseline

---

## Architecture

### MLPClassifier

```
Input: 36D Pauli features
    ↓
Linear(36 → 128) + BatchNorm + LeakyReLU(0.2) + Dropout(0.3)
    ↓
Linear(128 → 64) + BatchNorm + LeakyReLU(0.2) + Dropout(0.3)
    ↓
Linear(64 → 32) + BatchNorm + LeakyReLU(0.2) + Dropout(0.3)
    ↓
Linear(32 → 2)
    ↓
Output: Class logits (softmax for probabilities)
```

**Design choices:**
- LeakyReLU: Avoids dead neurons (discriminator-style)
- BatchNorm: Stabilizes training
- Dropout: Prevents overfitting
- No witness extraction: Pure classification

---

## File Changes

### New Files

#### 1. `src/ml_models/mlp_classifier.py` (~300 lines)

```python
"""
MLP Discriminator Classifier for Quantum State Classification

A discriminator-style deep neural network for binary classification
of quantum states based on Pauli expectation value features.

This is a pure classifier - no witness extraction.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class MLPDiscriminator(nn.Module):
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
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.3,
        leaky_slope: float = 0.2
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.LeakyReLU(leaky_slope),
                nn.Dropout(dropout),
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, 2))  # Binary classification

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
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
        if X_val is not None:
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.LongTensor(y_val).to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

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
            if X_val is not None:
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
                    best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if verbose:
                            logger.info(f"Early stopping at epoch {epoch}")
                        break

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

        if X_val is not None:
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
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1)
            return probs.cpu().numpy()

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Get raw logit difference (class 1 - class 0)."""
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
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.metrics = checkpoint['metrics']
        self.is_trained = checkpoint['is_trained']
        logger.info(f"Model loaded from {path}")
```

#### 2. `tests/test_mlp_classifier.py` (~250 lines)

```python
"""
Tests for MLP Discriminator Classifier

Tests the MLPClassifier model for quantum state classification.
"""

import pytest
import numpy as np

# Skip all tests if torch not available
torch_available = True
try:
    import torch
    from src.ml_models.mlp_classifier import (
        MLPClassifierLearner,
        MLPDiscriminator
    )
except ImportError:
    torch_available = False

pytestmark = pytest.mark.skipif(
    not torch_available,
    reason="PyTorch not installed"
)


class TestMLPDiscriminator:
    """Test MLPDiscriminator neural network."""

    def test_output_shape(self):
        """Test network produces correct output shape."""
        model = MLPDiscriminator(input_dim=36, hidden_dims=[64, 32])
        x = torch.randn(10, 36)
        out = model(x)
        assert out.shape == (10, 2)

    def test_custom_hidden_dims(self):
        """Test custom hidden layer configuration."""
        model = MLPDiscriminator(input_dim=36, hidden_dims=[128, 64, 32, 16])
        x = torch.randn(5, 36)
        out = model(x)
        assert out.shape == (5, 2)

    def test_get_features(self):
        """Test intermediate feature extraction."""
        model = MLPDiscriminator(input_dim=36, hidden_dims=[64, 32])
        x = torch.randn(10, 36)
        features = model.get_features(x)
        assert features.shape == (10, 32)  # Last hidden dim


class TestMLPClassifierLearner:
    """Test MLPClassifierLearner wrapper class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample classification data."""
        np.random.seed(42)
        n_samples = 200
        n_features = 36

        # Create linearly separable data with some noise
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.5 > 0).astype(int)

        return X, y

    def test_initialization(self):
        """Test learner initialization."""
        learner = MLPClassifierLearner(n_features=36)
        assert learner.n_features == 36
        assert learner.is_trained == False

    def test_train(self, sample_data):
        """Test training with automatic split."""
        X, y = sample_data
        learner = MLPClassifierLearner(
            n_features=36,
            n_epochs=20,
            random_state=42
        )

        metrics = learner.train(X, y, test_size=0.2, verbose=False)

        assert learner.is_trained == True
        assert 'train_accuracy' in metrics
        assert 'test_accuracy' in metrics
        assert metrics['train_accuracy'] > 0.5

    def test_fit(self, sample_data):
        """Test training with pre-split data."""
        X, y = sample_data
        X_train, X_test = X[:160], X[160:]
        y_train, y_test = y[:160], y[160:]

        learner = MLPClassifierLearner(n_features=36, n_epochs=20)
        metrics = learner.fit(X_train, y_train, X_test, y_test, verbose=False)

        assert learner.is_trained == True
        assert 'test_accuracy' in metrics

    def test_predict(self, sample_data):
        """Test prediction."""
        X, y = sample_data
        learner = MLPClassifierLearner(n_features=36, n_epochs=20)
        learner.train(X, y, verbose=False)

        predictions = learner.predict(X[:10])

        assert predictions.shape == (10,)
        assert set(predictions).issubset({0, 1})

    def test_predict_proba(self, sample_data):
        """Test probability prediction."""
        X, y = sample_data
        learner = MLPClassifierLearner(n_features=36, n_epochs=20)
        learner.train(X, y, verbose=False)

        probs = learner.predict_proba(X[:10])

        assert probs.shape == (10, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_decision_function(self, sample_data):
        """Test decision function."""
        X, y = sample_data
        learner = MLPClassifierLearner(n_features=36, n_epochs=20)
        learner.train(X, y, verbose=False)

        decisions = learner.decision_function(X[:10])

        assert decisions.shape == (10,)

    def test_witness_returns_none(self, sample_data):
        """Test that witness methods return None (pure classifier)."""
        X, y = sample_data
        learner = MLPClassifierLearner(n_features=36, n_epochs=10)
        learner.train(X, y, verbose=False)

        assert learner.get_witness_operator() is None
        assert learner.get_sparse_witness() is None
        assert learner.get_measurement_cost() == -1

    def test_save_load(self, sample_data, tmp_path):
        """Test model serialization."""
        X, y = sample_data
        learner = MLPClassifierLearner(n_features=36, n_epochs=20)
        learner.train(X, y, verbose=False)

        # Save
        save_path = tmp_path / "model.pt"
        learner.save(str(save_path))

        # Load into new learner
        learner2 = MLPClassifierLearner(n_features=36)
        learner2.load(str(save_path))

        # Compare predictions
        pred1 = learner.predict(X[:10])
        pred2 = learner2.predict(X[:10])

        assert np.array_equal(pred1, pred2)

    def test_early_stopping(self, sample_data):
        """Test that early stopping works."""
        X, y = sample_data
        learner = MLPClassifierLearner(
            n_features=36,
            n_epochs=1000,  # High max epochs
            patience=5,    # Low patience
            random_state=42
        )

        learner.train(X, y, verbose=False)

        # Should have stopped before 1000 epochs
        assert len(learner.training_history) < 1000


class TestMLPWithQuantumData:
    """Integration tests with quantum state features."""

    def test_with_pauli_features(self):
        """Test with actual Pauli feature dimensions."""
        from src.feature_extraction.pauli_features import create_sparse_measurement_set
        from src.quantum_states.state_generation import generate_distillability_dataset

        # Generate small dataset
        states, labels = generate_distillability_dataset(
            n_samples=100,
            noise_range=(0.0, 0.5),
            seed=42
        )

        # Extract features
        basis = create_sparse_measurement_set(3, 'two_body')
        from src.feature_extraction.pauli_features import extract_features_batch
        features = extract_features_batch(states, basis, verbose=False)

        # Train classifier
        learner = MLPClassifierLearner(
            n_features=len(basis),
            n_epochs=30,
            random_state=42
        )

        metrics = learner.train(features, np.array(labels), verbose=False)

        # Should achieve reasonable accuracy
        assert metrics['test_accuracy'] > 0.55
```

### Modified Files

#### 3. `src/config.py` - Add MLPConfig

```python
@dataclass
class MLPConfig:
    """Configuration for MLP discriminator classifier."""
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.3
    learning_rate: float = 1e-3
    batch_size: int = 64
    n_epochs: int = 100
    patience: int = 15

DEFAULT_MLP_CONFIG = MLPConfig()
```

Add to `__all__`: `'MLPConfig'`, `'DEFAULT_MLP_CONFIG'`

#### 4. `src/ml_models/__init__.py` - Export MLPClassifierLearner

```python
# Conditionally import MLP classifier (requires PyTorch)
try:
    from .mlp_classifier import MLPClassifierLearner, MLPDiscriminator
    _MLP_AVAILABLE = True
except ImportError:
    _MLP_AVAILABLE = False
    MLPClassifierLearner = None
    MLPDiscriminator = None

# Add to __all__ if available
if _MLP_AVAILABLE:
    __all__.extend(['MLPClassifierLearner', 'MLPDiscriminator'])
```

#### 5. `scripts/run_comparative_analysis.py` - Add MLP evaluation

Add to `ModelEvaluator` class:

```python
def evaluate_mlp(self) -> Optional[Dict]:
    """Train and evaluate MLP discriminator classifier."""
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available, skipping MLP")
        return None

    from src.ml_models.mlp_classifier import MLPClassifierLearner

    logger.info("Training MLP Discriminator...")

    learner = MLPClassifierLearner(
        n_features=len(self.basis),
        random_state=self.seed
    )

    learner.fit(self.X_train, self.y_train, self.X_test, self.y_test, verbose=False)

    y_pred = learner.predict(self.X_test)
    y_proba = learner.predict_proba(self.X_test)[:, 1]

    self.results['mlp'] = {
        'name': 'MLP Discriminator',
        'y_pred': y_pred,
        'y_proba': y_proba,
        'metrics': self._compute_metrics(y_pred, y_proba),
        'n_parameters': learner.metrics.get('n_parameters', 0),
        'model': learner
    }

    logger.info(f"MLP Test Accuracy: {self.results['mlp']['metrics']['accuracy']:.4f}")
    return self.results['mlp']
```

Update `run_all()`:
```python
def run_all(self) -> Dict:
    """Run evaluation for all available models."""
    self.evaluate_svm()
    self.evaluate_mlp()  # Add this
    self.evaluate_transformer_classifier()
    self.evaluate_transformer_hybrid()
    return self.results
```

Update colors in `plot_comparative_analysis()`:
```python
colors = {
    'svm': '#3498db',
    'mlp': '#9b59b6',  # Add purple for MLP
    'transformer_classifier': '#2ecc71',
    'transformer_hybrid': '#e74c3c'
}
```

#### 6. `tests/test_integration.py` - Add MLP integration test

```python
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_3qubit_distillability_pipeline_mlp(self):
    """
    Test end-to-end 3-qubit distillability with MLP classifier.
    """
    from src.ml_models.mlp_classifier import MLPClassifierLearner

    # Generate dataset
    states, labels = generate_distillability_dataset(
        n_samples=300,
        noise_range=(0.0, 0.5),
        seed=42
    )
    labels = np.array(labels)

    # Extract features
    basis = create_sparse_measurement_set(3, 'two_body')
    features = extract_features_batch(states, basis, verbose=False)

    # Train MLP
    learner = MLPClassifierLearner(
        n_features=len(basis),
        n_epochs=50,
        random_state=42
    )

    metrics = learner.train(features, labels, test_size=0.2, verbose=False)

    # Verify accuracy
    assert metrics['test_accuracy'] > 0.55, "MLP should beat random"

    # Verify witness methods return None
    assert learner.get_witness_operator() is None

    # Test on known states
    ghz_state = generate_entangled_state(3, 'ghz', noise_level=0.0)
    product_state = generate_3qubit_product_state()

    ghz_features = extract_pauli_features(ghz_state, basis).reshape(1, -1)
    product_features = extract_pauli_features(product_state, basis).reshape(1, -1)

    ghz_pred = learner.predict(ghz_features)[0]
    product_pred = learner.predict(product_features)[0]

    # GHZ should be distillable, product should not
    assert ghz_pred == 1, "Pure GHZ should be classified as distillable"
    assert product_pred == 0, "Product state should be non-distillable"
```

#### 7. `scripts/plot_results.py` - Add MLP to visualizations

Update `plot_model_comparison()` color map:
```python
colors = {
    'svm': '#3498db',
    'mlp': '#9b59b6',
    'transformer_classifier': '#2ecc71',
    'transformer_hybrid': '#e74c3c'
}
```

---

## Test Flow Integration

### Where MLP Appears

1. **Unit Tests**
   ```bash
   pytest tests/test_mlp_classifier.py -v
   ```

2. **Integration Tests**
   ```bash
   pytest tests/test_integration.py::TestIntegration::test_3qubit_distillability_pipeline_mlp -v
   ```

3. **Comparative Analysis**
   ```bash
   python scripts/run_comparative_analysis.py --n-samples 2000
   ```
   - MLP automatically included in comparison
   - Appears in ROC curves, metrics bar charts, confusion matrices

4. **All Tests**
   ```bash
   pytest tests/ -v
   ```

---

## Summary

| File | Lines | Action |
|------|-------|--------|
| `src/ml_models/mlp_classifier.py` | ~300 | **New** |
| `tests/test_mlp_classifier.py` | ~250 | **New** |
| `src/config.py` | +15 | Add MLPConfig |
| `src/ml_models/__init__.py` | +10 | Export MLP |
| `scripts/run_comparative_analysis.py` | +30 | Add MLP evaluation |
| `tests/test_integration.py` | +40 | Add MLP integration test |
| `scripts/plot_results.py` | +5 | Add MLP color |

**Total new code: ~550 lines** (vs. ~2000 in previous plan)

---

## Expected Results

| Model | Expected Accuracy | Parameters |
|-------|-------------------|------------|
| SVM | ~85% | ~500 support vectors |
| MLP | ~90-95% | ~15,000 parameters |
| Transformer | ~99% | ~5,000 parameters |

MLP should:
- Outperform linear SVM (non-linear decision boundary)
- Be comparable to or slightly below Transformer
- Train faster than Transformer (simpler architecture)
- Provide a strong deep learning baseline
