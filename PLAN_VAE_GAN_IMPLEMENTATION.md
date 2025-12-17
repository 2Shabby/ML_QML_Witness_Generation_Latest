# VAE/GAN Classification Integration Plan

## Overview

This document outlines the detailed implementation plan for extending the ML-QML Witness Generation codebase to include VAE (Variational Autoencoder) and GAN (Generative Adversarial Network) classification in the entire test flows.

## Current Architecture Summary

### Existing ML Models
1. **SVMWitnessLearner** (`src/ml_models/svm_witness.py`)
   - Linear SVM for classification
   - Achieves 85.3% accuracy
   - Extracts interpretable witness operator W = Σ wₖPₖ

2. **TransformerWitnessLearner** (`src/ml_models/transformer_witness.py`)
   - Two modes: `classifier` and `hybrid`
   - Achieves 99.7-100% accuracy
   - Hybrid mode extracts witness coefficients

### Common Interface Methods
All learners share this interface:
- `train(X, y, test_size, verbose)` / `fit(X_train, y_train, X_val, y_val)`
- `predict(X)` → binary labels
- `predict_proba(X)` → probability matrix
- `decision_function(X)` → raw scores
- `get_witness_operator()` → SparsePauliOp
- `get_sparse_witness(threshold)` → SparsePauliOp
- `get_measurement_cost()` → int

### Test Infrastructure
- **Unit tests**: `tests/test_*.py` (56 tests total)
- **Integration tests**: `tests/test_integration.py`
- **Experiment scripts**: `scripts/run_experiments.py`, `scripts/run_transformer_experiments.py`
- **Analysis scripts**: `scripts/run_comparative_analysis.py`
- **Visualization**: `scripts/plot_results.py`

---

## Implementation Plan

### Phase 1: Configuration and Base Classes

#### 1.1 Add VAE/GAN Configuration (`src/config.py`)

```python
# Add to src/config.py

@dataclass
class VAEConfig:
    """Configuration for VAE witness learner."""
    # Encoder architecture
    encoder_dims: List[int] = field(default_factory=lambda: [36, 64, 32])
    latent_dim: int = 16

    # Decoder architecture
    decoder_dims: List[int] = field(default_factory=lambda: [32, 64, 36])

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 64
    n_epochs: int = 100
    patience: int = 15
    beta: float = 1.0  # KL divergence weight (beta-VAE)

    # Classification head
    classifier_dims: List[int] = field(default_factory=lambda: [16, 8])
    dropout: float = 0.1


@dataclass
class GANConfig:
    """Configuration for GAN witness learner."""
    # Generator architecture
    generator_dims: List[int] = field(default_factory=lambda: [16, 32, 64, 36])
    noise_dim: int = 16

    # Discriminator architecture
    discriminator_dims: List[int] = field(default_factory=lambda: [36, 64, 32, 1])

    # Training
    learning_rate_g: float = 2e-4
    learning_rate_d: float = 2e-4
    batch_size: int = 64
    n_epochs: int = 100
    n_critic: int = 5  # Discriminator steps per generator step

    # Classification mode
    use_conditional: bool = True  # Conditional GAN for classification
    classifier_dims: List[int] = field(default_factory=lambda: [64, 32])


DEFAULT_VAE_CONFIG = VAEConfig()
DEFAULT_GAN_CONFIG = GANConfig()
```

#### 1.2 Exports Update (`src/config.py` `__all__`)

Add to `__all__`:
- `'VAEConfig'`
- `'GANConfig'`
- `'DEFAULT_VAE_CONFIG'`
- `'DEFAULT_GAN_CONFIG'`

---

### Phase 2: VAE/GAN Model Implementation

#### 2.1 Create `src/ml_models/vae_gan_witness.py`

**File structure:**
```python
"""
VAE and GAN Witness Learners

Implements:
1. VAEClassifier: Variational autoencoder with classification head
2. GANClassifier: Conditional GAN for classification + generation
3. VAEGANWitnessLearner: Unified wrapper matching existing API
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from qiskit.quantum_info import PauliList, SparsePauliOp
from typing import Dict, Optional, List, Tuple
import logging

from ..config import DEFAULT_VAE_CONFIG, DEFAULT_GAN_CONFIG
from ..utils import set_seed, get_split_seed

logger = logging.getLogger(__name__)
```

#### 2.2 VAE Architecture Components

```python
class Encoder(nn.Module):
    """VAE Encoder: maps input features to latent distribution parameters."""

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.LeakyReLU(0.2),
            ])
            prev_dim = dim

        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """VAE Decoder: maps latent samples back to feature space."""

    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()

        layers = []
        prev_dim = latent_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.LeakyReLU(0.2),
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, output_dim))
        # No activation - Pauli features can be any real value in [-1, 1]

        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class VAEClassifier(nn.Module):
    """
    VAE with classification head for quantum state classification.

    Architecture:
    - Encoder: 36D features → latent distribution (μ, σ)
    - Reparameterization: z = μ + σ * ε
    - Decoder: z → reconstructed features
    - Classifier: z → binary classification

    Loss: reconstruction + β*KL + classification
    """

    def __init__(
        self,
        n_features: int = 36,
        encoder_dims: List[int] = [64, 32],
        latent_dim: int = 16,
        decoder_dims: List[int] = [32, 64],
        classifier_dims: List[int] = [16, 8],
        beta: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_features = n_features
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder
        self.encoder = Encoder(n_features, encoder_dims, latent_dim)

        # Decoder
        self.decoder = Decoder(latent_dim, decoder_dims, n_features)

        # Classification head (from latent space)
        classifier_layers = []
        prev_dim = latent_dim
        for dim in classifier_dims:
            classifier_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        classifier_layers.append(nn.Linear(prev_dim, 2))

        self.classifier = nn.Sequential(*classifier_layers)

        # Store for witness extraction
        self.latent_classifier_weights = None

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for backprop through sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            x_recon: Reconstructed features
            mu: Latent mean
            logvar: Latent log-variance
            logits: Classification logits
        """
        # Encode
        mu, logvar = self.encoder(x)

        # Sample latent
        z = self.reparameterize(mu, logvar)

        # Decode
        x_recon = self.decoder(z)

        # Classify
        logits = self.classifier(z)

        return x_recon, mu, logvar, logits

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Classification only (for inference)."""
        mu, _ = self.encoder(x)
        return self.classifier(mu)

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation (mean)."""
        mu, _ = self.encoder(x)
        return mu
```

#### 2.3 GAN Architecture Components

```python
class Generator(nn.Module):
    """
    Conditional Generator for GAN.

    Maps (noise, label) → fake Pauli features
    """

    def __init__(
        self,
        noise_dim: int = 16,
        n_classes: int = 2,
        hidden_dims: List[int] = [32, 64],
        output_dim: int = 36
    ):
        super().__init__()

        # Embed class label
        self.label_emb = nn.Embedding(n_classes, noise_dim)

        # Generator network
        layers = []
        prev_dim = noise_dim * 2  # noise + label embedding
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())  # Pauli features typically in [-1, 1]

        self.generator = nn.Sequential(*layers)

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_emb = self.label_emb(labels)
        x = torch.cat([noise, label_emb], dim=1)
        return self.generator(x)


class Discriminator(nn.Module):
    """
    Conditional Discriminator for GAN.

    Maps (features, label) → real/fake probability
    Also serves as classifier when label is unknown.
    """

    def __init__(
        self,
        input_dim: int = 36,
        n_classes: int = 2,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.3
    ):
        super().__init__()

        # Embed class label
        self.label_emb = nn.Embedding(n_classes, input_dim)

        # Discriminator network
        layers = []
        prev_dim = input_dim * 2  # features + label embedding
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, 1))

        self.discriminator = nn.Sequential(*layers)

        # Auxiliary classifier (for AC-GAN style)
        self.aux_classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], n_classes)
        )
        self.hidden_layers = nn.Sequential(*layers[:-1])

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_emb = self.label_emb(labels)
        x = torch.cat([x, label_emb], dim=1)
        return self.discriminator(torch.cat([x[:, :36], x[:, 36:]], dim=1))

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Auxiliary classification."""
        # Use discriminator features for classification
        h = self.hidden_layers(torch.cat([x, torch.zeros_like(x)], dim=1))
        return self.aux_classifier(h)


class GANClassifier(nn.Module):
    """
    Conditional GAN for classification and data augmentation.

    Can be used for:
    1. Classification via discriminator's auxiliary head
    2. Generating synthetic training samples
    3. Adversarial robustness testing
    """

    def __init__(
        self,
        n_features: int = 36,
        noise_dim: int = 16,
        generator_dims: List[int] = [32, 64],
        discriminator_dims: List[int] = [64, 32],
        n_classes: int = 2
    ):
        super().__init__()

        self.n_features = n_features
        self.noise_dim = noise_dim
        self.n_classes = n_classes

        self.generator = Generator(
            noise_dim=noise_dim,
            n_classes=n_classes,
            hidden_dims=generator_dims,
            output_dim=n_features
        )

        self.discriminator = Discriminator(
            input_dim=n_features,
            n_classes=n_classes,
            hidden_dims=discriminator_dims
        )

    def generate(self, n_samples: int, labels: torch.Tensor = None, device='cpu') -> torch.Tensor:
        """Generate synthetic samples."""
        noise = torch.randn(n_samples, self.noise_dim, device=device)
        if labels is None:
            labels = torch.randint(0, self.n_classes, (n_samples,), device=device)
        return self.generator(noise, labels)

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Classify using discriminator's auxiliary head."""
        return self.discriminator.classify(x)
```

#### 2.4 Unified Wrapper Class

```python
class VAEGANWitnessLearner:
    """
    Wrapper class providing same API as SVMWitnessLearner for VAE/GAN models.

    Supports three modes:
    - 'vae': VAE classifier with latent space
    - 'gan': Conditional GAN with auxiliary classifier
    - 'vae_gan': Combined approach (VAE + GAN regularization)

    Example usage:
        learner = VAEGANWitnessLearner(
            pauli_basis=basis,
            mode='vae',
            latent_dim=16
        )
        metrics = learner.train(X, y)
        witness = learner.get_witness_operator()
    """

    def __init__(
        self,
        pauli_basis: PauliList,
        mode: str = 'vae',
        # VAE parameters
        encoder_dims: List[int] = None,
        latent_dim: int = 16,
        decoder_dims: List[int] = None,
        beta: float = 1.0,
        # GAN parameters
        noise_dim: int = 16,
        generator_dims: List[int] = None,
        discriminator_dims: List[int] = None,
        # Training parameters
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        n_epochs: int = 100,
        patience: int = 15,
        dropout: float = 0.1,
        random_state: Optional[int] = None,
        device: Optional[str] = None
    ):
        self.pauli_basis = pauli_basis
        self.mode = mode
        self.n_features = len(pauli_basis)

        # Default architecture parameters
        if encoder_dims is None:
            encoder_dims = [64, 32]
        if decoder_dims is None:
            decoder_dims = [32, 64]
        if generator_dims is None:
            generator_dims = [32, 64]
        if discriminator_dims is None:
            discriminator_dims = [64, 32]

        # Store parameters
        self.latent_dim = latent_dim
        self.beta = beta
        self.noise_dim = noise_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.patience = patience
        self.dropout = dropout
        self.random_state = random_state

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Set random seed
        if random_state is not None:
            set_seed(random_state)

        # Initialize model based on mode
        self._init_model(encoder_dims, decoder_dims, generator_dims, discriminator_dims)

        self.is_trained = False
        self.witness_operator = None
        self.bias = 0.0
        self.metrics = {}
        self.training_history = []

    def _init_model(self, encoder_dims, decoder_dims, generator_dims, discriminator_dims):
        """Initialize model based on mode."""
        if self.mode == 'vae':
            self.model = VAEClassifier(
                n_features=self.n_features,
                encoder_dims=encoder_dims,
                latent_dim=self.latent_dim,
                decoder_dims=decoder_dims,
                beta=self.beta,
                dropout=self.dropout
            )
        elif self.mode == 'gan':
            self.model = GANClassifier(
                n_features=self.n_features,
                noise_dim=self.noise_dim,
                generator_dims=generator_dims,
                discriminator_dims=discriminator_dims
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'vae' or 'gan'")

        self.model = self.model.to(self.device)

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
              verbose: bool = True) -> Dict[str, float]:
        """Train the model."""
        # Implementation follows same pattern as TransformerWitnessLearner
        # Split data, create DataLoaders, run training loop
        # ...
        pass

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray, verbose: bool = True) -> Dict[str, float]:
        """Train on pre-split data."""
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        pass

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function values."""
        pass

    def get_witness_operator(self) -> SparsePauliOp:
        """
        Extract witness operator from trained model.

        For VAE: Uses linear approximation from latent→classifier mapping
        For GAN: Uses discriminator feature importance
        """
        pass

    def get_sparse_witness(self, threshold: float = 0.01) -> SparsePauliOp:
        """Get sparse version of witness."""
        pass

    def get_measurement_cost(self) -> int:
        """Estimate measurement settings required."""
        pass

    def get_latent_representation(self, X: np.ndarray) -> np.ndarray:
        """Get latent space representation (VAE only)."""
        pass

    def generate_samples(self, n_samples: int, labels: np.ndarray = None) -> np.ndarray:
        """Generate synthetic samples (GAN only)."""
        pass

    def save(self, path: str):
        """Save model to file."""
        pass

    def load(self, path: str):
        """Load model from file."""
        pass
```

---

### Phase 3: Test Implementation

#### 3.1 Create `tests/test_vae_gan_witness.py`

**Test structure (matching existing test patterns):**

```python
"""
Tests for VAE and GAN Witness Learners

Tests both VAEClassifier and GANClassifier models.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Skip all tests if torch not available
torch_available = True
try:
    import torch
    from src.ml_models.vae_gan_witness import (
        VAEGANWitnessLearner,
        VAEClassifier,
        GANClassifier,
        Encoder,
        Decoder,
        Generator,
        Discriminator
    )
except ImportError:
    torch_available = False

from src.feature_extraction.pauli_features import create_sparse_measurement_set

pytestmark = pytest.mark.skipif(
    not torch_available,
    reason="PyTorch not installed"
)


class TestVAEComponents:
    """Test VAE architecture components."""

    def test_encoder_output_shape(self):
        """Test encoder produces correct mu, logvar shapes."""
        pass

    def test_decoder_output_shape(self):
        """Test decoder produces correct output shape."""
        pass

    def test_reparameterization(self):
        """Test reparameterization trick works correctly."""
        pass


class TestGANComponents:
    """Test GAN architecture components."""

    def test_generator_output_shape(self):
        """Test generator produces correct output shape."""
        pass

    def test_discriminator_output_shape(self):
        """Test discriminator produces correct output shape."""
        pass

    def test_conditional_generation(self):
        """Test conditional generation respects class labels."""
        pass


class TestVAEClassifier:
    """Test VAEClassifier model."""

    def test_forward_pass(self):
        """Test forward pass produces all outputs."""
        pass

    def test_reconstruction_loss(self):
        """Test reconstruction is reasonable after training."""
        pass

    def test_latent_space(self):
        """Test latent space has useful structure."""
        pass


class TestGANClassifier:
    """Test GANClassifier model."""

    def test_generation(self):
        """Test sample generation."""
        pass

    def test_classification(self):
        """Test auxiliary classification."""
        pass


class TestVAEGANWitnessLearner:
    """Test VAEGANWitnessLearner wrapper class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        n_samples = 200
        n_features = 36
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        basis = create_sparse_measurement_set(3, 'two_body')
        return X, y, basis

    def test_vae_mode_training(self, sample_data):
        """Test training in VAE mode."""
        pass

    def test_gan_mode_training(self, sample_data):
        """Test training in GAN mode."""
        pass

    def test_prediction(self, sample_data):
        """Test prediction after training."""
        pass

    def test_predict_proba(self, sample_data):
        """Test probability prediction."""
        pass

    def test_witness_extraction_vae(self, sample_data):
        """Test witness extraction from VAE."""
        pass

    def test_sparse_witness(self, sample_data):
        """Test sparse witness extraction."""
        pass

    def test_latent_representation(self, sample_data):
        """Test latent representation extraction."""
        pass

    def test_sample_generation(self, sample_data):
        """Test sample generation from GAN."""
        pass


class TestVAEGANIntegration:
    """Integration tests with quantum state data."""

    def test_with_quantum_features(self):
        """Test with actual quantum state features."""
        pass

    def test_comparable_to_svm(self):
        """Test that VAE/GAN is comparable to SVM on quantum data."""
        pass

    def test_comparable_to_transformer(self):
        """Test that VAE/GAN is comparable to Transformer."""
        pass
```

#### 3.2 Update Integration Tests (`tests/test_integration.py`)

Add new test method:

```python
def test_3qubit_distillability_pipeline_vae_gan(self):
    """
    Test end-to-end 3-qubit distillability with VAE/GAN.

    Mirrors test_3qubit_distillability_pipeline but uses VAE/GAN model.
    """
    # Same setup as SVM test
    # Train VAEGANWitnessLearner in 'vae' mode
    # Verify accuracy > 0.55
    # Extract and verify witness operator
    pass
```

---

### Phase 4: Experiment Scripts

#### 4.1 Create `scripts/run_vae_gan_experiments.py`

**Structure (matching existing scripts):**

```python
#!/usr/bin/env python3
"""
VAE/GAN Experimental Pipeline for 3-Qubit Distillability

Experiments:
1. comparison: Compare VAE, GAN, SVM, Transformer
2. cv: Cross-validation with multiple seeds
3. family: Per-family analysis (GHZ, W, cluster, random, product)
4. ablation: 36D vs 63D feature comparison
5. latent: Latent space analysis (VAE)
6. generation: Data augmentation via generation (GAN)
7. witness: Witness extraction comparison
8. all: Run all experiments

Usage:
    python scripts/run_vae_gan_experiments.py --experiment comparison
    python scripts/run_vae_gan_experiments.py --experiment all --n-samples 5000
"""

# Experiment functions:
# - run_model_comparison()
# - run_cross_validation_comparison()
# - run_per_family_comparison()
# - run_ablation_comparison()
# - run_latent_space_analysis()  # New: VAE-specific
# - run_generation_experiment()  # New: GAN-specific
# - run_witness_analysis()
# - run_all_experiments()
```

#### 4.2 Update `scripts/run_comparative_analysis.py`

Add VAE/GAN to ModelEvaluator:

```python
def evaluate_vae(self) -> Optional[Dict]:
    """Train and evaluate VAE model."""
    pass

def evaluate_gan(self) -> Optional[Dict]:
    """Train and evaluate GAN model."""
    pass
```

---

### Phase 5: Visualization Updates

#### 5.1 Update `scripts/plot_results.py`

Add new plot functions:

```python
def plot_latent_space(results: Dict, save_path: Optional[Path] = None):
    """
    Plot VAE latent space visualization.
    - t-SNE or UMAP projection
    - Color by class label
    - Show decision boundary
    """
    pass

def plot_generated_samples(results: Dict, save_path: Optional[Path] = None):
    """
    Plot GAN generated samples vs real samples.
    - Feature distribution comparison
    - Quality metrics
    """
    pass

def plot_vae_gan_comparison(results: Dict, save_path: Optional[Path] = None):
    """
    Plot VAE/GAN comparison with SVM and Transformer.
    - Accuracy bar chart
    - ROC curves
    - Confusion matrices
    """
    pass
```

Update `plot_all_from_directory()` to include VAE/GAN results.

---

### Phase 6: Module Exports

#### 6.1 Update `src/ml_models/__init__.py`

```python
# Conditionally import VAE/GAN models (requires PyTorch)
try:
    from .vae_gan_witness import (
        VAEGANWitnessLearner,
        VAEClassifier,
        GANClassifier
    )
    _VAE_GAN_AVAILABLE = True
except ImportError:
    _VAE_GAN_AVAILABLE = False
    VAEGANWitnessLearner = None
    VAEClassifier = None
    GANClassifier = None

# Add to __all__
if _VAE_GAN_AVAILABLE:
    __all__.extend([
        'VAEGANWitnessLearner',
        'VAEClassifier',
        'GANClassifier',
    ])
```

---

## File Changes Summary

### New Files
1. `src/ml_models/vae_gan_witness.py` (~800-1000 lines)
2. `tests/test_vae_gan_witness.py` (~400-500 lines)
3. `scripts/run_vae_gan_experiments.py` (~600-800 lines)

### Modified Files
1. `src/config.py` - Add VAEConfig, GANConfig dataclasses
2. `src/ml_models/__init__.py` - Export new models
3. `tests/test_integration.py` - Add VAE/GAN integration test
4. `scripts/plot_results.py` - Add VAE/GAN visualization
5. `scripts/run_comparative_analysis.py` - Add VAE/GAN evaluation

---

## Test Flow Integration Points

### Where VAE/GAN Appears in Test Flows

1. **Unit Tests** (`pytest tests/test_vae_gan_witness.py`)
   - Component tests (Encoder, Decoder, Generator, Discriminator)
   - Model tests (VAEClassifier, GANClassifier)
   - Wrapper tests (VAEGANWitnessLearner)

2. **Integration Tests** (`pytest tests/test_integration.py`)
   - `test_3qubit_distillability_pipeline_vae_gan`

3. **Experiment Scripts**
   - `scripts/run_vae_gan_experiments.py --experiment all`
   - `scripts/run_comparative_analysis.py` (includes VAE/GAN)

4. **Visualization**
   - `scripts/plot_results.py --plot all` (includes VAE/GAN results)

---

## Implementation Order

1. **Day 1**: Configuration + Base VAE architecture
2. **Day 2**: Complete VAE implementation + tests
3. **Day 3**: GAN architecture + tests
4. **Day 4**: Unified wrapper + witness extraction
5. **Day 5**: Experiment scripts + visualization
6. **Day 6**: Integration tests + documentation

---

## Expected Outcomes

### Performance Targets
- VAE accuracy: >80% (comparable to SVM baseline)
- GAN auxiliary classifier: >75%
- Generated sample quality: Pauli feature distributions match real data

### Key Benefits
1. **Latent representation**: VAE provides compressed representation of quantum states
2. **Data augmentation**: GAN can generate synthetic training data
3. **Interpretability**: Witness extraction from both models
4. **Robustness**: Ensemble potential with existing models

---

## Notes

- VAE witness extraction approximates the mapping from features to classification
- GAN focuses on generation quality; classification is auxiliary
- Both require PyTorch; follow same conditional import pattern as Transformer
- Test fixtures follow existing `@pytest.fixture` pattern
- Results JSON format matches existing experiment output schema
