# Feasibility Assessment: Experimental Upgrades

## Project Structure Verification

The codebase structure matches expectations:
- `scripts/run_experiments.py` - Linear SVM experiments ✓
- `scripts/run_transformer_experiments.py` - Transformer experiments ✓  
- `scripts/plot_results.py` - Visualization ✓
- `tests/` - Test suite ✓

Key source files:
- `src/quantum_states/state_generation.py` - State generation with noise
- `src/ml_models/svm_witness.py` - Linear SVM (scikit-learn)
- `src/ml_models/mlp_classifier.py` - MLP Discriminator (PyTorch)
- `src/ml_models/transformer_witness.py` - Transformer (PyTorch)
- `src/feature_extraction/pauli_features.py` - 36D Pauli feature extraction

---

## Upgrade A: Noise Mismatch Robustness Study

**Goal:** Train on one noise model, test on another to measure generalization gap.

### Code Locations

| File:Line | Description |
|-----------|-------------|
| `src/quantum_states/state_generation.py:145-148` | Depolarizing noise applied in `generate_entangled_state()` |
| `src/quantum_states/state_generation.py:256-259` | Depolarizing noise applied in `generate_dataset()` |
| `src/quantum_states/state_generation.py:335-338` | Depolarizing noise in `generate_noisy_cluster_state()` |
| `src/quantum_states/state_generation.py:386-501` | Main dataset generation function |

### Current State

**What exists:**
- Depolarizing noise channel implementation: `rho = (1 - noise_level) * rho + noise_level * identity/dim`
- Parameterized `noise_range` tuple for controlling noise levels
- Dataset generation supports custom noise ranges

**What's missing:**
- Dephasing noise channel
- Amplitude damping noise channel
- Ability to specify noise *type* (only noise *level* is parameterized)
- Cross-noise evaluation infrastructure

### Feasibility Assessment

| Aspect | Assessment |
|--------|------------|
| **Engineering effort** | **Medium** |
| **Estimated changes** | ~120 lines across 2 files |
| **Dependencies** | None (Qiskit already available) |
| **Blockers** | None |

### Implementation Sketch

1. **Add noise channel functions** (~40 lines in `state_generation.py`):
```python
def apply_dephasing_noise(rho: np.ndarray, p: float) -> np.ndarray:
    """Apply dephasing channel: ρ → (1-p)ρ + p Z ρ Z"""
    # Implement per-qubit dephasing
    
def apply_amplitude_damping(rho: np.ndarray, gamma: float) -> np.ndarray:
    """Apply amplitude damping: Kraus operators K0, K1"""
    # Implement amplitude damping channel
```

2. **Add noise_type parameter** to `generate_distillability_dataset()` (~20 lines):
```python
def generate_distillability_dataset(
    n_samples: int = 5000,
    noise_range: Tuple[float, float] = (0.0, 0.5),
    noise_type: str = 'depolarizing',  # NEW: 'depolarizing', 'dephasing', 'amplitude_damping'
    seed: Optional[int] = None
) -> Tuple[List[DensityMatrix], np.ndarray]:
```

3. **Create cross-noise evaluation script** (~60 lines):
```python
# scripts/run_noise_mismatch.py
def run_noise_mismatch_study():
    # Train on depolarizing p∈[0,0.3]
    train_states, train_labels = generate_distillability_dataset(
        noise_range=(0.0, 0.3), noise_type='depolarizing'
    )
    # Test on: depolarizing high-noise, dephasing, amplitude damping
    # Output: 3×3 accuracy matrix
```

### Risk Factors

- Amplitude damping is non-unital and may significantly alter class balance (distillable fraction changes)
- Need to verify NPT oracle remains valid for non-depolarizing noise
- Cross-noise results may show low transfer (expected behavior, not a bug)

---

## Upgrade C: Sample Efficiency / Data Budget Curves

**Goal:** Train with subsets (1%, 5%, 10%, 50%, 100%) of training data.

### Code Locations

| File:Line | Description |
|-----------|-------------|
| `src/ml_models/svm_witness.py:103-106` | Train/test split with stratification |
| `src/ml_models/mlp_classifier.py:182-184` | MLP train/test split |
| `src/ml_models/transformer_witness.py:720-723` | Transformer train/test split |
| `scripts/run_transformer_experiments.py:278-279` | External split before training |

### Current State

**What exists:**
- Stratified train/test split via `train_test_split(stratify=y)`
- All models support `.fit(X_train, y_train, X_val, y_val)` pattern
- Fixed 80/20 train/test ratio by default

**What's missing:**
- Training data subsampling utility
- Data budget sweep loop
- Fixed test set with variable training size

### Feasibility Assessment

| Aspect | Assessment |
|--------|------------|
| **Engineering effort** | **Low** |
| **Estimated changes** | ~80 lines across 2 files |
| **Dependencies** | None |
| **Blockers** | None |

### Implementation Sketch

1. **Add subsampling utility** (~15 lines in `src/utils/__init__.py`):
```python
def subsample_training_data(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    fraction: float,
    seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified subsampling of training data."""
    if fraction >= 1.0:
        return X_train, y_train
    from sklearn.model_selection import train_test_split
    _, X_sub, _, y_sub = train_test_split(
        X_train, y_train, test_size=fraction, 
        random_state=seed, stratify=y_train
    )
    return X_sub, y_sub
```

2. **Create data budget experiment** (~65 lines in `scripts/run_data_budget.py`):
```python
DATA_FRACTIONS = [0.01, 0.05, 0.1, 0.5, 1.0]

def run_data_budget_experiment():
    # Generate full dataset
    X, y = generate_dataset(n_samples=5000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    results = {}
    for fraction in DATA_FRACTIONS:
        X_sub, y_sub = subsample_training_data(X_train, y_train, fraction)
        for model_name, model in models.items():
            model.fit(X_sub, y_sub, X_test, y_test)
            acc = accuracy_score(y_test, model.predict(X_test))
            results[model_name][fraction] = acc
    
    # Plot: accuracy vs training_fraction per model
```

### Risk Factors

- 1% of 4000 training samples = 40 samples — may cause training instability for transformers
- Stratified sampling may fail if subset is too small for minority class
- Multiple seeds recommended for error bars at low-data regime

---

## Upgrade D: Latency / Compute Cost Analysis

**Goal:** Measure inference time and create accuracy-vs-latency Pareto frontier.

### Code Locations

| File:Line | Description |
|-----------|-------------|
| `src/ml_models/svm_witness.py:179-192` | SVM `predict()` method |
| `src/ml_models/mlp_classifier.py:325-334` | MLP `predict()` method |
| `src/ml_models/transformer_witness.py:831-851` | Transformer `predict()` method |
| `src/ml_models/mlp_classifier.py:319-322` | MLP parameter count via `p.numel()` |
| `src/ml_models/transformer_witness.py:645` | Transformer parameter count |

### Current State

**What exists:**
- Parameter count computed for MLP and Transformer (`n_parameters`)
- PyTorch models on device (CPU/GPU configurable)
- scikit-learn SVM with standard predict interface
- Models already have device argument (defaults to CPU if no GPU)

**What's missing:**
- Timing infrastructure
- Memory footprint measurement
- Batch vs single-sample inference benchmarking
- Pareto plotting code

### Feasibility Assessment

| Aspect | Assessment |
|--------|------------|
| **Engineering effort** | **Low** |
| **Estimated changes** | ~100 lines across 2 files |
| **Dependencies** | None (time module is stdlib) |
| **Blockers** | None |

### Implementation Sketch

1. **Add timing utilities** (~30 lines in `src/utils/__init__.py`):
```python
import time

def measure_inference_time(
    model, X: np.ndarray, n_warmup: int = 10, n_runs: int = 100
) -> Dict[str, float]:
    """Measure inference latency."""
    # Warmup
    for _ in range(n_warmup):
        _ = model.predict(X[:1])
    
    # Single sample timing
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict(X[:1])
        times.append(time.perf_counter() - start)
    
    # Batch timing
    batch_start = time.perf_counter()
    _ = model.predict(X)
    batch_time = time.perf_counter() - batch_start
    
    return {
        'single_sample_ms': np.mean(times) * 1000,
        'single_sample_std_ms': np.std(times) * 1000,
        'batch_total_ms': batch_time * 1000,
        'batch_per_sample_ms': (batch_time / len(X)) * 1000,
    }
```

2. **Add memory measurement** (~20 lines):
```python
def get_model_memory_footprint(model) -> Dict[str, float]:
    """Estimate model memory in MB."""
    if hasattr(model, 'model'):  # PyTorch wrapper
        params = sum(p.numel() * p.element_size() for p in model.model.parameters())
        buffers = sum(b.numel() * b.element_size() for b in model.model.buffers())
        return {'memory_mb': (params + buffers) / 1e6}
    elif hasattr(model, 'svm'):  # SVM
        # Approximate: support vectors × features × 8 bytes
        n_sv = len(model.svm.support_vectors_)
        n_feat = model.svm.support_vectors_.shape[1]
        return {'memory_mb': (n_sv * n_feat * 8) / 1e6}
```

3. **Create latency benchmark script** (~50 lines in `scripts/run_latency_analysis.py`):
```python
def run_latency_analysis():
    # Train all models
    # Measure: inference_time, param_count, memory
    # Plot: Pareto frontier (accuracy vs latency)
    # Output: Table with all metrics
```

### Risk Factors

- GPU timing requires `torch.cuda.synchronize()` — focus on CPU for realistic deployment
- First inference may be slow due to JIT compilation (warmup required)
- Memory measurement is approximate for SVM

---

## Upgrade E: Calibration / Confidence Reliability

**Goal:** Assess whether model confidence correlates with correctness.

### Code Locations

| File:Line | Description |
|-----------|-------------|
| `src/ml_models/svm_witness.py:64` | SVM has `probability=True` (Platt scaling enabled) |
| `src/ml_models/svm_witness.py:194-207` | SVM `predict_proba()` method |
| `src/ml_models/mlp_classifier.py:336-346` | MLP `predict_proba()` with softmax |
| `src/ml_models/transformer_witness.py:853-873` | Transformer `predict_proba()` with softmax |

### Current State

**What exists:**
- All models output probabilities:
  - **SVM**: Platt scaling calibration (`probability=True`)
  - **MLP**: Softmax over 2-class logits (`torch.softmax(logits, dim=1)`)
  - **Transformer**: Softmax over 2-class logits
- `predict_proba()` returns shape `(n_samples, 2)` for all models

**What's missing:**
- ECE (Expected Calibration Error) computation
- Reliability diagram plotting
- Confidence histogram plotting
- Calibration comparison infrastructure

### Feasibility Assessment

| Aspect | Assessment |
|--------|------------|
| **Engineering effort** | **Very Low** |
| **Estimated changes** | ~80 lines across 2 files |
| **Dependencies** | None (matplotlib already in requirements) |
| **Blockers** | None |

### Implementation Sketch

1. **Add ECE computation** (~30 lines in `src/utils/__init__.py`):
```python
def compute_ece(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    n_bins: int = 10
) -> Tuple[float, Dict]:
    """
    Compute Expected Calibration Error.
    
    ECE = Σ (|B_m|/n) |acc(B_m) - conf(B_m)|
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []
    
    confidences = y_prob.max(axis=1)  # Max probability
    predictions = y_prob.argmax(axis=1)
    accuracies = (predictions == y_true)
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if in_bin.sum() > 0:
            bin_acc = accuracies[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            bin_size = in_bin.sum()
            ece += (bin_size / len(y_true)) * abs(bin_acc - bin_conf)
            bin_data.append({'bin': i, 'acc': bin_acc, 'conf': bin_conf, 'count': bin_size})
    
    return ece, bin_data
```

2. **Add reliability diagram plotting** (~25 lines):
```python
def plot_reliability_diagram(bin_data: List[Dict], model_name: str, ax=None):
    """Plot reliability diagram (calibration curve)."""
    if ax is None:
        fig, ax = plt.subplots()
    
    confidences = [b['conf'] for b in bin_data]
    accuracies = [b['acc'] for b in bin_data]
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.bar(confidences, accuracies, width=0.08, alpha=0.7, label=model_name)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.legend()
```

3. **Create calibration analysis script** (~25 lines in `scripts/run_calibration_analysis.py`):
```python
def run_calibration_analysis():
    # Train all models
    for model_name, model in models.items():
        y_prob = model.predict_proba(X_test)
        ece, bin_data = compute_ece(y_test, y_prob)
        results[model_name] = {'ece': ece, 'bins': bin_data}
        
    # Plot: reliability diagrams, confidence histograms
    # Output: ECE table
```

### Risk Factors

- Platt scaling on SVM may be overconfident on small datasets
- 10 bins may be too many for small test sets — consider adaptive binning
- Neural networks often overconfident by default

---

## Summary Table

| Upgrade | Effort | Lines | Files | Dependencies | Key Challenge |
|---------|--------|-------|-------|--------------|---------------|
| **A: Noise Mismatch** | Medium | ~120 | 2 | None | Implementing dephasing/amplitude damping channels |
| **C: Data Budget** | Low | ~80 | 2 | None | Handling very small subsets (1%) |
| **D: Latency Analysis** | Low | ~100 | 2 | None | Accurate timing (warmup needed) |
| **E: Calibration** | Very Low | ~80 | 2 | None | ECE interpretation at bin boundaries |

## Recommended Priority Order

1. **Upgrade E (Calibration)** — Lowest effort, high insight value
2. **Upgrade D (Latency)** — Low effort, practical deployment relevance
3. **Upgrade C (Data Budget)** — Low effort, important for sample efficiency claims
4. **Upgrade A (Noise Mismatch)** — Medium effort, most valuable for robustness claims

## Total Estimated Effort

~380 lines of code across 6-8 files, assuming clean implementation without extensive testing.
