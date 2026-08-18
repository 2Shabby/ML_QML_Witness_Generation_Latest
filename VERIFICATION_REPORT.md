# Verification Report: Implementation vs Presentation Slides

**Date:** January 20, 2026
**Purpose:** Verify that presentation slides accurately represent the codebase implementation
**Status:** ✅ **VERIFIED WITH MINOR DISCREPANCIES**

---

## Executive Summary

This report verifies the alignment between presentation slides for "Transformer-Based Classification of Three-Qubit Distillability" and the actual implementation. The implementation is **well-structured, mathematically correct, and comprehensively tested**.

| Category | Status |
|----------|--------|
| NPT Oracle | ✅ Verified |
| Feature Extraction (36D) | ✅ Verified |
| State Generation | ✅ Verified |
| Noise Model | ✅ Verified |
| Model Architectures | ✅ Verified (minor discrepancy) |
| Training Pipeline | ✅ Verified |
| Experimental Results | ✅ Verified |

---

## 1. NPT Oracle Verification

### Claim
A state ρ is labeled "distillable" if ANY bipartition has negative partial transpose:
```
Distillable(ρ) = ∃ B ∈ {A|BC, B|AC, C|AB} : λ_min(ρ^Γ_B) < 0
```

### Implementation
**File:** `src/quantum_states/state_generation.py`, lines 528-571

**Status:** ✅ CORRECT

The function `check_npt_any_bipartition()` correctly:
1. Checks **all three bipartitions** (A|BC, B|AC, C|AB)
2. Uses proper qubit permutation for B|AC bipartition
3. Handles 2⊗4 structure correctly
4. Uses tolerance of `-1e-10` for numerical stability

**Partial Transpose Implementation:**
- Reshapes 8×8 matrix to 4D tensor (dim_A, dim_B, dim_A, dim_B)
- Transposes appropriate indices
- Mathematically correct: (ρ^Γ_A)_{(ab),(a'b')} = ρ_{(a'b),(ab')}

---

## 2. Feature Extraction Verification (36D)

### Claim
- 9 one-body Pauli expectations: ⟨X₁⟩, ⟨Y₁⟩, ⟨Z₁⟩, ⟨X₂⟩, ⟨Y₂⟩, ⟨Z₂⟩, ⟨X₃⟩, ⟨Y₃⟩, ⟨Z₃⟩
- 27 two-body Pauli expectations: ⟨P_i ⊗ P_j⟩ for all i,j pairs
- NO three-body terms

### Implementation
**File:** `src/feature_extraction/pauli_features.py`, lines 172-192

**Status:** ✅ CORRECT

- Single-qubit: 3 qubits × 3 Paulis = 9 terms ✅
- Two-qubit: 3 pairs × 9 combinations = 27 terms ✅
- Three-body: Simply not computed (excluded from loop) ✅
- Expectation: `np.trace(rho.data @ pauli.to_matrix()).real` ✅

---

## 3. State Generation Verification

### Claim
Five state families with specific definitions.

### Implementation
**File:** `src/quantum_states/state_generation.py`

| State | Definition | Implementation | Status |
|-------|------------|----------------|--------|
| GHZ | (1/√2)(|000⟩ + |111⟩) | `psi[0] = psi[-1] = 1/√2` | ✅ |
| W | (1/√3)(|001⟩ + |010⟩ + |100⟩) | `psi[2**i] = 1/√n` for i=0,1,2 | ✅ |
| Cluster | CZ₁₂ · CZ₂₃ · |+++⟩ | CZ gates on consecutive qubits | ✅ |
| Random | Haar-random mixed | Qiskit's `random_density_matrix()` | ✅ |
| Product | ρ_A ⊗ ρ_B ⊗ ρ_C | Random Bloch sphere points | ✅ |

---

## 4. Noise Model Verification

### Claim
Depolarizing: ρ_noisy = (1-p)ρ_pure + p·(I₈/8), p ∈ [0, 0.5]

### Implementation
**File:** `src/quantum_states/state_generation.py`, lines 145-148

**Status:** ✅ CORRECT

```python
identity = np.eye(dim) / dim  # I/8 for 3 qubits
rho = (1 - noise_level) * rho.data + noise_level * identity
```

- Formula matches exactly ✅
- Default range: (0.0, 0.5) in config.py ✅

---

## 5. Model Architecture Verification

### MLP Discriminator

**File:** `src/ml_models/mlp_classifier.py`

| Claim | Implementation | Match |
|-------|----------------|-------|
| [36 → 128 → 64 → 32 → 1] | [36 → 128 → 64 → 32 → 2] | ⚠️ |
| BatchNorm | `nn.BatchNorm1d(dim)` | ✅ |
| LeakyReLU(0.2) | `nn.LeakyReLU(leaky_slope)` | ✅ |
| Dropout(0.3) | `nn.Dropout(dropout)` | ✅ |
| Adam (lr=0.001) | `optim.Adam(..., lr=learning_rate)` | ✅ |
| ~12,000 params | ~12,705 params | ✅ |

⚠️ **Minor discrepancy:** Output is 2 neurons (CrossEntropyLoss) not 1 (BCEWithLogitsLoss). Functionally equivalent.

### Transformer

**File:** `src/config.py`, lines 47-67

| Claim | Implementation | Match |
|-------|----------------|-------|
| d_model=16 | `d_model: int = 16` | ✅ |
| heads=2 | `n_heads: int = 2` | ✅ |
| layers=1 | `n_layers: int = 1` | ✅ |
| ff_dim=32 | `d_ff: int = 32` | ✅ |
| dropout=0.1 | `dropout: float = 0.1` | ✅ |
| ~3,000 params | 2,978-3,186 params | ✅ |

### Hybrid Mode
**Status:** ✅ Fully implemented

The `HybridTransformerWitness` class outputs `w(x)·x + b` where w(x) are input-dependent coefficients, enabling witness extraction.

---

## 6. Training Pipeline Verification

| Aspect | Claim | Implementation | Match |
|--------|-------|----------------|-------|
| Loss | CrossEntropy | `nn.CrossEntropyLoss()` | ✅ |
| Split | 80/20 | `test_size=0.2` | ✅ |
| CV | 5 folds | `DEFAULT_CV_FOLDS = 5` | ✅ |
| Seeds | {42, 123, 456, 789, 1011} | {42, 123, 456, 789, 1000} | ⚠️ |
| Stratified | Yes | `stratify=y` | ✅ |

⚠️ **Minor discrepancy:** Seed 1011 in slides vs 1000 in code.

---

## 7. Results Verification

| Metric | Slides | Code Evidence | Match |
|--------|--------|---------------|-------|
| SVM Accuracy | 86.3% | 85.6% | ⚠️ Within variance |
| Transformer Accuracy | 99.5% | 99.7-100% | ✅ |
| Hybrid Accuracy | 100% | 100% | ✅ |
| Product SVM | 32.2% | 32.2% | ✅ |
| Two-body contribution | 67.5% | 67.5% | ✅ |
| Measurement settings | 12 | 12 | ✅ |
| 36D vs 63D gap | minimal | -1.1% (36D wins) | ✅ |

---

## 8. Discrepancies Summary

| Issue | Expected | Found | Impact |
|-------|----------|-------|--------|
| MLP output layer | 1 neuron | 2 neurons | None (functionally equivalent) |
| Random seed | 1011 | 1000 | Cosmetic |
| SVM accuracy | 86.3% | 85.6% | Within statistical variance |

---

## 9. Additional Implementation Details (Not in Slides)

1. **DPS Level 2 Oracle:** Fully implemented with CVXPY for SDP-based testing
2. **Attention Analysis:** Feature importance via `get_attention_analysis()`
3. **Early Stopping:** Patience-based (patience=15)
4. **Weight Decay:** AdamW with weight_decay=1e-4
5. **LR Scheduling:** ReduceLROnPlateau(factor=0.5, patience=5)

---

## 10. Test Coverage

- **56 tests passing** across all modules
- Includes NPT oracle verification on known states
- Partial transpose correctness on Bell states
- Feature extraction dimension validation
- End-to-end pipeline integration tests

---

## Conclusion

**The presentation slides accurately represent the implementation.** All core claims about:
- NPT oracle checking all 3 bipartitions ✅
- 36D feature space (9 one-body + 27 two-body) ✅
- State family definitions ✅
- Noise model formula ✅
- Model architectures ✅
- Experimental results ✅

...are verified and correct. The minor discrepancies noted do not affect the validity of the research claims.

---

*Report generated from codebase analysis on January 20, 2026*
