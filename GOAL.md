# Research Goal: Learning Restricted Witnesses for Three-Qubit Distillability

## Problem Statement

**Learning Restricted Witnesses for Distillability of Three-Qubit QEC Resource States**

---

## Motivation

Fault-tolerant quantum computation relies on the continuous availability of useful entangled resource states. In the three-qubit regime—relevant to foundational quantum error-correction codes (e.g., repetition codes, Shor code primitives, and small measurement-based quantum computing fragments)—the presence of entanglement alone is insufficient. States may be entangled yet undistillable (bound entanglement), rendering them operationally useless for quantum error correction (QEC).

**The Fundamental Gap:**

| Need | Reality |
|------|---------|
| Fast certification of QEC-useful states | No closed-form distillability criterion exists |
| Experimentally feasible measurements | Full tomography requires O(d⁴) = O(4096) measurements for 3 qubits |
| Real-time verification | SDP-based tests (DPS hierarchy) are computationally expensive |

**Experimental Constraint:** In many architectures, only single-qubit and two-qubit Pauli observables are natively accessible. Higher-weight measurements (3-body terms like X₁Y₂Z₃) introduce prohibitive noise and overhead.

---

## Core Hypothesis

> The distillability of mixed three-qubit quantum states can be approximately certified using a **linear witness** constructed solely from **one- and two-body Pauli expectation values**, where the witness is learned from data labeled by a numerical SDP oracle.

Equivalently: The decision boundary separating distillable states from separable/bound-entangled states has a meaningful projection onto the low-dimensional space of experimentally accessible 1- and 2-body observables.

---

## Research Question

> **Does there exist a physically measurable, low-weight Hermitian operator W, expressible as:**
>
> W = Σᵢ aᵢ Pᵢ⁽¹⁾ + Σᵢⱼ bᵢⱼ Pᵢ⁽¹⁾Pⱼ⁽¹⁾
>
> **(1-body and 2-body Pauli terms only)**
>
> **whose expectation value ⟨W⟩ reliably distinguishes distillable three-qubit states from non-distillable ones?**

### Scope
- **System size:** 3 qubits (smallest genuine multipartite QEC-relevant setting)
- **Classification:** Binary (distillable vs. non-distillable)
- **Measurements:** Restricted to 1-body + 2-body Paulis (36 features, not 63)
- **Output:** Linear witness operator as `SparsePauliOp`

---

## Methodology

### Phase 1: Data Generation (Labeling Oracle)

**Distillability Criterion:**
- **NPT Proxy (Initial):** State is distillable if ρ^Γ has negative eigenvalues across ANY bipartition (A|BC, B|AC, C|AB)
- **DPS Hierarchy (Full):** SDP-based certification using symmetric extensions

**State Families:**
| Family | Relevance | Expected Behavior |
|--------|-----------|-------------------|
| Noisy GHZ states | Standard QEC resource | Should be distillable above threshold |
| Noisy W states | Robust entanglement | Different noise resilience |
| Noisy cluster states | MBQC resources | QEC-relevant |
| Random mixed states | Boundary exploration | Near decision boundary |
| Bound entangled states | Negative examples | Should NOT be detected as distillable |

### Phase 2: Feature Extraction (Restricted Basis)

**Feature Space:**
```
1-body: {X₁, Y₁, Z₁, X₂, Y₂, Z₂, X₃, Y₃, Z₃}           → 9 features
2-body: {X₁X₂, X₁Y₂, ..., Y₂Z₃, Z₂Z₃}                  → 27 features
                                                        ─────────────
Total:                                                    36 features
```

**Excluded (3-body):** {X₁X₂X₃, X₁X₂Y₃, ..., Z₁Z₂Z₃} → 27 terms excluded

**Rationale:** 3-body measurements require simultaneous 3-qubit gates, which are:
- Not natively available on most hardware
- Subject to higher noise accumulation
- Experimentally expensive

### Phase 3: Witness Learning

**Model:** Linear SVM
- Input: 36-dimensional feature vector (1+2 body expectations)
- Output: Binary classification (distillable / non-distillable)
- Extraction: Hyperplane w·x + b = 0 → Witness W = Σ wₖPₖ

**Why Linear?**
1. Direct correspondence to Hermitian witness operator
2. Interpretable coefficients = measurement weights
3. Baseline for complexity analysis

### Phase 4: Validation

**Metrics:**
| Metric | Target | Rationale |
|--------|--------|-----------|
| Recall (distillable) | >90% | Don't reject good QEC states |
| Precision | >80% | Minimize false distillability claims |
| Measurement cost | <15 settings | Experimentally feasible |

**Ablation Studies:**
1. Accuracy vs. number of 2-body terms
2. Performance across noise models (depolarizing, amplitude damping, dephasing)
3. Comparison: restricted (36D) vs. full (63D) feature space

---

## Expected Outcomes

### If Hypothesis Holds ✓
- **Practical:** Single-scalar witness evaluation replaces expensive SDP
- **Experimental:** Verification aligned with hardware measurement constraints
- **Scientific:** Identifies which QEC state families are amenable to low-complexity verification

### If Hypothesis Fails ✗
- **Scientific insight:** Proves that 3-body correlations are necessary for distillability certification
- **Quantifies gap:** Measures how much accuracy is lost by restricting to 2-body
- **Guides hardware:** Motivates development of efficient 3-body measurement protocols

---

## Alignment with Codebase

| Requirement | Codebase Component | Status |
|-------------|-------------------|--------|
| 3-qubit states | `generate_dataset(n_qubits=3)` | Exists, needs validation |
| Restricted features | `create_sparse_measurement_set(strategy='two_body')` | ✅ Ready |
| Linear SVM | `SVMWitnessLearner` | ✅ Ready |
| Witness extraction | `get_witness_operator() → SparsePauliOp` | ✅ Ready |
| Measurement grouping | `group_commuting_paulis()` | ✅ Ready |
| Distillability oracle | Not implemented | **To be built** |
| DPS hierarchy | Not implemented | **To be built** |

---

## Implementation Milestones

### Milestone 1: Infrastructure (Day 1-2)
- [ ] Validate 3-qubit state generation
- [ ] Implement NPT-based distillability proxy
- [ ] Verify restricted feature extraction (36D)
- [ ] End-to-end pipeline test

### Milestone 2: Dataset Generation (Day 3-4)
- [ ] Generate noisy GHZ/W/cluster state families
- [ ] Implement bound entangled state generators (if needed)
- [ ] Create balanced training set (~5000 states)
- [ ] Validate labels with known analytical cases

### Milestone 3: Witness Learning (Day 5-6)
- [ ] Train SVM on restricted features
- [ ] Extract witness operator
- [ ] Evaluate on held-out test set
- [ ] Compute measurement cost

### Milestone 4: Analysis (Day 7-10)
- [ ] Ablation: restricted vs. full features
- [ ] Noise robustness characterization
- [ ] State-family-specific analysis
- [ ] Failure case identification

### Milestone 5: Enhancement (Optional)
- [ ] Implement DPS hierarchy for better labeling
- [ ] Nonlinear models (MLP) for comparison
- [ ] Sparse witness optimization (L1 regularization)

---

## Success Criteria

**Minimum Viable Result:**
- Linear witness achieving >80% accuracy on 3-qubit distillability
- Witness expressible in <20 Pauli terms
- Measurement cost <15 distinct settings

**Strong Result:**
- >90% recall on distillable states
- Clear separation of QEC-relevant state families
- Identified failure modes with physical interpretation

**Publication-Ready Result:**
- Comparison with full-feature witness (quantified information loss)
- Noise threshold analysis for major state families
- Demonstrated operational advantage over SDP-based testing

---

## Non-Goals

This work does **NOT** aim to:
- Solve the general separability/distillability problem
- Replace SDP methods for arbitrary states
- Achieve perfect classification (the boundary may be inherently nonlinear)
- Extend beyond 3 qubits (future work)

---

## References

- Doherty, Parrilo, Spedalieri (DPS hierarchy for separability)
- Horodecki family (bound entanglement, distillability theory)
- Gühne & Tóth (entanglement witnesses review)
- Framework document: `ML_QML_Quantum_Witness_Framework.md`

---

*This goal document serves as the north star for development. All implementation decisions should be evaluated against these objectives.*
