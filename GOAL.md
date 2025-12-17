# CANONICAL: Research Goal

# Learning Restricted Witnesses for Three-Qubit Distillability

**Document Status:** CANONICAL
**Version:** 1.0
**Last Updated:** December 17, 2025

> This document defines the authoritative research objective. All implementation decisions must align with these goals.

---

## Problem Significance and Rationale

Certifying the usefulness of quantum states for quantum error correction (QEC) remains a central practical challenge in near-term and early fault-tolerant quantum devices. In the three-qubit regime—the smallest setting where multipartite entanglement, bound entanglement, and QEC-relevant structure first appear—the operational property of interest is **distillability**, not merely the presence of entanglement. While numerical methods based on semidefinite programming (e.g., the DPS hierarchy) can determine distillability for general mixed states, they require full state knowledge and are computationally incompatible with real-time experimental use. **No closed-form analytical criterion is known.**

At the same time, experimental constraints severely limit accessible measurements: in many architectures, reliable readout is restricted to single-qubit and two-qubit Pauli observables, while higher-weight measurements incur substantial noise and overhead. This restriction fundamentally alters the certification problem, transforming it into a **quantum marginal setting** where relevant global properties may or may not survive projection onto low-order correlators.

This project investigates whether the distillability of three-qubit QEC resource states admits a physically measurable linear witness when only one- and two-body Pauli expectation values are available. **Rather than assuming the existence of such witnesses, we treat this as an open question.** Ground-truth labels are generated using SDP-based criteria, which are **deliberately kept separate** from the restricted measurement space used for learning. Linear machine-learning models are employed not as black-box classifiers, but as tools to extract explicit Hermitian operators whose expectation values can be directly evaluated in experiment.

**The significance of this approach lies in explicitly probing the limits of certifiability under realistic informational constraints.** Where restricted witnesses exist, they offer a practical alternative to tomography and real-time optimization. Where they provably fail, the result identifies fundamental limitations imposed by measurement locality and multipartite entanglement structure. **In both cases, the outcome informs the design of verification protocols for QEC resource states, grounded in operational relevance and experimental feasibility.**

---

## Problem Statement

**Learning Restricted Witnesses for Distillability of Three-Qubit QEC Resource States**

### The Fundamental Gap

| What is Needed | What Exists | The Gap |
|----------------|-------------|---------|
| Fast certification of QEC-useful states | No closed-form distillability criterion | Certification is computationally hard |
| Experimentally feasible measurements | Full tomography requires O(d⁴) measurements | Tomography is infeasible |
| Real-time verification | SDP tests require full state knowledge | SDP is incompatible with real-time use |
| Hardware-compatible observables | Only 1+2 body Paulis are reliable | 3-body measurements are noisy |

### Experimental Reality

In most quantum architectures:
- **Single-qubit Paulis** (X₁, Y₁, Z₁, etc.): Reliable, low noise
- **Two-qubit Paulis** (X₁X₂, Z₁Z₂, etc.): Feasible with moderate overhead
- **Three-qubit Paulis** (X₁Y₂Z₃, etc.): Require 3-qubit entangling gates, high noise, often impractical

This hardware constraint defines the **restricted measurement space** central to this investigation.

---

## Research Question

> **Does there exist a physically measurable, low-weight Hermitian operator W, expressible as a linear combination of one- and two-body Pauli operators:**
>
> $$W = \sum_i a_i P_i^{(1)} + \sum_{i<j} b_{ij} P_i^{(1)} \otimes P_j^{(1)}$$
>
> **whose expectation value ⟨W⟩ reliably distinguishes distillable three-qubit states from non-distillable ones?**

### This is an Open Question

We do **NOT** assume such a witness exists. The investigation probes whether:

1. **The distillability boundary has meaningful structure** in the 36-dimensional space of 1+2 body observables
2. **Global entanglement properties survive projection** onto local/pairwise correlators
3. **Linear separability is sufficient** or if the boundary is inherently nonlinear in this restricted space

### Scope

| Dimension | Choice | Rationale |
|-----------|--------|-----------|
| System size | 3 qubits | Smallest multipartite QEC-relevant setting |
| Classification | Binary (distillable / non-distillable) | Operationally meaningful |
| Feature space | 36D (1+2 body Paulis) | Experimentally accessible |
| Excluded | 27 three-body terms | Hardware constraints |
| Labeling | SDP-based (DPS hierarchy) | Ground truth, deliberately separate from features |
| Model | Linear SVM | Extracts explicit Hermitian operator |

---

## Methodological Framework

### Key Design Principle: Separation of Labeling and Features

```
┌─────────────────────────────────────────────────────────────────┐
│                    LABELING ORACLE (Ground Truth)                │
│         SDP-based: DPS hierarchy, full state knowledge          │
│                   Computationally expensive                      │
│                    NOT measurement-restricted                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓ Labels
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE SPACE (Restricted)                    │
│              36 expectation values: ⟨P⟩ for 1+2 body            │
│                  Experimentally accessible                       │
│                    Measurement-restricted                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓ Learn
┌─────────────────────────────────────────────────────────────────┐
│                    LINEAR WITNESS (Output)                       │
│                 W = Σ wₖPₖ  (explicit operator)                  │
│               Directly evaluable in experiment                   │
│              Single scalar: ⟨W⟩ = Tr(Wρ)                        │
└─────────────────────────────────────────────────────────────────┘
```

**Critical:** The labeling oracle uses information (full density matrix, SDP optimization) that is **not available** to the learned witness. The witness must generalize from the restricted feature space alone.

### Phase 1: Ground Truth Generation

**Distillability Oracle:**

| Method | Description | Use |
|--------|-------------|-----|
| NPT Proxy | ρ^Γ has negative eigenvalues across any bipartition | Fast initial approximation |
| DPS Hierarchy | SDP relaxation of separability at level k | Rigorous ground truth |
| Known Constructions | Analytically characterized state families | Validation benchmarks |

**State Families for Training:**

| Family | Role | Expected Challenge |
|--------|------|-------------------|
| Noisy GHZ | QEC resource baseline | Threshold behavior |
| Noisy W | Robust entanglement | Different noise resilience |
| Noisy cluster | MBQC resources | Graph state structure |
| Random mixed | Boundary sampling | Near-boundary states |
| Bound entangled | Hard negatives | PPT but entangled |
| Werner states | Analytical benchmark | Known thresholds |

### Phase 2: Restricted Feature Extraction

**Feature Space (36 dimensions):**

```
1-body (9 features):
  X₁, Y₁, Z₁, X₂, Y₂, Z₂, X₃, Y₃, Z₃

2-body (27 features):
  X₁X₂, X₁Y₂, X₁Z₂, Y₁X₂, Y₁Y₂, Y₁Z₂, Z₁X₂, Z₁Y₂, Z₁Z₂  (qubits 1-2)
  X₁X₃, X₁Y₃, X₁Z₃, Y₁X₃, Y₁Y₃, Y₁Z₃, Z₁X₃, Z₁Y₃, Z₁Z₃  (qubits 1-3)
  X₂X₃, X₂Y₃, X₂Z₃, Y₂X₃, Y₂Y₃, Y₂Z₃, Z₂X₃, Z₂Y₃, Z₂Z₃  (qubits 2-3)

EXCLUDED 3-body (27 terms):
  X₁X₂X₃, X₁X₂Y₃, ..., Z₁Z₂Z₃
```

### Phase 3: Witness Learning

**Model:** Linear SVM (Support Vector Machine)

**Why Linear?**
1. Hyperplane w·x + b = 0 **directly corresponds** to Hermitian operator W = Σ wₖPₖ
2. Coefficients wₖ are **interpretable** as measurement weights
3. Provides **baseline** for understanding boundary complexity
4. Failure of linear model → evidence for nonlinear structure

**Extraction:**
```
SVM hyperplane: w·x + b = 0
        ↓
Witness operator: W = Σₖ wₖ Pₖ
        ↓
Classification rule: Tr(Wρ) + b < 0 → distillable
```

### Phase 4: Validation and Analysis

**Primary Metrics:**

| Metric | Target | Interpretation |
|--------|--------|----------------|
| Recall (distillable) | >90% | Don't reject good QEC states |
| Precision | >80% | Minimize false positives |
| Accuracy | >85% | Overall performance |
| Witness sparsity | <20 terms | Practical measurement |
| Measurement settings | <15 | Experimental feasibility |

**Ablation Studies:**

1. **Restricted vs. Full:** Compare 36D (1+2 body) vs. 63D (all Paulis)
2. **Feature importance:** Which correlators matter most?
3. **Noise models:** Depolarizing, amplitude damping, dephasing
4. **State families:** Per-family performance analysis
5. **Boundary analysis:** Characterize failure cases

---

## Outcome Interpretation

### If Restricted Witnesses Succeed

**Implications:**
- Distillability boundary has meaningful projection onto 1+2 body space
- Practical certification possible without tomography or SDP
- Single-scalar evaluation replaces expensive optimization
- Hardware-compatible verification protocols enabled

**Deliverables:**
- Explicit witness operator W as `SparsePauliOp`
- Measurement protocol with grouped commuting observables
- Noise thresholds for QEC state families
- Experimental deployment guidelines

### If Restricted Witnesses Fail

**Implications:**
- 3-body correlations carry essential distillability information
- Fundamental limitation of measurement locality identified
- Quantum marginal problem exhibits genuine 3-body structure
- Motivates hardware development for efficient 3-body measurements

**Deliverables:**
- Quantified accuracy gap: restricted vs. full features
- Identification of state families where restriction fails
- Characterization of information lost in projection
- Theoretical analysis of boundary nonlinearity

### Both Outcomes Are Valuable

This is a **diagnostic investigation**, not a guaranteed solution. The scientific contribution exists regardless of which outcome obtains.

---

## Codebase Alignment

| Requirement | Component | Status | Gap |
|-------------|-----------|--------|-----|
| 3-qubit states | `generate_dataset(n_qubits=3)` | Exists | Validate |
| Restricted features | `create_sparse_measurement_set('two_body')` | ✅ Ready | None |
| Linear SVM | `SVMWitnessLearner` | ✅ Ready | None |
| Witness extraction | `get_witness_operator()` | ✅ Ready | None |
| Measurement grouping | `group_commuting_paulis()` | ✅ Ready | None |
| NPT oracle | Not implemented | ❌ Missing | **Build** |
| DPS hierarchy | Not implemented | ❌ Missing | **Build** |
| QEC state generators | Not implemented | ❌ Missing | **Build** |

---

## Implementation Milestones

### Milestone 1: Infrastructure Validation (Day 1-2)
- [ ] Validate 3-qubit state generation pipeline
- [ ] Implement NPT-based distillability proxy (all bipartitions)
- [ ] Verify restricted feature extraction produces 36D vectors
- [ ] End-to-end pipeline test: states → features → SVM → witness

### Milestone 2: Dataset Generation (Day 3-4)
- [ ] Implement noisy GHZ/W/cluster state generators
- [ ] Create training set with SDP labels (~5000 states)
- [ ] Validate labels against known analytical cases
- [ ] Ensure boundary region adequately sampled

### Milestone 3: Witness Learning (Day 5-6)
- [ ] Train linear SVM on restricted features
- [ ] Extract witness operator as `SparsePauliOp`
- [ ] Evaluate on held-out test set
- [ ] Compute measurement cost (grouped settings)

### Milestone 4: Analysis (Day 7-10)
- [ ] Ablation: restricted (36D) vs. full (63D) feature space
- [ ] Per-family analysis: where does restriction succeed/fail?
- [ ] Noise robustness characterization
- [ ] Failure case identification and interpretation

### Milestone 5: Refinement (Optional)
- [ ] Implement full DPS hierarchy for rigorous labeling
- [ ] Nonlinear model (MLP) comparison
- [ ] L1 regularization for sparse witnesses
- [ ] Theoretical analysis of failure modes

---

## Success Criteria

**Minimum Viable Result:**
- Pipeline operational for 3-qubit restricted witness learning
- Linear witness with measurable accuracy (even if modest)
- Clear quantification of restricted vs. full feature gap

**Strong Result:**
- >85% accuracy with restricted features
- Identified state families where restriction works
- Witness with <15 measurement settings

**Publication-Ready Result:**
- Complete ablation study with statistical significance
- Physical interpretation of success/failure modes
- Demonstrated operational advantage or quantified fundamental limitation
- Reproducible experimental protocol

---

## Non-Goals

This work does **NOT** aim to:
- Solve the general distillability problem for arbitrary dimensions
- Prove universal existence of restricted witnesses
- Replace SDP methods where full state access is available
- Achieve perfect classification (boundary may be inherently complex)
- Extend beyond 3 qubits (future work)

---

## References

**Distillability and Bound Entanglement:**
- Horodecki et al., "Quantum entanglement" (Rev. Mod. Phys. 2009)
- Bennett et al., "Mixed-state entanglement and quantum error correction" (PRA 1996)

**SDP Methods:**
- Doherty, Parrilo, Spedalieri, "Complete family of separability criteria" (PRA 2004)

**Entanglement Witnesses:**
- Gühne & Tóth, "Entanglement detection" (Physics Reports 2009)

**ML for Quantum States:**
- Framework document: `ML_QML_Quantum_Witness_Framework.md`

---

*This document is CANONICAL. All implementation decisions should be evaluated against these objectives. Updates require explicit versioning.*
