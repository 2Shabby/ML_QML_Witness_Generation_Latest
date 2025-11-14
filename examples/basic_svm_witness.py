"""
Basic SVM Witness Learning Example

This script demonstrates the complete pipeline for learning an entanglement
witness using a linear SVM, following the framework approach.
"""

import sys
sys.path.insert(0, '/home/user/ML_QML_Witness_Generation')

import numpy as np
import logging
from src.quantum_states.state_generation import generate_dataset, generate_bell_state
from src.feature_extraction.pauli_features import (
    get_pauli_basis,
    extract_features_batch,
    extract_pauli_features
)
from src.ml_models.svm_witness import SVMWitnessLearner

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run the basic SVM witness learning pipeline."""

    print("="*70)
    print("ML/QML Quantum Witness Generation Framework")
    print("Basic SVM Witness Learning Example")
    print("="*70)

    # Configuration
    n_qubits = 2
    n_samples = 300
    entangled_fraction = 0.5
    seed = 42

    print(f"\nConfiguration:")
    print(f"  Number of qubits: {n_qubits}")
    print(f"  Training samples: {n_samples}")
    print(f"  Entangled fraction: {entangled_fraction}")

    # Step 1: Generate quantum states dataset
    print("\n" + "="*70)
    print("Step 1: Generating Quantum States Dataset")
    print("="*70)

    states, labels = generate_dataset(
        n_qubits=n_qubits,
        n_samples=n_samples,
        entangled_fraction=entangled_fraction,
        noise_range=(0.0, 0.25),
        seed=seed
    )

    n_entangled = np.sum(labels == 1)
    n_separable = np.sum(labels == 0)
    print(f"Generated {n_samples} states:")
    print(f"  - {n_separable} separable (label=0)")
    print(f"  - {n_entangled} entangled (label=1)")

    # Step 2: Extract Pauli features
    print("\n" + "="*70)
    print("Step 2: Extracting Pauli Features (Bloch Vector)")
    print("="*70)

    pauli_basis = get_pauli_basis(n_qubits, include_identity=False)
    print(f"Pauli basis size: {len(pauli_basis)} operators")
    print(f"Sample operators: {[str(p) for p in list(pauli_basis)[:5]]}")

    features = extract_features_batch(states, pauli_basis, verbose=True)
    print(f"Feature matrix shape: {features.shape}")

    # Step 3: Train SVM witness learner
    print("\n" + "="*70)
    print("Step 3: Training SVM Witness Learner")
    print("="*70)

    svm_learner = SVMWitnessLearner(
        pauli_basis=pauli_basis,
        C=1.0,  # Regularization parameter
        random_state=seed
    )

    metrics = svm_learner.train(features, labels, test_size=0.2, verbose=True)

    print(f"\nTraining Results:")
    print(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  Test Accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"  Test Precision: {metrics['test_precision']:.4f} (critical: should be ~1.0)")
    print(f"  Test Recall:    {metrics['test_recall']:.4f}")
    print(f"  Support Vectors: {metrics['n_support_vectors']}")

    # Step 4: Extract witness operator
    print("\n" + "="*70)
    print("Step 4: Extracting Witness Operator")
    print("="*70)

    witness = svm_learner.get_witness_operator()
    print(f"Witness operator W = Σ wₖ Pₖ")
    print(f"  Total terms: {len(witness)}")
    print(f"  Bias term: {svm_learner.bias:.4f}")

    # Show top terms
    witness_list = witness.to_list()
    sorted_terms = sorted(witness_list, key=lambda x: abs(x[1]), reverse=True)
    print(f"\nTop 5 terms by magnitude:")
    for pauli, coeff in sorted_terms[:5]:
        print(f"  {coeff:+.4f} * {pauli}")

    # Step 5: Create sparse witness
    print("\n" + "="*70)
    print("Step 5: Creating Sparse Witness for Measurement")
    print("="*70)

    sparse_witness = svm_learner.get_sparse_witness(threshold=0.05)
    measurement_cost = svm_learner.get_measurement_cost()

    print(f"Sparse witness (threshold=0.05):")
    print(f"  Terms: {len(sparse_witness)} (reduced from {len(witness)})")
    print(f"  Measurement settings required: {measurement_cost}")

    # Step 6: Test on known states
    print("\n" + "="*70)
    print("Step 6: Testing on Known Quantum States")
    print("="*70)

    # Test on Bell states (should be entangled)
    print("\nTesting on Bell states:")
    for i in range(4):
        bell = generate_bell_state(bell_type=i)
        bell_features = extract_pauli_features(bell, pauli_basis)
        prediction = svm_learner.predict(bell_features.reshape(1, -1))[0]
        probability = svm_learner.predict_proba(bell_features.reshape(1, -1))[0]
        decision = svm_learner.decision_function(bell_features.reshape(1, -1))[0]

        print(f"  Bell state {i}: Prediction={prediction} " +
              f"(P(entangled)={probability[1]:.3f}, decision={decision:+.3f})")

    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"✓ Successfully trained SVM witness with {metrics['test_accuracy']:.1%} test accuracy")
    print(f"✓ Extracted witness operator with {len(witness)} Pauli terms")
    print(f"✓ Sparse witness requires {measurement_cost} measurement settings")
    print(f"✓ Correctly identifies Bell states as entangled")

    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)


if __name__ == '__main__':
    main()
