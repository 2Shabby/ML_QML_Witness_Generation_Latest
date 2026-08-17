"""Integration tests for the end-to-end witness-learning pipelines."""

import numpy as np

from src.feature_extraction.pauli_features import (
    create_sparse_measurement_set,
    extract_features_batch,
    get_pauli_basis,
)
from src.ml_models.svm_witness import SVMWitnessLearner
from src.quantum_states.state_generation import (
    generate_bell_state,
    generate_dataset,
    generate_distillability_dataset,
    generate_separable_state,
)


class TestIntegration:
    """Exercise the complete state-to-feature-to-classifier pipelines."""

    def test_end_to_end_svm_witness(self):
        states, labels = generate_dataset(
            n_qubits=2,
            n_samples=200,
            entangled_fraction=0.5,
            noise_range=(0.0, 0.1),
            seed=42,
        )
        basis = get_pauli_basis(2, include_identity=False)
        features = extract_features_batch(states, basis, verbose=False)
        learner = SVMWitnessLearner(pauli_basis=basis, C=1.0, random_state=42)
        metrics = learner.train(features, labels, test_size=0.2, verbose=False)

        witness = learner.get_witness_operator()
        assert features.shape == (200, len(basis))
        assert metrics["test_accuracy"] > 0.55
        assert witness is not None
        assert len(learner.get_sparse_witness(threshold=0.05)) > 0
        assert learner.get_measurement_cost() > 0

    def test_incomplete_measurements_pipeline(self):
        states, labels = generate_dataset(
            n_qubits=2,
            n_samples=150,
            entangled_fraction=0.5,
            seed=42,
        )
        basis = create_sparse_measurement_set(2, strategy="two_body")
        features = extract_features_batch(states, basis, verbose=False)
        learner = SVMWitnessLearner(pauli_basis=basis, C=1.0, random_state=42)
        metrics = learner.train(features, labels, test_size=0.2, verbose=False)

        assert metrics["test_accuracy"] > 0.52

    def test_witness_on_known_states(self):
        states, labels = generate_dataset(
            n_qubits=2,
            n_samples=100,
            entangled_fraction=0.5,
            seed=42,
        )
        basis = get_pauli_basis(2, include_identity=False)
        features = extract_features_batch(states, basis, verbose=False)
        learner = SVMWitnessLearner(pauli_basis=basis, C=1.0, random_state=42)
        learner.train(features, labels, test_size=0.2, verbose=False)

        bell_features = extract_features_batch(
            [generate_bell_state(i) for i in range(4)], basis, verbose=False
        )
        separable_features = extract_features_batch(
            [generate_separable_state(2, seed=42 + i) for i in range(5)],
            basis,
            verbose=False,
        )

        assert np.sum(learner.predict(bell_features)) >= 1
        assert np.sum(learner.predict(separable_features)) < 5

    def test_3qubit_distillability_pipeline(self):
        states, labels = generate_distillability_dataset(
            n_samples=500,
            noise_range=(0.0, 0.5),
            seed=42,
        )
        basis = create_sparse_measurement_set(3, "two_body")
        features = extract_features_batch(states, basis, verbose=False)
        learner = SVMWitnessLearner(
            pauli_basis=basis,
            C=1.0,
            kernel="linear",
            random_state=42,
        )
        metrics = learner.train(features, labels, test_size=0.2, verbose=False)
        witness = learner.get_witness_operator()

        assert len(states) == 500
        assert all(state.dim == 8 for state in states)
        assert len(basis) == 36
        assert features.shape == (500, 36)
        assert metrics["test_accuracy"] > 0.55
        assert witness is not None
        assert len(witness) <= 36
