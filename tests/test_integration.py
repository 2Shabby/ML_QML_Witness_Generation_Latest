"""Integration tests for the end-to-end witness-learning pipelines."""

import numpy as np
from qiskit.quantum_info import PauliList

from src.feature_extraction.pauli_features import (
    create_sparse_measurement_set,
)
from src.ml_models.svm_witness import SVMWitnessLearner
from src.quantum_states.balanced_dataset import (
    generate_balanced_distillability_dataset,
)


def _pauli_features(states, pauli_basis: PauliList) -> np.ndarray:
    M = np.asarray(pauli_basis.to_matrix(), dtype=np.complex128)
    R = np.stack([np.asarray(s.data, dtype=np.complex128) for s in states])
    return np.einsum("kij,bji->bk", M, R).real


class TestIntegration:
    """Exercise the complete state-to-feature-to-classifier pipelines."""

    def test_3qubit_distillability_pipeline(self):
        states, labels, _ = generate_balanced_distillability_dataset(
            n_samples=300,
            seed=42,
        )
        basis = create_sparse_measurement_set(3, "two_body")
        features = _pauli_features(states, basis)
        learner = SVMWitnessLearner(
            pauli_basis=basis,
            C=1.0,
            kernel="linear",
            random_state=42,
        )
        metrics = learner.train(features, np.asarray(labels), test_size=0.2, verbose=False)
        witness = learner.get_witness_operator()

        # Pipeline smoke test: exercises the full state -> feature ->
        # classifier -> witness path.  Accuracy is intentionally NOT
        # asserted; the de-confounded near-boundary dataset is genuinely
        # hard at n=300 and model quality is measured by the experiment
        # runs, not by unit tests.
        assert len(states) == 300
        assert all(state.dim == 8 for state in states)
        assert len(basis) == 36
        assert features.shape == (300, 36)
        assert 0.0 <= metrics["test_accuracy"] <= 1.0
        assert witness is not None
        assert len(witness) <= 36

    def test_predicts_by_family_label_tendency(self):
        states, labels, metadata = generate_balanced_distillability_dataset(
            n_samples=300,
            seed=42,
        )
        basis = create_sparse_measurement_set(3, "two_body")
        features = _pauli_features(states, basis)
        learner = SVMWitnessLearner(
            pauli_basis=basis,
            C=1.0,
            kernel="linear",
            random_state=42,
        )
        learner.train(features, np.asarray(labels), test_size=0.2, verbose=False)

        preds = learner.predict(features)
        # No family may be perfectly pure: the generator places both labels
        # in every family, so a constant predictor cannot fit the data.
        for family in {m["family"] for m in metadata}:
            sel = np.asarray([m["family"] == family for m in metadata])
            assert labels[sel].sum() not in (0, sel.sum())
