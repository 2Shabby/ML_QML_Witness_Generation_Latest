"""Tests for normalization controls and shared comparison splits."""

import numpy as np
import pytest

from src.feature_extraction.preprocessing import create_amplitude_encoding_controls
from src.utils import stratified_split_indices


def test_amplitude_encoding_controls_preserve_expected_information():
    features = np.array([[3.0, 4.0], [0.0, 2.0]])

    controls = create_amplitude_encoding_controls(features)

    np.testing.assert_array_equal(controls["raw"], features)
    np.testing.assert_allclose(
        np.linalg.norm(controls["l2_normalized"], axis=1),
        1.0,
    )
    np.testing.assert_allclose(controls["l2_plus_norm"][:, -1], [5.0, 2.0])
    assert controls["l2_plus_norm"].shape == (2, 3)


def test_amplitude_encoding_controls_reject_zero_norm_samples():
    with pytest.raises(ValueError, match="Zero-norm"):
        create_amplitude_encoding_controls(np.zeros((1, 36)))


def test_stratified_split_indices_are_reusable_and_reproducible():
    labels = np.array([0, 1] * 10)

    first = stratified_split_indices(labels, test_size=0.2, random_state=42)
    second = stratified_split_indices(labels, test_size=0.2, random_state=42)

    np.testing.assert_array_equal(first["train"], second["train"])
    np.testing.assert_array_equal(first["test"], second["test"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert sorted(np.concatenate(tuple(first.values()))) == list(range(len(labels)))
    assert labels[first["test"]].tolist().count(1) == 2
