"""Tests for the Experiment 7 helpers (knee detection and distance geometry)."""

from __future__ import annotations

import numpy as np

from experiments.unsupervised_analysis import (
    _k_distances, _knee_index, _pairwise_distances,
)


def test_knee_index_finds_corner_of_elbow_curve() -> None:
    # Steep drop for the first three points, flat afterwards: the corner sits
    # where the two regimes meet (index 2).
    values = np.array([100.0, 50.0, 20.0, 19.0, 18.0, 17.0, 16.0])
    assert _knee_index(values) == 2


def test_knee_index_is_scale_invariant() -> None:
    values = np.array([100.0, 50.0, 20.0, 19.0, 18.0, 17.0, 16.0])
    assert _knee_index(values) == _knee_index(values * 1000.0)


def test_knee_index_handles_flat_and_tiny_curves() -> None:
    assert _knee_index(np.ones(6)) == 0
    assert _knee_index(np.array([1.0, 2.0])) == 0


def test_pairwise_distances_matches_direct_computation() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((25, 4))
    expected = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
    # The squared-norm identity trades a little precision for the memory saving.
    np.testing.assert_allclose(_pairwise_distances(X), expected, atol=1e-6)


def test_pairwise_distances_diagonal_is_zero() -> None:
    rng = np.random.default_rng(1)
    distances = _pairwise_distances(rng.standard_normal((10, 3)))
    np.testing.assert_allclose(np.diag(distances), 0.0, atol=1e-8)


def test_k_distances_on_evenly_spaced_line() -> None:
    # Points at x = 0, 1, ..., 9: every point's 1st neighbour is 1 away, so the
    # sorted 1-distance curve is constant at 1.0.
    X = np.arange(10, dtype=np.float64).reshape(-1, 1)
    np.testing.assert_allclose(_k_distances(X, 1), np.ones(10), atol=1e-8)
    # The 2nd neighbour is 2 away except for the two endpoints' interior side.
    assert _k_distances(X, 2)[-1] == 2.0


def test_k_distances_is_sorted_ascending() -> None:
    rng = np.random.default_rng(2)
    curve = _k_distances(rng.standard_normal((40, 3)), 5)
    assert np.all(np.diff(curve) >= 0.0)
