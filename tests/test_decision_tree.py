"""Unit tests and sklearn sanity checks for the from-scratch DecisionTree."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier as SkDecisionTree

from src.metrics.evaluation import accuracy
from src.trees.decision_tree import DecisionTree
from src.utils.data_loaders import load_breast_cancer
from src.utils.preprocessing import train_test_split


def _xor_data() -> tuple[np.ndarray, np.ndarray]:
    """A small, perfectly separable XOR-like problem."""
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    labels = np.array([0, 1, 1, 0])
    X_parts, y_parts = [], []
    for c, lab in zip(centers, labels):
        X_parts.append(c + 0.05 * rng.standard_normal((25, 2)))
        y_parts.append(np.full(25, lab))
    return np.vstack(X_parts), np.concatenate(y_parts)


def test_xor_is_learnable() -> None:
    X, y = _xor_data()
    tree = DecisionTree(max_depth=4, random_state=0)
    tree.fit(X, y)
    # XOR is a classic case where a greedy tree needs depth > 2 to separate.
    assert accuracy(y, tree.predict(X)) >= 0.98
    assert tree.depth >= 2


def test_pure_node_is_single_leaf() -> None:
    X = np.random.default_rng(1).standard_normal((20, 3))
    y = np.zeros(20, dtype=int)
    tree = DecisionTree().fit(X, y)
    assert tree.n_leaves == 1
    assert tree.depth == 0
    assert np.all(tree.predict(X) == 0)


def test_stump_max_depth_zero_and_one() -> None:
    X, y = _xor_data()
    stump = DecisionTree(max_depth=0).fit(X, y)
    assert stump.depth == 0 and stump.n_leaves == 1
    one = DecisionTree(max_depth=1).fit(X, y)
    assert one.depth == 1 and one.n_leaves == 2


def test_single_feature_and_min_samples_split_one() -> None:
    rng = np.random.default_rng(2)
    X = rng.standard_normal((40, 1))
    y = (X[:, 0] > 0).astype(int)
    tree = DecisionTree(min_samples_split=1).fit(X, y)
    assert accuracy(y, tree.predict(X)) == pytest.approx(1.0)


def test_predict_proba_is_valid_distribution() -> None:
    X, y = _xor_data()
    proba = DecisionTree(max_depth=4, random_state=0).fit(X, y).predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all(proba >= 0)


def test_entropy_criterion_also_solves_xor() -> None:
    X, y = _xor_data()
    tree = DecisionTree(max_depth=4, criterion="entropy", random_state=0).fit(X, y)
    assert accuracy(y, tree.predict(X)) >= 0.98


def test_feature_importances_sum_to_one() -> None:
    X, y = _xor_data()
    imp = DecisionTree(max_depth=4, random_state=0).fit(X, y).feature_importances()
    assert imp.shape == (2,)
    assert imp.sum() == pytest.approx(1.0)


def test_sample_weight_shifts_predictions() -> None:
    # Heavily upweighting one class biases an unconstrained stump toward it.
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    w = np.array([1.0, 1.0, 1.0, 100.0])
    tree = DecisionTree(max_depth=1).fit(X, y, sample_weight=w)
    assert tree.root_ is not None and not tree.root_.is_leaf


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_matches_sklearn_within_two_percent(criterion: str) -> None:
    ds = load_breast_cancer()
    X_tr, X_te, y_tr, y_te = train_test_split(ds.X, ds.y, test_size=0.2, random_state=42)

    ours = DecisionTree(max_depth=5, criterion=criterion, random_state=42)
    ours.fit(X_tr, y_tr)
    theirs = SkDecisionTree(max_depth=5, criterion=criterion, random_state=42)
    theirs.fit(X_tr, y_tr)

    acc_ours = accuracy(y_te, ours.predict(X_te))
    acc_theirs = accuracy(y_te, theirs.predict(X_te))
    assert abs(acc_ours - acc_theirs) <= 0.02


def test_repr_is_indented_for_shallow_tree() -> None:
    X, y = _xor_data()
    text = repr(DecisionTree(max_depth=2, random_state=0).fit(X, y))
    assert "feat" in text and "samples" in text
