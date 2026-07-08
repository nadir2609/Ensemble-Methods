from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import AdaBoostClassifier as SkAdaBoost

from src.boosting.adaboost import AdaBoostClassifier, DecisionStump
from src.metrics.evaluation import accuracy
from src.trees.decision_tree import DecisionTree
from src.utils.data_loaders import load_breast_cancer
from src.utils.preprocessing import train_test_split


def _two_gaussians(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X0 = rng.normal(-1.0, 1.0, size=(n, 2))
    X1 = rng.normal(1.0, 1.0, size=(n, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * n + [1] * n)
    return X, y


def test_decision_stump_is_depth_one() -> None:
    X, y = _two_gaussians()
    stump = DecisionStump().fit(X, y)
    assert isinstance(stump, DecisionTree)
    assert stump.depth <= 1


def test_alpha_matches_error_formula() -> None:
    # With learning_rate=1 and K=2, alpha_m must equal ln((1-err)/err).
    X, y = _two_gaussians()
    model = AdaBoostClassifier(n_estimators=20, random_state=42).fit(X, y)
    errs = model.estimator_errors
    alphas = model.estimator_weights
    expected = np.log((1.0 - errs) / errs)  # ln(K-1)=ln(1)=0 for binary
    assert np.allclose(alphas, expected, atol=1e-9)


def test_all_errors_below_random_guess() -> None:
    X, y = _two_gaussians()
    model = AdaBoostClassifier(n_estimators=30, random_state=42).fit(X, y)
    assert np.all(model.estimator_errors < 0.5)
    assert np.all(model.estimator_weights > 0)


def test_boosting_reduces_training_error() -> None:
    X, y = _two_gaussians(n=150, seed=1)
    model = AdaBoostClassifier(n_estimators=50, random_state=42).fit(X, y)
    staged = list(model.staged_predict(X))
    first_acc = accuracy(y, staged[0])
    last_acc = accuracy(y, staged[-1])
    assert last_acc >= first_acc  # ensemble should not get worse on train


def test_staged_predict_matches_final_predict() -> None:
    X, y = _two_gaussians()
    model = AdaBoostClassifier(n_estimators=25, random_state=42).fit(X, y)
    staged = list(model.staged_predict(X))
    assert len(staged) == len(model.estimators_)
    assert np.array_equal(staged[-1], model.predict(X))


def test_predict_proba_is_valid_distribution() -> None:
    X, y = _two_gaussians()
    proba = AdaBoostClassifier(n_estimators=15, random_state=42).fit(X, y).predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all(proba >= 0)


def test_reproducible_with_seed() -> None:
    X, y = _two_gaussians()
    a = AdaBoostClassifier(n_estimators=20, random_state=7).fit(X, y).predict(X)
    b = AdaBoostClassifier(n_estimators=20, random_state=7).fit(X, y).predict(X)
    assert np.array_equal(a, b)


def test_competitive_with_sklearn_adaboost() -> None:
    ds = load_breast_cancer()
    X_tr, X_te, y_tr, y_te = train_test_split(ds.X, ds.y, test_size=0.2, random_state=42)

    ours = AdaBoostClassifier(n_estimators=50, random_state=42).fit(X_tr, y_tr)
    theirs = SkAdaBoost(n_estimators=50, random_state=42).fit(X_tr, y_tr)

    acc_ours = accuracy(y_te, ours.predict(X_te))
    acc_theirs = accuracy(y_te, theirs.predict(X_te))
    # Sanity baseline: our ensemble should be in the same ballpark.
    assert acc_ours >= acc_theirs - 0.05
    assert acc_ours >= 0.90
