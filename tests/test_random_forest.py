"""Unit tests and sklearn sanity checks for the from-scratch Random Forest."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier as SkRandomForest

from src.bagging.random_forest import RandomForestClassifier
from src.metrics.evaluation import accuracy
from src.utils.data_loaders import load_breast_cancer
from src.utils.preprocessing import train_test_split


def _split_breast_cancer() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ds = load_breast_cancer()
    return train_test_split(ds.X, ds.y, test_size=0.2, random_state=42)


def test_forest_beats_single_tree_variance() -> None:
    X_tr, X_te, y_tr, y_te = _split_breast_cancer()
    forest = RandomForestClassifier(n_estimators=25, random_state=42).fit(X_tr, y_tr)
    assert accuracy(y_te, forest.predict(X_te)) >= 0.93


def test_predict_proba_valid_distribution() -> None:
    X_tr, X_te, y_tr, _ = _split_breast_cancer()
    proba = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_tr, y_tr).predict_proba(X_te)
    assert proba.shape == (X_te.shape[0], 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all(proba >= 0)


def test_oob_score_is_reasonable() -> None:
    X_tr, _, y_tr, _ = _split_breast_cancer()
    forest = RandomForestClassifier(
        n_estimators=50, oob_score=True, random_state=42
    ).fit(X_tr, y_tr)
    assert 0.85 <= forest.oob_score_ <= 1.0


def test_oob_score_requires_flag() -> None:
    X_tr, _, y_tr, _ = _split_breast_cancer()
    forest = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_tr, y_tr)
    with pytest.raises(AttributeError):
        _ = forest.oob_score_


def test_reproducible_with_seed() -> None:
    X_tr, X_te, y_tr, _ = _split_breast_cancer()
    a = RandomForestClassifier(n_estimators=20, random_state=1).fit(X_tr, y_tr).predict(X_te)
    b = RandomForestClassifier(n_estimators=20, random_state=1).fit(X_tr, y_tr).predict(X_te)
    assert np.array_equal(a, b)


def test_parallel_matches_sequential() -> None:
    X_tr, X_te, y_tr, _ = _split_breast_cancer()
    seq = RandomForestClassifier(n_estimators=16, random_state=3, n_jobs=1).fit(X_tr, y_tr).predict(X_te)
    par = RandomForestClassifier(n_estimators=16, random_state=3, n_jobs=2).fit(X_tr, y_tr).predict(X_te)
    assert np.array_equal(seq, par)


def test_feature_importances_shape_and_normalisation() -> None:
    X_tr, _, y_tr, _ = _split_breast_cancer()
    forest = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_tr, y_tr)
    imp = forest.feature_importances_
    assert imp.shape == (X_tr.shape[1],)
    assert imp.sum() == pytest.approx(1.0, abs=1e-6)


def test_max_features_variants() -> None:
    X_tr, X_te, y_tr, y_te = _split_breast_cancer()
    for mf in ("sqrt", "log2", 5, None):
        forest = RandomForestClassifier(
            n_estimators=10, max_features=mf, random_state=42
        ).fit(X_tr, y_tr)
        assert accuracy(y_te, forest.predict(X_te)) >= 0.85


def test_competitive_with_sklearn_forest() -> None:
    X_tr, X_te, y_tr, y_te = _split_breast_cancer()
    ours = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
    theirs = SkRandomForest(n_estimators=100, random_state=42).fit(X_tr, y_tr)
    acc_ours = accuracy(y_te, ours.predict(X_te))
    acc_theirs = accuracy(y_te, theirs.predict(X_te))
    assert abs(acc_ours - acc_theirs) <= 0.03
