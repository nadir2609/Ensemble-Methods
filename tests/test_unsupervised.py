from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from src.unsupervised.dbscan import NOISE, DBSCAN
from src.unsupervised.kmeans import KMeans
from src.unsupervised.pca import PCA


def _three_blobs(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Three well-separated Gaussian blobs in 2-D."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    X_parts, y_parts = [], []
    for k, c in enumerate(centers):
        X_parts.append(c + rng.standard_normal((60, 2)) * 0.5)
        y_parts.append(np.full(60, k))
    return np.vstack(X_parts), np.concatenate(y_parts)



def test_pca_variance_ratio_sums_and_orders() -> None:
    rng = np.random.default_rng(0)
    # Strong variance along axis 0, little along axis 1.
    X = rng.standard_normal((300, 2)) * np.array([10.0, 0.1])
    pca = PCA(n_components=2).fit(X)
    ratio = pca.explained_variance_ratio_
    assert ratio is not None
    assert ratio[0] > ratio[1]           
    assert ratio[0] > 0.98               
    assert ratio.sum() == pytest.approx(1.0, abs=1e-6)


def test_pca_projects_to_requested_dimension() -> None:
    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 5))
    Z = PCA(n_components=2).fit_transform(X)
    assert Z.shape == (100, 2)


def test_pca_recovers_principal_axis() -> None:
    rng = np.random.default_rng(2)
    # Data stretched along the (1,1) direction.
    t = rng.standard_normal(400)
    X = np.column_stack([t, t]) + 0.01 * rng.standard_normal((400, 2))
    pca = PCA(n_components=1).fit(X)
    comp = pca.components_[0]

    cos = abs(comp @ np.array([1.0, 1.0]) / np.sqrt(2))
    assert cos > 0.99



def test_kmeans_recovers_blobs() -> None:
    X, y = _three_blobs()
    km = KMeans(n_clusters=3, random_state=42).fit(X)
    assert km.labels_ is not None
    assert adjusted_rand_score(y, km.labels_) > 0.95


def test_kmeans_inertia_decreases_with_k() -> None:
    X, _ = _three_blobs()
    inertias = [KMeans(n_clusters=k, random_state=42).fit(X).inertia_ for k in (1, 2, 3)]
    assert inertias[0] > inertias[1] > inertias[2]


def test_kmeans_reproducible() -> None:
    X, _ = _three_blobs()
    a = KMeans(n_clusters=3, random_state=5).fit(X).labels_
    b = KMeans(n_clusters=3, random_state=5).fit(X).labels_
    assert a is not None and b is not None
    assert np.array_equal(a, b)


def test_dbscan_separates_blobs() -> None:
    X, y = _three_blobs()
    labels = DBSCAN(eps=1.0, min_samples=5).fit_predict(X)
    # Three dense blobs -> three clusters, high agreement with truth.
    non_noise = labels != NOISE
    assert adjusted_rand_score(y[non_noise], labels[non_noise]) > 0.95
    assert len(set(labels[non_noise])) == 3


def test_dbscan_labels_outliers_as_noise() -> None:
    X, _ = _three_blobs()
    # Add far-away outliers that belong to no dense region.
    outliers = np.array([[50.0, 50.0], [-40.0, -40.0]])
    X_aug = np.vstack([X, outliers])
    labels = DBSCAN(eps=1.0, min_samples=5).fit_predict(X_aug)
    assert labels[-1] == NOISE and labels[-2] == NOISE


def test_dbscan_noise_fraction_reasonable() -> None:
    X, _ = _three_blobs()
    labels = DBSCAN(eps=1.0, min_samples=5).fit_predict(X)
    noise_fraction = np.mean(labels == NOISE)
    assert noise_fraction < 0.1
