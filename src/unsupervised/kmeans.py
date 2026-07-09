from __future__ import annotations

import numpy as np


class KMeans:
    """Lloyd's algorithm with k-means++ initialisation."""
    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float = np.inf
        self.n_iter_: int = 0

    def _init_centroids(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """k-means++ seeding: spread initial centroids by distance weighting."""
        n = X.shape[0]
        centroids = [X[rng.integers(n)]]
        for _ in range(1, self.n_clusters):
            dist_sq = np.min(
                [np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0
            )
            total = dist_sq.sum()
            if total == 0:  # all remaining points coincide with a centroid
                centroids.append(X[rng.integers(n)])
                continue
            probs = dist_sq / total
            centroids.append(X[rng.choice(n, p=probs)])
        return np.array(centroids, dtype=np.float64)

    @staticmethod
    def _distances(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Squared Euclidean distance from each point to each centroid."""
        # (n, 1, d) - (1, k, d) -> (n, k, d), summed over d.
        diff = X[:, None, :] - centroids[None, :, :]
        return np.sum(diff**2, axis=2)

    def fit(self, X: np.ndarray) -> "KMeans":
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        centroids = self._init_centroids(X, rng)

        labels = np.zeros(X.shape[0], dtype=np.int64)
        for iteration in range(self.max_iter):
            dist_sq = self._distances(X, centroids)
            labels = np.argmin(dist_sq, axis=1)

            new_centroids = np.empty_like(centroids)
            for k in range(self.n_clusters):
                members = X[labels == k]
                if members.shape[0] == 0:
                    # Re-seed an empty cluster at the worst-fit point.
                    new_centroids[k] = X[np.argmax(np.min(dist_sq, axis=1))]
                else:
                    new_centroids[k] = members.mean(axis=0)

            shift = np.sqrt(np.sum((new_centroids - centroids) ** 2))
            centroids = new_centroids
            self.n_iter_ = iteration + 1
            if shift <= self.tol:
                break

        dist_sq = self._distances(X, centroids)
        labels = np.argmin(dist_sq, axis=1)
        self.centroids_ = centroids
        self.labels_ = labels
        self.inertia_ = float(np.sum(np.min(dist_sq, axis=1)))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new points to the nearest fitted centroid."""
        if self.centroids_ is None:
            raise RuntimeError("KMeans must be fit before predict.")
        X = np.asarray(X, dtype=np.float64)
        return np.argmin(self._distances(X, self.centroids_), axis=1)
