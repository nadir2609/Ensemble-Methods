from __future__ import annotations

import numpy as np


NOISE = -1

_UNCLASSIFIED = -2


class DBSCAN:
    """Density-based spatial clustering of applications with noise."""

    def __init__(self, eps: float, min_samples: int) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.labels_: np.ndarray | None = None

    def _region_query(self, X: np.ndarray, idx: int) -> np.ndarray:
        """Indices of all points within ``eps`` of point ``idx``"""
        dist = np.sqrt(np.sum((X - X[idx]) ** 2, axis=1))
        return np.where(dist <= self.eps)[0]

    def fit(self, X: np.ndarray) -> "DBSCAN":
        """Assign a cluster id to every point (``-1`` for noise)."""
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        labels = np.full(n, _UNCLASSIFIED, dtype=np.int64)
        cluster_id = 0

        for i in range(n):
            if labels[i] != _UNCLASSIFIED:
                continue
            neighbours = self._region_query(X, i)
            if neighbours.shape[0] < self.min_samples:
                labels[i] = NOISE  # provisional; may join a cluster as a border
                continue
            self._expand_cluster(X, labels, i, neighbours, cluster_id)
            cluster_id += 1

        self.labels_ = labels
        return self

    def _expand_cluster(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        seed_idx: int,
        neighbours: np.ndarray,
        cluster_id: int,
    ) -> None:
        """Grow a cluster from a core point using a breadth-first frontier."""
        labels[seed_idx] = cluster_id
        queue = list(neighbours)
        while queue:
            j = queue.pop()
            if labels[j] == NOISE:
                labels[j] = cluster_id  # border point of this cluster
            if labels[j] != _UNCLASSIFIED:
                continue
            labels[j] = cluster_id
            j_neighbours = self._region_query(X, j)
            if j_neighbours.shape[0] >= self.min_samples:
                queue.extend(j_neighbours.tolist())

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Convenience: fit and return the cluster labels."""
        self.fit(X)
        assert self.labels_ is not None
        return self.labels_
