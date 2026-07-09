from __future__ import annotations

import numpy as np


class PCA:
    """Linear dimensionality reduction via eigen-decomposition of the covariance."""

    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCA":
        """Compute the principal components from the data covariance matrix."""
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Covariance matrix (features x features); rowvar=False treats columns
        cov = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # eigh returns ascending eigenvalues; reverse to get descending order.
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        total_variance = eigenvalues.sum()
        self.components_ = eigenvectors[:, : self.n_components].T
        self.explained_variance_ = eigenvalues[: self.n_components]
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total_variance
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project (centred) data onto the retained principal components."""
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("PCA must be fit before transform.")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
