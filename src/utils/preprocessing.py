from __future__ import annotations

import numpy as np


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int | None = None,
    stratify: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into train/test partitions.

    When ``stratify`` is true the class proportions are preserved in both
    partitions, which matters for the imbalanced datasets.
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]

    if stratify:
        test_idx_parts: list[np.ndarray] = []
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            rng.shuffle(cls_idx)
            n_test = int(round(test_size * cls_idx.shape[0]))
            test_idx_parts.append(cls_idx[:n_test])
        test_idx = np.concatenate(test_idx_parts)
    else:
        perm = rng.permutation(n)
        n_test = int(round(test_size * n))
        test_idx = perm[:n_test]

    mask = np.zeros(n, dtype=bool)
    mask[test_idx] = True
    return X[~mask], X[mask], y[~mask], y[mask]


class StandardScaler:
    """Standardize features to zero mean and unit variance.

    The scaler is fit on training data only and then applied to any split,
    matching the leakage-free protocol required by the brief.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        # Guard against zero-variance columns (e.g. one-hot dummies that are
        # constant within a subsample) to avoid division by zero.
        std[std == 0.0] = 1.0
        self.scale_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fit before transform.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def flip_labels(
    y: np.ndarray, eta: float, random_state: int | None = None
) -> np.ndarray:
    """Return a copy of ``y`` with a fraction ``eta`` of labels randomly flipped.

    Each selected label is reassigned uniformly at random to one of the
    *other* classes, so the operation always changes the label.
    """
    rng = np.random.default_rng(random_state)
    y_noisy = y.copy()
    classes = np.unique(y)
    n_flip = int(round(eta * y.shape[0]))
    if n_flip == 0:
        return y_noisy
    flip_idx = rng.choice(y.shape[0], size=n_flip, replace=False)
    for i in flip_idx:
        others = classes[classes != y[i]]
        y_noisy[i] = rng.choice(others)
    return y_noisy


def bootstrap_sample(
    n: int, random_state: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a bootstrap sample of ``n`` indices with replacement.

    Returns ``(in_bag, oob)`` where ``oob`` holds the out-of-bag indices not
    selected by the bootstrap draw.
    """
    rng = np.random.default_rng(random_state)
    in_bag = rng.integers(0, n, size=n)
    oob = np.setdiff1d(np.arange(n), np.unique(in_bag), assume_unique=True)
    return in_bag, oob


def random_oversample(
    X: np.ndarray, y: np.ndarray, random_state: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Balance classes by random oversampling of minority classes.

    Every class is resampled with replacement up to the size of the largest
    class. This is the imbalance treatment documented in the report for the
    severely imbalanced Covertype problem.
    """
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    target = counts.max()
    idx_parts: list[np.ndarray] = []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        extra = rng.choice(cls_idx, size=target - cls_idx.shape[0], replace=True)
        idx_parts.append(np.concatenate([cls_idx, extra]))
    idx = np.concatenate(idx_parts)
    rng.shuffle(idx)
    return X[idx], y[idx]


def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """Balanced class weights ``n_samples / (n_classes * count_c)``.

    These can be turned into per-sample weights and passed to the tree /
    AdaBoost ``sample_weight`` interface as an alternative imbalance
    treatment.
    """
    classes, counts = np.unique(y, return_counts=True)
    n = y.shape[0]
    k = classes.shape[0]
    return {int(c): float(n / (k * cnt)) for c, cnt in zip(classes, counts)}


def sample_weights_from_class_weights(
    y: np.ndarray, class_weights: dict[int, float]
) -> np.ndarray:
    """Map a class-weight dict to a per-sample weight vector."""
    return np.array([class_weights[int(label)] for label in y], dtype=np.float64)
