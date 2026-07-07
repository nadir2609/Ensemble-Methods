from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Absolute path to the repository's ``data`` directory, resolved relative to
# this file so nothing depends on the current working directory.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "data"))


@dataclass
class Dataset:
    """Container bundling a design matrix, labels and metadata."""

    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    class_names: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_classes(self) -> int:
        return int(len(np.unique(self.y)))

    @property
    def is_binary(self) -> bool:
        return self.n_classes == 2

    def class_balance(self) -> dict[int, float]:
        """Return the fraction of samples belonging to each class."""
        classes, counts = np.unique(self.y, return_counts=True)
        total = counts.sum()
        return {int(c): float(n / total) for c, n in zip(classes, counts)}


def _one_hot(frame: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, list[str]]:
    """One-hot encode ``columns`` of ``frame`` into a 0/1 float matrix."""
    dummies = pd.get_dummies(frame[columns], columns=columns, dtype=np.float64)
    return dummies.to_numpy(dtype=np.float64), list(dummies.columns)


def load_breast_cancer(data_dir: str = DATA_DIR) -> Dataset:
    """Load the Breast Cancer Wisconsin (Diagnostic) dataset.

    Columns: id, diagnosis (M/B), then 30 continuous features.
    ``M`` (malignant) is encoded as class 1, ``B`` (benign) as class 0.
    """
    path = os.path.join(data_dir, "wdbc.data")
    frame = pd.read_csv(path, header=None)
    feature_names = [f"feat_{i}" for i in range(frame.shape[1] - 2)]
    X = frame.iloc[:, 2:].to_numpy(dtype=np.float64)
    y = (frame.iloc[:, 1].to_numpy() == "M").astype(np.int64)
    return Dataset("breast_cancer", X, y, feature_names, class_names=["benign", "malignant"])


def load_adult(
    data_dir: str = DATA_DIR,
    subsample: int | None = 8000,
    random_state: int | None = 42,
) -> Dataset:
    """Load the Adult Income dataset (train split ``adult.data``).

    Categorical columns are one-hot encoded; rows containing the missing
    marker ``?`` are dropped. The label ``>50K`` is class 1. When
    ``subsample`` is given, a class-stratified subsample of that size is
    returned to keep the from-scratch models tractable.
    """
    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income",
    ]
    categorical = [
        "workclass", "education", "marital_status", "occupation",
        "relationship", "race", "sex", "native_country",
    ]
    numeric = [
        "age", "fnlwgt", "education_num", "capital_gain",
        "capital_loss", "hours_per_week",
    ]

    path = os.path.join(data_dir, "adult.data")
    frame = pd.read_csv(path, header=None, names=columns, skipinitialspace=True,
                        na_values="?")
    frame = frame.dropna(axis=0).reset_index(drop=True)

    y = (frame["income"].str.replace(".", "", regex=False) == ">50K").astype(np.int64).to_numpy()
    X_num = frame[numeric].to_numpy(dtype=np.float64)
    X_cat, cat_names = _one_hot(frame, categorical)
    X = np.hstack([X_num, X_cat])
    feature_names = list(numeric) + cat_names

    if subsample is not None and subsample < X.shape[0]:
        idx = _stratified_subsample(y, subsample, random_state)
        X, y = X[idx], y[idx]

    return Dataset("adult", X, y, feature_names, class_names=["<=50K", ">50K"])


def load_covertype(
    data_dir: str = DATA_DIR,
    subsample: int = 10000,
    random_state: int | None = 42,
) -> Dataset:
    """Load a class-stratified subset of the Covertype dataset.

    The full file has 581,012 rows across 7 forest cover types; the smallest
    class (Cottonwood/Willow) is ~0.47% of the data. A stratified subsample
    preserves that proportion, giving the required severe class imbalance
    while keeping the from-scratch models tractable. The 54 features are
    already numeric (10 continuous + 44 binary indicators). Labels are
    remapped from ``1..7`` to ``0..6``.
    """
    path = os.path.join(data_dir, "covtype.data")
    frame = pd.read_csv(path, header=None)
    X = frame.iloc[:, :-1].to_numpy(dtype=np.float64)
    y = frame.iloc[:, -1].to_numpy(dtype=np.int64) - 1  # remap 1..7 -> 0..6
    feature_names = [f"feat_{i}" for i in range(X.shape[1])]

    if subsample < X.shape[0]:
        idx = _stratified_subsample(y, subsample, random_state)
        X, y = X[idx], y[idx]

    class_names = [
        "Spruce/Fir", "Lodgepole", "Ponderosa", "Cottonwood/Willow",
        "Aspen", "Douglas-fir", "Krummholz",
    ]
    return Dataset("covertype", X, y, feature_names, class_names=class_names)


def _stratified_subsample(
    y: np.ndarray, size: int, random_state: int | None
) -> np.ndarray:
    """Return indices of a class-proportional subsample of ``size`` rows.

    Each class receives ``round(size * fraction)`` samples but at least one,
    so rare classes are never dropped entirely.
    """
    rng = np.random.default_rng(random_state)
    n = y.shape[0]
    classes, counts = np.unique(y, return_counts=True)
    selected: list[np.ndarray] = []
    for cls, count in zip(classes, counts):
        cls_idx = np.where(y == cls)[0]
        take = max(1, int(round(size * count / n)))
        take = min(take, cls_idx.shape[0])
        chosen = rng.choice(cls_idx, size=take, replace=False)
        selected.append(chosen)
    idx = np.concatenate(selected)
    rng.shuffle(idx)
    return idx


# Registry so experiments can iterate over datasets uniformly.
LOADERS = {
    "breast_cancer": load_breast_cancer,
    "adult": load_adult,
    "covertype": load_covertype,
}


def load_all() -> list[Dataset]:
    """Load every project dataset with default settings."""
    return [loader() for loader in LOADERS.values()]
