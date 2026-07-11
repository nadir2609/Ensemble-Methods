"""Shared helpers for the experiment scripts."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt  
import numpy as np  

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loaders import Dataset, load_all 
from src.utils.preprocessing import StandardScaler, train_test_split  

# Global seed fixed for reproducibility across every experiment.
SEED = 42

# Upper bound on training rows for the heavy scaling sweeps. The from-scratch
# trees are exact but slower than sklearn's C implementation, so the larger
# datasets are sub-sampled (stratified) to this size for the repeated-fit
# experiments. Baselines and cross-validation use the full data.
SCALING_MAX_TRAIN = 2000

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(_ROOT, "figures")


def ensure_figures_dir() -> str:
    """Create the figures directory if needed and return its path."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    return FIGURES_DIR


@dataclass
class Split:
    """A ready-to-use, standardized train/test split of one dataset."""

    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    is_binary: bool
    n_classes: int

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]


def make_split(ds: Dataset, cap: int | None = None, seed: int = SEED) -> Split:
    """Standardized 80/20 split, scaler fit on train, optional train cap."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        ds.X, ds.y, test_size=0.2, random_state=seed, stratify=True
    )
    if cap is not None and X_tr.shape[0] > cap:
        X_tr, y_tr = _stratified_cap(X_tr, y_tr, cap, seed)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    return Split(ds.name, X_tr, X_te, y_tr, y_te, ds.is_binary, ds.n_classes)


def _stratified_cap(
    X: np.ndarray, y: np.ndarray, cap: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a class-proportional subset of at most ``cap`` rows."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    classes, counts = np.unique(y, return_counts=True)
    parts: list[np.ndarray] = []
    for cls, count in zip(classes, counts):
        idx = np.where(y == cls)[0]
        take = max(1, int(round(cap * count / n)))
        take = min(take, idx.shape[0])
        parts.append(rng.choice(idx, size=take, replace=False))
    sel = np.concatenate(parts)
    rng.shuffle(sel)
    return X[sel], y[sel]


def get_datasets() -> list[Dataset]:
    """Load all project datasets (cached call to the loaders)."""
    return load_all()


def savefig(fig: "plt.Figure", filename: str) -> str:
    """Save a figure into the figures directory and close it."""
    ensure_figures_dir()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {os.path.relpath(path, _ROOT)}")
    return path


def print_header(title: str) -> None:
    """Print a consistent section header to stdout."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
