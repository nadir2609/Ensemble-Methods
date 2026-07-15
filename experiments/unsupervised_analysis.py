"""Experiment 7 - Unsupervised analysis."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

from experiments.common import (
    SEED, _stratified_cap, get_datasets, print_header, savefig,
)
from src.unsupervised.dbscan import NOISE, DBSCAN
from src.unsupervised.kmeans import KMeans
from src.unsupervised.pca import PCA
from src.utils.data_loaders import Dataset
from src.utils.preprocessing import StandardScaler

UNSUPERVISED_MAX_SAMPLES = 2000

# Fraction of variance the retained principal components must capture.
VARIANCE_TARGET = 0.90

# Elbow sweep: k = 1..10, each with several restarts keeping the lowest inertia.
KMEANS_K_VALUES = tuple(range(1, 11))
KMEANS_RESTARTS = 10

# Core-point threshold. sklearn's default; close to the MinPts = 4 recommended
# for low-dimensional data by Ester et al. (1996). The k-distance plot uses the
# same k, as the paper's heuristic requires.
DBSCAN_MIN_SAMPLES = 5

# eps candidates, expressed as multiples of the k-distance knee, so the sweep
# is anchored to the data's own density scale rather than absolute distances.
EPS_MULTIPLIERS = (0.6, 0.8, 1.0, 1.2, 1.5, 2.0)

NOISE_COLOR = "#bbbbbb"

@dataclass
class DBSCANResult:
    """Outcome of one DBSCAN fit at a given eps."""

    eps: float
    labels: np.ndarray
    ari: float
    n_clusters: int
    noise_fraction: float


@dataclass
class UnsupervisedResult:
    """Everything Experiment 7 reports for a single dataset."""

    name: str
    n_components_90: int
    cumulative_variance: np.ndarray
    elbow_k: int
    inertias: np.ndarray
    kmeans_aris: dict[int, float]
    best_k: int
    best_kmeans_ari: float
    kmeans_labels: np.ndarray
    eps_knee: float
    dbscan: DBSCANResult


def _knee_index(values: np.ndarray) -> int:
    """Index of the curve point furthest from the chord joining its endpoints."""
    y = np.asarray(values, dtype=np.float64)
    n = y.shape[0]
    if n < 3:
        return 0
    x = np.linspace(0.0, 1.0, n)
    span = y.max() - y.min()
    y = (y - y.min()) / span if span > 0 else np.zeros_like(y)

    dx, dy = x[-1] - x[0], y[-1] - y[0]
    norm = float(np.hypot(dx, dy))
    if norm == 0.0:
        return 0
    distance = np.abs(dy * (x - x[0]) - dx * (y - y[0])) / norm
    return int(np.argmax(distance))


def _pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix via the."""
    sq_norms = np.sum(X**2, axis=1)
    sq_dist = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (X @ X.T)
    distances = np.sqrt(np.maximum(sq_dist, 0.0))
    np.fill_diagonal(distances, 0.0)
    return distances


def _k_distances(X: np.ndarray, k: int) -> np.ndarray:
    """Ascending-sorted distances from each point to its ``k``-th neighbour."""
    distances = np.sort(_pairwise_distances(X), axis=1)
    return np.sort(distances[:, k])


def _prepare(ds: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Stratified-capped, standardized copy of a dataset (no train/test split)."""
    X, y = ds.X, ds.y
    if X.shape[0] > UNSUPERVISED_MAX_SAMPLES:
        X, y = _stratified_cap(X, y, UNSUPERVISED_MAX_SAMPLES, SEED)
    return StandardScaler().fit_transform(X), y


def run_pca(split_x: np.ndarray, name: str) -> tuple[np.ndarray, int, np.ndarray]:
    """Fit PCA, plot the scree curve and return the retained projection."""
    pca = PCA(n_components=split_x.shape[1]).fit(split_x)
    assert pca.explained_variance_ratio_ is not None
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_90 = int(np.searchsorted(cumulative, VARIANCE_TARGET) + 1)
    n_90 = min(n_90, split_x.shape[1])

    projection = pca.transform(split_x)[:, :n_90]
    _plot_scree(name, cumulative, n_90)
    return projection, n_90, cumulative


def _plot_scree(name: str, cumulative: np.ndarray, n_90: int) -> None:
    """Cumulative explained variance against the number of components."""
    xs = np.arange(1, cumulative.shape[0] + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, cumulative, marker="o", markersize=3, color="#1f77b4")
    ax.axhline(VARIANCE_TARGET, ls="--", color="#d62728", lw=1,
               label=f"{VARIANCE_TARGET:.0%} variance")
    ax.axvline(n_90, ls=":", color="#7f7f7f", lw=1,
               label=f"{n_90} components")
    ax.set_xlabel("number of principal components")
    ax.set_ylabel("cumulative explained variance")
    ax.set_title(f"PCA scree - {name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, f"exp7_scree_{name}.png")


def _best_kmeans(X: np.ndarray, k: int) -> KMeans:
    """Lowest-inertia K-Means over ``KMEANS_RESTARTS`` seeded restarts."""
    best: KMeans | None = None
    for restart in range(KMEANS_RESTARTS):
        model = KMeans(n_clusters=k, random_state=SEED + restart).fit(X)
        if best is None or model.inertia_ < best.inertia_:
            best = model
    assert best is not None
    return best


def run_kmeans(
    X: np.ndarray, y: np.ndarray, name: str
) -> tuple[np.ndarray, int, np.ndarray, dict[int, float], int, float]:
    """Elbow sweep over k, returning the labelling with the highest ARI."""
    inertias: list[float] = []
    aris: dict[int, float] = {}
    labels_by_k: dict[int, np.ndarray] = {}

    for k in KMEANS_K_VALUES:
        model = _best_kmeans(X, k)
        assert model.labels_ is not None
        inertias.append(model.inertia_)
        labels_by_k[k] = model.labels_
        if k >= 2:  # ARI is undefined/degenerate for a single cluster
            aris[k] = float(adjusted_rand_score(y, model.labels_))

    inertia_arr = np.array(inertias)
    elbow_k = KMEANS_K_VALUES[_knee_index(inertia_arr)]
    best_k = max(aris, key=lambda k: aris[k])
    _plot_elbow(name, inertia_arr, elbow_k)
    return labels_by_k[best_k], elbow_k, inertia_arr, aris, best_k, aris[best_k]


def _plot_elbow(name: str, inertias: np.ndarray, elbow_k: int) -> None:
    """Inertia against k, with the detected elbow marked."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(KMEANS_K_VALUES, inertias, marker="o", color="#2ca02c")
    ax.axvline(elbow_k, ls="--", color="#d62728", lw=1, label=f"elbow at k={elbow_k}")
    ax.set_xlabel("number of clusters k")
    ax.set_ylabel("inertia (within-cluster SSE)")
    ax.set_title(f"K-Means elbow - {name}")
    ax.set_xticks(list(KMEANS_K_VALUES))
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, f"exp7_elbow_{name}.png")


def run_dbscan(
    X: np.ndarray, y: np.ndarray, name: str
) -> tuple[DBSCANResult, float, list[DBSCANResult]]:
    """Pick eps from the k-distance knee, then sweep around it for the best ARI."""
    k_dist = _k_distances(X, DBSCAN_MIN_SAMPLES)
    eps_knee = float(k_dist[_knee_index(k_dist)])
    _plot_k_distance(name, k_dist, eps_knee)

    sweep: list[DBSCANResult] = []
    for multiplier in EPS_MULTIPLIERS:
        eps = eps_knee * multiplier
        labels = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(X)
        sweep.append(
            DBSCANResult(
                eps=eps,
                labels=labels,
                ari=float(adjusted_rand_score(y, labels)),
                n_clusters=int(np.unique(labels[labels != NOISE]).shape[0]),
                noise_fraction=float(np.mean(labels == NOISE)),
            )
        )
    return max(sweep, key=lambda r: r.ari), eps_knee, sweep


def _plot_k_distance(name: str, k_dist: np.ndarray, eps_knee: float) -> None:
    """Sorted k-distance curve used to justify the choice of eps."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(k_dist.shape[0]), k_dist, color="#9467bd")
    ax.axhline(eps_knee, ls="--", color="#d62728", lw=1,
               label=f"knee: eps={eps_knee:.3f}")
    ax.set_xlabel("points sorted by distance")
    ax.set_ylabel(f"distance to {DBSCAN_MIN_SAMPLES}-th nearest neighbour")
    ax.set_title(f"DBSCAN k-distance - {name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, f"exp7_kdistance_{name}.png")


def _scatter(ax: "plt.Axes", Z: np.ndarray, labels: np.ndarray, title: str) -> None:
    """Draw one 2D PCA scatter panel, greying out DBSCAN noise points."""
    for label in np.unique(labels):
        mask = labels == label
        is_noise = label == NOISE
        ax.scatter(
            Z[mask, 0], Z[mask, 1], s=6, alpha=0.6,
            color=NOISE_COLOR if is_noise else None,
            marker="x" if is_noise else "o",
            label="noise" if is_noise else str(label),
        )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def _plot_scatter_panels(
    name: str,
    Z: np.ndarray,
    y: np.ndarray,
    kmeans_labels: np.ndarray,
    dbscan: DBSCANResult,
    best_k: int,
) -> None:
    """Side-by-side PCA scatters coloured by true / K-Means / DBSCAN labels."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    _scatter(axes[0], Z, y, "true class labels")
    _scatter(axes[1], Z, kmeans_labels, f"K-Means clusters (k={best_k})")
    _scatter(axes[2], Z, dbscan.labels, f"DBSCAN clusters (eps={dbscan.eps:.3f})")
    for ax in axes:
        ax.legend(markerscale=2, fontsize=7, loc="best")
    fig.suptitle(f"First two principal components - {name}")
    savefig(fig, f"exp7_pca_scatter_{name}.png")


def _print_report(result: UnsupervisedResult, sweep: list[DBSCANResult]) -> None:
    """Print the numbers the report's unsupervised section quotes."""
    print(f"  [{result.name}]")
    print(
        f"    PCA: {result.n_components_90} components reach "
        f"{VARIANCE_TARGET:.0%} variance "
        f"(PC1+PC2 = {result.cumulative_variance[1]:.1%})"
    )
    print(f"    K-Means: elbow at k={result.elbow_k}")
    for k, ari in result.kmeans_aris.items():
        marker = "  <- best ARI" if k == result.best_k else ""
        print(f"      k={k:2d}  inertia={result.inertias[k - 1]:10.1f}  "
              f"ARI={ari:+.4f}{marker}")
    print(f"    DBSCAN: k-distance knee at eps={result.eps_knee:.3f}")
    for res in sweep:
        marker = "  <- best ARI" if res.eps == result.dbscan.eps else ""
        print(f"      eps={res.eps:6.3f}  clusters={res.n_clusters:2d}  "
              f"noise={res.noise_fraction:6.1%}  ARI={res.ari:+.4f}{marker}")
    winner = "K-Means" if result.best_kmeans_ari > result.dbscan.ari else "DBSCAN"
    print(
        f"    Best ARI: K-Means {result.best_kmeans_ari:+.4f} (k={result.best_k}) "
        f"vs DBSCAN {result.dbscan.ari:+.4f} (eps={result.dbscan.eps:.3f}) "
        f"-> {winner}"
    )


def run_unsupervised(ds: Dataset) -> UnsupervisedResult:
    """Full PCA / K-Means / DBSCAN pipeline and figures for one dataset."""
    X, y = _prepare(ds)
    Z, n_90, cumulative = run_pca(X, ds.name)

    # Cluster in the retained PCA subspace: it keeps >=90% of the variance while
    # dropping the noise directions that make density estimates unreliable in high dimensions
    kmeans_labels, elbow_k, inertias, aris, best_k, best_ari = run_kmeans(
        Z, y, ds.name
    )
    dbscan, eps_knee, sweep = run_dbscan(Z, y, ds.name)
    _plot_scatter_panels(ds.name, Z, y, kmeans_labels, dbscan, best_k)

    result = UnsupervisedResult(
        name=ds.name,
        n_components_90=n_90,
        cumulative_variance=cumulative,
        elbow_k=elbow_k,
        inertias=inertias,
        kmeans_aris=aris,
        best_k=best_k,
        best_kmeans_ari=best_ari,
        kmeans_labels=kmeans_labels,
        eps_knee=eps_knee,
        dbscan=dbscan,
    )
    _print_report(result, sweep)
    return result


def run() -> None:
    """Run the unsupervised pipeline on every dataset."""
    print_header("Experiment 7 - Unsupervised analysis (PCA, K-Means, DBSCAN)")
    for ds in get_datasets():
        run_unsupervised(ds)


if __name__ == "__main__":
    run()
