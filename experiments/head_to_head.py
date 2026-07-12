"""Experiment 4 - Head-to-head: boosting vs bagging under 5-fold CV."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from experiments.common import (
    SCALING_MAX_TRAIN, SEED, _stratified_cap, get_datasets, print_header, savefig,
)
from src.bagging.random_forest import RandomForestClassifier
from src.boosting.adaboost import AdaBoostClassifier
from src.metrics.evaluation import evaluate
from src.trees.decision_tree import DecisionTree
from src.utils.data_loaders import Dataset
from src.utils.preprocessing import StandardScaler

N_ESTIMATORS = 100
N_FOLDS = 5
METRICS = ("accuracy", "f1_macro", "auc")
METRIC_LABELS = {"accuracy": "accuracy", "f1_macro": "macro-F1", "auc": "ROC-AUC"}
MODEL_COLORS = {
    "DecisionTree": "#7f7f7f",
    "AdaBoost": "#d62728",
    "RandomForest": "#2ca02c",
}


def _make_models() -> dict[str, object]:
    """Fresh, identically seeded model instances for one fold."""
    return {
        "DecisionTree": DecisionTree(criterion="gini", random_state=SEED),
        "AdaBoost": AdaBoostClassifier(n_estimators=N_ESTIMATORS, random_state=SEED),
        "RandomForest": RandomForestClassifier(
            n_estimators=N_ESTIMATORS, random_state=SEED
        ),
    }


def _stratified_folds(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    """Return ``k`` test-index arrays with class proportions preserved."""
    rng = np.random.default_rng(seed)
    fold_id = np.empty(y.shape[0], dtype=np.int64)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        # Round-robin assignment keeps each class balanced across folds.
        fold_id[idx] = np.arange(idx.shape[0]) % k
    return [np.where(fold_id == f)[0] for f in range(k)]


def run_head_to_head(ds: Dataset) -> dict[str, dict[str, tuple[float, float]]]:
    """Cross-validate every model on one dataset; return mean/std per metric."""
    X, y = ds.X, ds.y
    if X.shape[0] > SCALING_MAX_TRAIN:
        X, y = _stratified_cap(X, y, SCALING_MAX_TRAIN, SEED)

    folds = _stratified_folds(y, N_FOLDS, SEED)
    n_classes = ds.n_classes
    # scores[model][metric] -> list of per-fold values
    scores: dict[str, dict[str, list[float]]] = {
        name: {m: [] for m in METRICS} for name in _make_models()
    }

    for test_idx in folds:
        train_idx = np.setdiff1d(
            np.arange(y.shape[0]), test_idx, assume_unique=True
        )

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        y_tr, y_te = y[train_idx], y[test_idx]

        for name, model in _make_models().items():
            model.fit(X_tr, y_tr)  
            metrics = evaluate(
                y_te,
                model.predict(X_te),  
                model.predict_proba(X_te),  
                n_classes,
            )
            for m in METRICS:
                scores[name][m].append(metrics.get(m, float("nan")))

    agg: dict[str, dict[str, tuple[float, float]]] = {}
    for name, per_metric in scores.items():
        agg[name] = {
            m: (float(np.nanmean(vals)), float(np.nanstd(vals)))
            for m, vals in per_metric.items()
        }

    _plot(ds.name, agg)
    _print_table(ds.name, agg)
    return agg


def _plot(name: str, agg: dict[str, dict[str, tuple[float, float]]]) -> None:
    """Grouped bar chart of mean metrics with std-dev error bars."""
    models = list(agg)
    x = np.arange(len(METRICS))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, model_name in enumerate(models):
        means = [agg[model_name][m][0] for m in METRICS]
        stds = [agg[model_name][m][1] for m in METRICS]
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(
            x + offset, means, width, yerr=stds, capsize=3,
            label=model_name, color=MODEL_COLORS.get(model_name),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.set_ylabel("score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Head-to-head ({N_FOLDS}-fold CV, {N_ESTIMATORS} est.) - {name}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    savefig(fig, f"exp4_head_to_head_{name}.png")


def _print_table(name: str, agg: dict[str, dict[str, tuple[float, float]]]) -> None:
    """Print the per-model mean +/- std table for one dataset."""
    print(f"  [{name}]")
    for model_name, per_metric in agg.items():
        cells = " ".join(
            f"{METRIC_LABELS[m]}={per_metric[m][0]:.4f}+/-{per_metric[m][1]:.4f}"
            for m in METRICS
        )
        print(f"    {model_name:14s} {cells}")


def run() -> None:
    """Run the head-to-head cross-validation on every dataset."""
    print_header(
        f"Experiment 4 - Head-to-head: boosting vs bagging "
        f"({N_ESTIMATORS} estimators, {N_FOLDS}-fold CV)"
    )
    for ds in get_datasets():
        run_head_to_head(ds)


if __name__ == "__main__":
    run()
