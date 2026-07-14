"""Experiment 6 - Bias-variance decomposition via bootstrap replicates"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from experiments.common import SEED, Split, get_datasets, make_split, print_header, savefig
from src.bagging.random_forest import RandomForestClassifier
from src.boosting.adaboost import AdaBoostClassifier
from src.trees.decision_tree import DecisionTree
from src.utils.preprocessing import bootstrap_sample

DATASET = "breast_cancer"
N_REPLICATES = 100
RF_ESTIMATORS = 30
ADA_ESTIMATORS = 50

BIAS_COLOR = "#1f77b4"
VARIANCE_COLOR = "#ff7f0e"


def _make_models(seed: int) -> dict[str, object]:
    """Fresh model instances seeded for one bootstrap replicate."""
    return {
        "DecisionTree": DecisionTree(criterion="gini", random_state=seed),
        "AdaBoost": AdaBoostClassifier(n_estimators=ADA_ESTIMATORS, random_state=seed),
        "RandomForest": RandomForestClassifier(
            n_estimators=RF_ESTIMATORS, random_state=seed
        ),
    }


def _column_majority(preds: np.ndarray) -> np.ndarray:
    """Majority-vote label per test point across the replicate axis."""
    n_test = preds.shape[1]
    main = np.empty(n_test, dtype=preds.dtype)
    for i in range(n_test):
        values, counts = np.unique(preds[:, i], return_counts=True)
        main[i] = values[np.argmax(counts)]
    return main


def _decompose(preds: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """Return average error, bias and variance from a prediction matrix."""
    main = _column_majority(preds)

    # y_test broadcasts over rows
    error = float(np.mean(preds != y_test))       
    bias = float(np.mean(main != y_test))
    variance = float(np.mean(preds != main))     
    return {"error": error, "bias": bias, "variance": variance}


def run_bias_variance(split: Split) -> dict[str, dict[str, float]]:
    """Train every model on N_REPLICATES bootstraps and decompose the error."""
    n_train = split.X_train.shape[0]
    n_test = split.X_test.shape[0]
    model_names = list(_make_models(SEED))
    
    preds: dict[str, np.ndarray] = {
        name: np.empty((N_REPLICATES, n_test), dtype=split.y_train.dtype)
        for name in model_names
    }

    for b in range(N_REPLICATES):
        seed = SEED + b
        in_bag, _ = bootstrap_sample(n_train, random_state=seed)
        X_b, y_b = split.X_train[in_bag], split.y_train[in_bag]
        for name, model in _make_models(seed).items():
            model.fit(X_b, y_b)  
            preds[name][b] = model.predict(split.X_test)  

    result = {name: _decompose(rows, split.y_test) for name, rows in preds.items()}
    _plot(split.name, result)
    _print_table(split.name, result)
    return result


def _plot(name: str, result: dict[str, dict[str, float]]) -> None:
    """Grouped bar chart of bias vs variance per model"""
    models = list(result)
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    bias = [result[m]["bias"] for m in models]
    variance = [result[m]["variance"] for m in models]
    ax.bar(x - width / 2, bias, width, label="bias", color=BIAS_COLOR)
    ax.bar(x + width / 2, variance, width, label="variance", color=VARIANCE_COLOR)
    ax.plot(
        x, [result[m]["error"] for m in models],
        "D", color="#2ca02c", label="avg error",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("0/1 loss contribution")
    ax.set_title(f"Bias-variance decomposition ({N_REPLICATES} bootstraps) - {name}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    savefig(fig, f"exp6_bias_variance_{name}.png")


def _print_table(name: str, result: dict[str, dict[str, float]]) -> None:
    """Print the per-model error bias variance table for one dataset."""
    print(f"  [{name}] {N_REPLICATES} bootstrap replicates")
    for model_name, parts in result.items():
        print(
            f"    {model_name:14s} error={parts['error']:.4f}  "
            f"bias={parts['bias']:.4f}  variance={parts['variance']:.4f}"
        )


def run() -> None:
    """Run the bias-variance decomposition on the balanced binary dataset."""
    print_header(
        f"Experiment 6 - Bias-variance decomposition "
        f"({N_REPLICATES} bootstraps, dataset={DATASET})"
    )
    datasets = {ds.name: ds for ds in get_datasets()}
    if DATASET not in datasets:
        print(f"  [skip] dataset {DATASET!r} not available")
        return
    split = make_split(datasets[DATASET])
    run_bias_variance(split)


if __name__ == "__main__":
    run()
