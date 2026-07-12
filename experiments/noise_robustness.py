"""Experiment 5 - Noise robustness under label noise."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from experiments.common import (
    SCALING_MAX_TRAIN, SEED, Split, get_datasets, make_split, print_header, savefig,
)
from src.bagging.random_forest import RandomForestClassifier
from src.boosting.adaboost import AdaBoostClassifier
from src.metrics.evaluation import accuracy, f1_macro
from src.trees.decision_tree import DecisionTree
from src.utils.preprocessing import flip_labels

N_ESTIMATORS = 100
NOISE_LEVELS = (0.0, 0.05, 0.10, 0.20)
MODEL_COLORS = {
    "DecisionTree": "#7f7f7f",
    "AdaBoost": "#d62728",
    "RandomForest": "#2ca02c",
}


def _make_models() -> dict[str, object]:
    """Fresh, identically seeded model instances for one noise level."""
    return {
        "DecisionTree": DecisionTree(criterion="gini", random_state=SEED),
        "AdaBoost": AdaBoostClassifier(n_estimators=N_ESTIMATORS, random_state=SEED),
        "RandomForest": RandomForestClassifier(
            n_estimators=N_ESTIMATORS, random_state=SEED
        ),
    }


def run_noise_robustness(split: Split) -> dict[str, np.ndarray]:
    """Test accuracy of every model across the noise levels for one dataset."""
    test_acc: dict[str, list[float]] = {name: [] for name in _make_models()}
    test_f1: dict[str, list[float]] = {name: [] for name in _make_models()}

    for i, eta in enumerate(NOISE_LEVELS):
        # Distinct but reproducible corruption per level.
        y_noisy = flip_labels(split.y_train, eta, random_state=SEED + i)
        for name, model in _make_models().items():
            model.fit(split.X_train, y_noisy)  
            pred = model.predict(split.X_test) 
            test_acc[name].append(accuracy(split.y_test, pred))
            test_f1[name].append(f1_macro(split.y_test, pred, split.n_classes))

    acc_arr = {name: np.array(vals) for name, vals in test_acc.items()}
    _plot(split.name, acc_arr)
    _print_table(split.name, acc_arr, {n: np.array(v) for n, v in test_f1.items()})
    return acc_arr


def _plot(name: str, acc: dict[str, np.ndarray]) -> None:
    """Line plot of test accuracy versus the label-noise fraction."""
    xs = np.array(NOISE_LEVELS) * 100.0
    fig, ax = plt.subplots(figsize=(6, 4))
    for model_name, ys in acc.items():
        ax.plot(
            xs, ys, marker="o", label=model_name,
            color=MODEL_COLORS.get(model_name),
        )
    ax.set_xlabel("training label noise (%)")
    ax.set_ylabel("test accuracy")
    ax.set_title(f"Noise robustness - {name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, f"exp5_noise_robustness_{name}.png")


def _print_table(
    name: str, acc: dict[str, np.ndarray], f1: dict[str, np.ndarray]
) -> None:
    """Print accuracy per noise level plus the clean->max-noise drop."""
    levels = " ".join(f"{int(e * 100):>4d}%" for e in NOISE_LEVELS)
    print(f"  [{name}] noise levels: {levels}")
    max_noise = int(NOISE_LEVELS[-1] * 100)
    for model_name in acc:
        cells = " ".join(f"{a:5.3f}" for a in acc[model_name])
        drop = acc[model_name][0] - acc[model_name][-1]
        print(
            f"    {model_name:14s} acc=[{cells}]  "
            f"drop(0->{max_noise}%)={drop:+.4f}  "
            f"macroF1@{max_noise}%={f1[model_name][-1]:.4f}"
        )


def run() -> None:
    """Run the noise-robustness sweep on every dataset."""
    print_header(
        f"Experiment 5 - Noise robustness "
        f"(label noise {', '.join(f'{int(e * 100)}%' for e in NOISE_LEVELS)})"
    )
    for ds in get_datasets():
        split = make_split(ds, cap=SCALING_MAX_TRAIN)
        run_noise_robustness(split)


if __name__ == "__main__":
    run()
