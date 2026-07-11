"""Experiment 2 - AdaBoost scaling with the number of estimators."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from experiments.common import (
    SCALING_MAX_TRAIN, Split, get_datasets, make_split, print_header, savefig,
)
from src.boosting.adaboost import AdaBoostClassifier
from src.metrics.evaluation import accuracy

MAX_ESTIMATORS = 200


def _staged_errors(model: AdaBoostClassifier, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """1 - accuracy after each boosting round."""
    return np.array([1.0 - accuracy(y, pred) for pred in model.staged_predict(X)])


def run_adaboost_scaling(split: Split) -> dict[str, np.ndarray]:
    """Fit AdaBoost once and return staged train/test error curves."""
    model = AdaBoostClassifier(n_estimators=MAX_ESTIMATORS, random_state=42)
    model.fit(split.X_train, split.y_train)

    train_err = _staged_errors(model, split.X_train, split.y_train)
    test_err = _staged_errors(model, split.X_test, split.y_test)
    rounds = np.arange(1, train_err.shape[0] + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rounds, train_err, label="train error", color="#1f77b4")
    ax.plot(rounds, test_err, label="test error", color="#d62728")
    ax.set_xlabel("number of estimators (stumps)")
    ax.set_ylabel("error (1 - accuracy)")
    ax.set_title(f"AdaBoost scaling - {split.name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, f"exp2_adaboost_scaling_{split.name}.png")

    final_test = test_err[-1] if test_err.size else float("nan")
    best_test = test_err.min() if test_err.size else float("nan")
    print(
        f"  [{split.name}] rounds={train_err.shape[0]} "
        f"final_test_err={final_test:.4f} best_test_err={best_test:.4f}"
    )
    return {"rounds": rounds, "train_err": train_err, "test_err": test_err}


def run() -> None:
    """Run AdaBoost scaling on every dataset."""
    print_header("Experiment 2 - AdaBoost scaling (1..200 stumps)")
    for ds in get_datasets():
        split = make_split(ds, cap=SCALING_MAX_TRAIN)
        run_adaboost_scaling(split)


if __name__ == "__main__":
    run()
