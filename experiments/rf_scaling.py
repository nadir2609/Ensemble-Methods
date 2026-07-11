"""Experiment 3 - Random Forest scaling."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from experiments.common import (
    SCALING_MAX_TRAIN, Split, get_datasets, make_split, print_header, savefig,
)
from src.bagging.random_forest import RandomForestClassifier
from src.metrics.evaluation import accuracy

MAX_TREES = 200
DEPTH_TREES = 50
DEPTHS = [1, 2, 3, 5, 7, 10, 15, 20]


def _n_estimators_curve(split: Split) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Incremental test and OOB accuracy as trees are added to one forest."""
    forest = RandomForestClassifier(
        n_estimators=MAX_TREES, max_depth=None, random_state=42
    ).fit(split.X_train, split.y_train)
    classes = forest.classes_
    assert classes is not None
    k = classes.shape[0]

    test_sum = np.zeros((split.X_test.shape[0], k))
    oob_votes = np.zeros((split.X_train.shape[0], k))
    test_acc, oob_acc = [], []

    for tree, oob in zip(forest.estimators_, forest.oob_indices_):
        test_sum += forest._tree_proba_full(tree, split.X_test)
        test_acc.append(accuracy(split.y_test, classes[np.argmax(test_sum, axis=1)]))
        if oob.shape[0] > 0:
            oob_votes[oob] += forest._tree_proba_full(tree, split.X_train[oob])
        has_vote = oob_votes.sum(axis=1) > 0
        oob_pred = classes[np.argmax(oob_votes, axis=1)]
        oob_acc.append(accuracy(split.y_train[has_vote], oob_pred[has_vote]))

    return np.arange(1, MAX_TREES + 1), np.array(test_acc), np.array(oob_acc)


def _max_depth_curve(split: Split) -> tuple[np.ndarray, np.ndarray]:
    """Test accuracy of a fixed-size forest as max_depth increases."""
    accs = []
    for depth in DEPTHS:
        forest = RandomForestClassifier(
            n_estimators=DEPTH_TREES, max_depth=depth, random_state=42
        ).fit(split.X_train, split.y_train)
        accs.append(accuracy(split.y_test, forest.predict(split.X_test)))
    return np.array(DEPTHS), np.array(accs)


def run_rf_scaling(split: Split) -> None:
    """Produce the two RF scaling figures for one dataset."""
    n_trees, test_acc, oob_acc = _n_estimators_curve(split)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(n_trees, test_acc, label="test accuracy", color="#1f77b4")
    ax.plot(n_trees, oob_acc, label="OOB accuracy", color="#2ca02c", alpha=0.8)
    ax.set_xlabel("number of trees")
    ax.set_ylabel("accuracy")
    ax.set_title(f"RF scaling vs n_estimators - {split.name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, f"exp3_rf_n_estimators_{split.name}.png")

    depths, depth_acc = _max_depth_curve(split)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(depths, depth_acc, marker="o", color="#9467bd")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("test accuracy")
    ax.set_title(f"RF scaling vs max_depth ({DEPTH_TREES} trees) - {split.name}")
    ax.grid(True, alpha=0.3)
    savefig(fig, f"exp3_rf_max_depth_{split.name}.png")

    print(
        f"  [{split.name}] test_acc@200={test_acc[-1]:.4f} "
        f"OOB@200={oob_acc[-1]:.4f} best_depth_acc={depth_acc.max():.4f}"
    )


def run() -> None:
    """Run Random Forest scaling on every dataset."""
    print_header("Experiment 3 - Random Forest scaling")
    for ds in get_datasets():
        split = make_split(ds, cap=SCALING_MAX_TRAIN)
        run_rf_scaling(split)


if __name__ == "__main__":
    run()
