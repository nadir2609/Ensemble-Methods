"""Experiment 1 - Baseline single tree, stump, and sklearn reference."""

from __future__ import annotations

from sklearn.tree import DecisionTreeClassifier as SkTree

from experiments.common import Split, get_datasets, make_split, print_header
from src.boosting.adaboost import DecisionStump
from src.metrics.evaluation import evaluate
from src.trees.decision_tree import DecisionTree


def _row(name: str, metrics: dict[str, float]) -> str:
    auc = metrics.get("auc", float("nan"))
    return (
        f"    {name:22s} acc={metrics['accuracy']:.4f}  "
        f"f1={metrics['f1_macro']:.4f}  auc={auc:.4f}"
    )


def run_baseline(split: Split) -> dict[str, dict[str, float]]:
    """Train the three baselines on one split and return their metrics."""
    n_classes = split.n_classes

    tree = DecisionTree(criterion="gini", random_state=42).fit(split.X_train, split.y_train)
    tree_m = evaluate(
        split.y_test, tree.predict(split.X_test),
        tree.predict_proba(split.X_test), n_classes,
    )

    stump = DecisionStump(random_state=42).fit(split.X_train, split.y_train)
    stump_m = evaluate(
        split.y_test, stump.predict(split.X_test),
        stump.predict_proba(split.X_test), n_classes,
    )

    sk = SkTree(criterion="gini", random_state=42).fit(split.X_train, split.y_train)
    sk_m = evaluate(
        split.y_test, sk.predict(split.X_test),
        sk.predict_proba(split.X_test), n_classes,
    )

    print(f"  [{split.name}] depth={tree.depth} leaves={tree.n_leaves}")
    print(_row("DecisionTree (ours)", tree_m))
    print(_row("DecisionStump (ours)", stump_m))
    print(_row("sklearn DecisionTree", sk_m))
    diff = abs(tree_m["accuracy"] - sk_m["accuracy"])
    flag = "OK" if diff <= 0.02 else "WARN"
    print(f"    |acc(ours) - acc(sklearn)| = {diff:.4f}  [{flag} within 2%]")

    return {"tree": tree_m, "stump": stump_m, "sklearn": sk_m}


def run() -> None:
    """Run the baseline experiment on every dataset."""
    print_header("Experiment 1 - Baseline: single tree vs stump vs sklearn")
    for ds in get_datasets():
        split = make_split(ds)  
        run_baseline(split)


if __name__ == "__main__":
    run()
