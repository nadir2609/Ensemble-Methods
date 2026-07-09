from __future__ import annotations

from multiprocessing import Pool

import numpy as np

from src.metrics.evaluation import accuracy
from src.trees.decision_tree import DecisionTree
from src.utils.preprocessing import bootstrap_sample

_TreeJob = tuple[np.ndarray, np.ndarray, dict[str, object], int | None, bool]


def _train_one_tree(job: _TreeJob) -> tuple[DecisionTree, np.ndarray]:
    """Train a single tree on a bootstrap sample; return it with OOB indices."""
    X, y, tree_kwargs, seed, bootstrap = job
    n = X.shape[0]
    if bootstrap:
        in_bag, oob = bootstrap_sample(n, seed)
    else:
        in_bag, oob = np.arange(n), np.empty(0, dtype=np.int64)
    tree = DecisionTree(random_state=seed, **tree_kwargs)  
    tree.fit(X[in_bag], y[in_bag])
    return tree, oob


class RandomForestClassifier:
    """Bootstrap-aggregated ensemble of decision trees."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        max_features: int | str | None = "sqrt",
        min_samples_split: int = 2,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: int = 1,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.estimators_: list[DecisionTree] = []
        self.oob_indices_: list[np.ndarray] = []
        self.classes_: np.ndarray | None = None
        self._oob_score: float | None = None
        self._feature_importances: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        """Train ``n_estimators`` trees, optionally in parallel."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        tree_kwargs: dict[str, object] = {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "max_features": self.max_features,
        }
        jobs: list[_TreeJob] = [
            (
                X, y, tree_kwargs,
                None if self.random_state is None else self.random_state + t,
                self.bootstrap,
            )
            for t in range(self.n_estimators)
        ]

        if self.n_jobs == 1:
            results = [_train_one_tree(job) for job in jobs]
        else:
            with Pool(processes=self.n_jobs) as pool:
                results = pool.map(_train_one_tree, jobs)

        self.estimators_ = [tree for tree, _ in results]
        self.oob_indices_ = [oob for _, oob in results]

        # RF.6 feature importances: average across trees.
        importances = np.mean(
            [tree.feature_importances() for tree in self.estimators_], axis=0
        )
        self._feature_importances = importances

        if self.oob_score:
            self._oob_score = self._compute_oob_score(X, y)
        return self

    def _tree_proba_full(self, tree: DecisionTree, X: np.ndarray) -> np.ndarray:
        """Expand a tree's probabilities to the forest's full class columns."""
        assert self.classes_ is not None and tree.classes_ is not None
        proba = tree.predict_proba(X)
        full = np.zeros((X.shape[0], self.classes_.shape[0]), dtype=np.float64)
        cols = np.searchsorted(self.classes_, tree.classes_)
        full[:, cols] = proba
        return full

    def _compute_oob_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """RF.4: majority-vote each sample using only trees where it was OOB."""
        assert self.classes_ is not None
        votes = np.zeros((X.shape[0], self.classes_.shape[0]), dtype=np.float64)
        for tree, oob in zip(self.estimators_, self.oob_indices_):
            if oob.shape[0] == 0:
                continue
            votes[oob] += self._tree_proba_full(tree, X[oob])
        has_vote = votes.sum(axis=1) > 0
        oob_pred = self.classes_[np.argmax(votes, axis=1)]
        return accuracy(y[has_vote], oob_pred[has_vote])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """RF.5: average the probability vectors of all trees."""
        if not self.estimators_:
            raise RuntimeError("RandomForestClassifier must be fit before predict.")
        assert self.classes_ is not None
        X = np.asarray(X, dtype=np.float64)
        acc = np.zeros((X.shape[0], self.classes_.shape[0]), dtype=np.float64)
        for tree in self.estimators_:
            acc += self._tree_proba_full(tree, X)
        return acc / len(self.estimators_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """RF.5: majority vote (argmax of averaged probabilities)."""
        assert self.classes_ is not None
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    @property
    def oob_score_(self) -> float:
        """Out-of-bag accuracy (requires ``oob_score=True``)."""
        if self._oob_score is None:
            raise AttributeError("oob_score_ is only available when oob_score=True.")
        return self._oob_score

    @property
    def feature_importances_(self) -> np.ndarray:
        """Mean feature importance across all trees in the forest."""
        if self._feature_importances is None:
            raise RuntimeError("RandomForestClassifier must be fit first.")
        return self._feature_importances
