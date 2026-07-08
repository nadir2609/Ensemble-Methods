

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from src.trees.decision_tree import DecisionTree

_ERR_FLOOR = 1e-10


class DecisionStump(DecisionTree):
    """Convenience subclass: a depth-1 tree (a single binary split)."""

    def __init__(
        self, criterion: str = "gini", random_state: int | None = None
    ) -> None:
        super().__init__(max_depth=1, criterion=criterion, random_state=random_state)


class AdaBoostClassifier:
    """Discrete SAMME AdaBoost using decision stumps as weak learners."""

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        criterion: str = "gini",
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.random_state = random_state

        self.estimators_: list[DecisionStump] = []
        self.classes_: np.ndarray | None = None
        self._estimator_weights: list[float] = []
        self._estimator_errors: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoostClassifier":
        """Fit the boosted ensemble via the SAMME weight-update rule."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = self.classes_.shape[0]
        n_samples = X.shape[0]

        # (1) Initialise uniform sample weights.
        w = np.full(n_samples, 1.0 / n_samples, dtype=np.float64)

        self.estimators_ = []
        self._estimator_weights = []
        self._estimator_errors = []

        for m in range(self.n_estimators):
            # Seed each round distinctly but reproducibly (random_state + m).
            seed = None if self.random_state is None else self.random_state + m
            stump = DecisionStump(criterion=self.criterion, random_state=seed)
            stump.fit(X, y, sample_weight=w)  

            incorrect = stump.predict(X) != y
            
            err = float(np.average(incorrect, weights=w))
            if err >= 1.0 - 1.0 / n_classes:
                if m == 0:
                    raise ValueError(
                        "BaseClassifier in AdaBoost is worse than random; "
                        "ensemble cannot be fit."
                    )
                break
            err = max(err, _ERR_FLOOR)

            alpha = self.learning_rate * (
                np.log((1.0 - err) / err) + np.log(n_classes - 1)
            )

            w = w * np.exp(alpha * incorrect)
            w /= w.sum()

            self.estimators_.append(stump)
            self._estimator_weights.append(alpha)
            self._estimator_errors.append(err)

            if err <= _ERR_FLOOR:
                break  
        return self

    def _class_index(self, labels: np.ndarray) -> np.ndarray:
        """Map predicted labels to column indices in ``classes_``."""
        assert self.classes_ is not None
        return np.searchsorted(self.classes_, labels)

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """Accumulated weighted votes per class, shape ``(n_samples, K)``."""
        assert self.classes_ is not None
        n_classes = self.classes_.shape[0]
        scores = np.zeros((X.shape[0], n_classes), dtype=np.float64)
        for stump, alpha in zip(self.estimators_, self._estimator_weights):
            idx = self._class_index(stump.predict(X))
            scores[np.arange(X.shape[0]), idx] += alpha
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the class with the highest total weighted vote."""
        if not self.estimators_:
            raise RuntimeError("AdaBoostClassifier must be fit before predict.")
        assert self.classes_ is not None
        X = np.asarray(X, dtype=np.float64)
        scores = self._decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability estimates from normalised weighted votes. """
        X = np.asarray(X, dtype=np.float64)
        scores = self._decision_function(X)
        total = scores.sum(axis=1, keepdims=True)
        total[total == 0.0] = 1.0
        return scores / total

    def staged_predict(self, X: np.ndarray) -> Iterator[np.ndarray]:
        """Yield the ensemble prediction after each successive boosting round."""
        assert self.classes_ is not None
        X = np.asarray(X, dtype=np.float64)
        n_classes = self.classes_.shape[0]
        scores = np.zeros((X.shape[0], n_classes), dtype=np.float64)
        for stump, alpha in zip(self.estimators_, self._estimator_weights):
            idx = self._class_index(stump.predict(X))
            scores[np.arange(X.shape[0]), idx] += alpha
            yield self.classes_[np.argmax(scores, axis=1)]

    @property
    def estimator_weights(self) -> np.ndarray:
        """The ``alpha_m`` weight assigned to each weak learner."""
        return np.array(self._estimator_weights, dtype=np.float64)

    @property
    def estimator_errors(self) -> np.ndarray:
        """The weighted error ``epsilon_m`` of each weak learner."""
        return np.array(self._estimator_errors, dtype=np.float64)
