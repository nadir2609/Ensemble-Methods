from __future__ import annotations

import numpy as np

_EPS = 1e-12


class Node:
    """A single node of the decision tree."""

    __slots__ = (
        "feature_index", "threshold", "left", "right",
        "value", "samples", "impurity",
    )

    def __init__(
        self,
        value: np.ndarray,
        samples: int,
        impurity: float,
        feature_index: int | None = None,
        threshold: float | None = None,
        left: "Node | None" = None,
        right: "Node | None" = None,
    ) -> None:
        self.value = value
        self.samples = samples
        self.impurity = impurity
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTree:
    """CART decision tree for classification of continuous features."""

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        criterion: str = "gini",
        max_features: int | str | None = None,
        random_state: int | None = None,
    ) -> None:
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

        self.root_: Node | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_: int = 0
        self._importances: np.ndarray | None = None
        self._rng: np.random.Generator = np.random.default_rng(random_state)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "DecisionTree":
        """Grow the tree from training data, optionally with sample weights."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self._rng = np.random.default_rng(self.random_state)

        self.classes_ = np.unique(y)
        n_classes = self.classes_.shape[0]
        # Map arbitrary labels to contiguous indices 0..K-1.
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[v] for v in y], dtype=np.int64)

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)

        self.n_features_ = X.shape[1]
        self._importances = np.zeros(self.n_features_, dtype=np.float64)
        self.root_ = self._build(X, y_idx, sample_weight, n_classes, depth=0)
        return self

    def _n_candidate_features(self) -> int:
        """Resolve ``max_features`` to a concrete number of features."""
        p = self.n_features_
        if self.max_features is None:
            return p
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                return max(1, int(np.sqrt(p)))
            if self.max_features == "log2":
                return max(1, int(np.log2(p)))
            raise ValueError("max_features string must be 'sqrt' or 'log2'")
        return max(1, min(int(self.max_features), p))

    def _build(
        self,
        X: np.ndarray,
        y_idx: np.ndarray,
        w: np.ndarray,
        n_classes: int,
        depth: int,
    ) -> Node:
        """Recursively grow the tree and return the constructed node."""
        counts = np.bincount(y_idx, weights=w, minlength=n_classes)
        total_w = float(counts.sum())
        value = counts / total_w
        impurity = float(self._impurity(counts, total_w))
        node = Node(value=value, samples=y_idx.shape[0], impurity=impurity)

        # Stopping criteria (DT.4).
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or y_idx.shape[0] < self.min_samples_split
            or impurity <= _EPS
        ):
            return node

        feature, threshold, reduction = self._best_split(
            X, y_idx, w, counts, total_w, impurity, n_classes
        )
        if feature is None or reduction <= _EPS:
            return node

        mask = X[:, feature] <= threshold
        if mask.all() or (~mask).all():
            return node  # degenerate split 

        # Accumulate impurity-decrease contribution for feature importances.
        assert self._importances is not None
        self._importances[feature] += total_w * reduction

        node.feature_index = feature
        node.threshold = threshold
        node.left = self._build(X[mask], y_idx[mask], w[mask], n_classes, depth + 1)
        node.right = self._build(X[~mask], y_idx[~mask], w[~mask], n_classes, depth + 1)
        return node

    def _best_split(
        self,
        X: np.ndarray,
        y_idx: np.ndarray,
        w: np.ndarray,
        parent_counts: np.ndarray,
        total_w: float,
        parent_impurity: float,
        n_classes: int,
    ) -> tuple[int | None, float, float]:
        """Find the best split."""
        n = y_idx.shape[0]
        n_candidates = self._n_candidate_features()
        if n_candidates < self.n_features_:
            features = self._rng.choice(
                self.n_features_, size=n_candidates, replace=False
            )
        else:
            features = np.arange(self.n_features_)

        best_feature: int | None = None
        best_threshold = 0.0
        best_reduction = -np.inf

        for j in features:
            xs = X[:, j]
            order = np.argsort(xs, kind="mergesort")
            xs_sorted = xs[order]
            # Splits are only valid between distinct consecutive values.
            distinct = xs_sorted[:-1] != xs_sorted[1:]
            if not distinct.any():
                continue

            w_sorted = w[order]
            onehot = np.zeros((n, n_classes), dtype=np.float64)
            onehot[np.arange(n), y_idx[order]] = w_sorted
            cum_counts = np.cumsum(onehot, axis=0)[:-1]      
            cum_w = np.cumsum(w_sorted)[:-1]                

            right_counts = parent_counts - cum_counts
            right_w = total_w - cum_w

            left_imp = self._impurity(cum_counts, cum_w)
            right_imp = self._impurity(right_counts, right_w)
            child_imp = (cum_w / total_w) * left_imp + (right_w / total_w) * right_imp
            reduction = parent_impurity - child_imp
            reduction[~distinct] = -np.inf

            k = int(np.argmax(reduction))
            if reduction[k] > best_reduction:
                best_reduction = float(reduction[k])
                best_feature = int(j)
                best_threshold = float((xs_sorted[k] + xs_sorted[k + 1]) / 2.0)

        return best_feature, best_threshold, best_reduction

    def _impurity(self, counts: np.ndarray, weight: np.ndarray | float) -> np.ndarray:
        """Vectorised Gini or entropy impurity from class counts."""
        weight_arr = np.asarray(weight, dtype=np.float64)
        safe_weight = np.where(weight_arr <= 0, 1.0, weight_arr)
        probs = counts / safe_weight[..., None]
        if self.criterion == "gini":
            imp = 1.0 - np.sum(probs**2, axis=-1)
        else:
            imp = -np.sum(probs * np.log2(probs + _EPS), axis=-1)
        # Nodes with no weight are pure by convention.
        return np.where(weight_arr <= 0, 0.0, imp)

    def _leaf_value(self, x: np.ndarray) -> np.ndarray:
        node = self.root_
        assert node is not None
        while not node.is_leaf:
            assert node.feature_index is not None and node.threshold is not None
            if x[node.feature_index] <= node.threshold:
                node = node.left  
            else:
                node = node.right  
            assert node is not None
        return node.value

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return the leaf class distribution for each sample."""
        if self.root_ is None:
            raise RuntimeError("DecisionTree must be fit before predict_proba.")
        X = np.asarray(X, dtype=np.float64)
        return np.vstack([self._leaf_value(x) for x in X])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the majority class label for each sample."""
        assert self.classes_ is not None
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    @property
    def depth(self) -> int:
        """Depth of the tree (a single-node tree has depth 0)."""
        def _depth(node: Node | None) -> int:
            if node is None or node.is_leaf:
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))

        return _depth(self.root_)

    @property
    def n_leaves(self) -> int:
        """Number of leaf nodes in the tree."""
        def _count(node: Node | None) -> int:
            if node is None:
                return 0
            if node.is_leaf:
                return 1
            return _count(node.left) + _count(node.right)

        return _count(self.root_)

    def feature_importances(self) -> np.ndarray:
        """Normalised total impurity decrease attributed to each feature."""
        if self._importances is None:
            raise RuntimeError("DecisionTree must be fit before feature_importances.")
        total = self._importances.sum()
        if total <= 0:
            return np.zeros_like(self._importances)
        return self._importances / total

    def __repr__(self) -> str:
        if self.root_ is None:
            return "DecisionTree(unfitted)"
        if self.depth > 4:
            return (
                f"DecisionTree(depth={self.depth}, n_leaves={self.n_leaves}, "
                f"criterion='{self.criterion}')"
            )
        lines: list[str] = []
        self._repr_node(self.root_, depth=0, lines=lines)
        return "\n".join(lines)

    def _repr_node(self, node: Node, depth: int, lines: list[str]) -> None:
        indent = "    " * depth
        dist = np.array2string(node.value, precision=3, suppress_small=True)
        if node.is_leaf:
            cls = int(np.argmax(node.value)) if self.classes_ is None else \
                self.classes_[int(np.argmax(node.value))]
            lines.append(
                f"{indent}leaf: pred={cls} {self.criterion}={node.impurity:.3f} "
                f"samples={node.samples} dist={dist}"
            )
            return
        lines.append(
            f"{indent}[feat {node.feature_index} <= {node.threshold:.3f}] "
            f"{self.criterion}={node.impurity:.3f} samples={node.samples} dist={dist}"
        )
        assert node.left is not None and node.right is not None
        self._repr_node(node.left, depth + 1, lines)
        self._repr_node(node.right, depth + 1, lines)
