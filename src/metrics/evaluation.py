from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correctly predicted labels."""
    return float(np.mean(y_true == y_pred))


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None
) -> np.ndarray:
    """Return the ``K x K`` confusion matrix with rows=true, cols=predicted."""
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        matrix[int(t), int(p)] += 1
    return matrix


def precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class precision, recall and F1 scores."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(np.float64)
    pred_pos = cm.sum(axis=0).astype(np.float64)
    actual_pos = cm.sum(axis=1).astype(np.float64)

    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    recall = np.divide(tp, actual_pos, out=np.zeros_like(tp), where=actual_pos > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1


def f1_macro(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None
) -> float:
    """Macro-averaged F1 score (unweighted mean over classes)."""
    _, _, f1 = precision_recall_f1(y_true, y_pred, n_classes)
    return float(np.mean(f1))


def _binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUC for a single binary problem via the rank (Mann-Whitney) statistic."""
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    n_pos, n_neg = pos.shape[0], neg.shape[0]
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Rank all scores; ties receive their average rank.
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.shape[0] + 1)
    _assign_average_ranks(scores, ranks)
    rank_sum_pos = ranks[y_true == 1].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _assign_average_ranks(scores: np.ndarray, ranks: np.ndarray) -> None:
    """In-place: replace ranks of tied scores with their average rank."""
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    i = 0
    n = scores.shape[0]
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = ranks[order[i:j + 1]].mean()
            ranks[order[i:j + 1]] = avg
        i = j + 1


def roc_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    """ROC-AUC score.

    For binary problems ``proba`` may be a 1-D array of positive-class
    probabilities or a 2-D ``(n, 2)`` array. For multi-class problems a 2-D
    ``(n, K)`` array is expected and the macro one-vs-rest average is
    returned.
    """
    proba = np.asarray(proba, dtype=np.float64)
    classes = np.unique(y_true)
    if proba.ndim == 1:
        return _binary_auc(y_true, proba)
    if proba.shape[1] == 2 and classes.shape[0] == 2:
        return _binary_auc(y_true, proba[:, 1])

    aucs: list[float] = []
    for k in range(proba.shape[1]):
        binary_true = (y_true == k).astype(np.int64)
        if binary_true.sum() == 0 or binary_true.sum() == binary_true.shape[0]:
            continue
        aucs.append(_binary_auc(binary_true, proba[:, k]))
    return float(np.mean(aucs)) if aucs else float("nan")


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None = None,
    n_classes: int | None = None,
) -> dict[str, float]:
    """Bundle the standard metrics into a dictionary for reporting."""
    results = {
        "accuracy": accuracy(y_true, y_pred),
        "f1_macro": f1_macro(y_true, y_pred, n_classes),
    }
    if proba is not None:
        results["auc"] = roc_auc(y_true, proba)
    return results
