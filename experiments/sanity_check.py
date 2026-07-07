"""Quick sanity check that all datasets load and preprocessing works.

Run with:  python experiments/sanity_check.py
Prints shapes, class balance and a scaled-split summary for each dataset so
we can confirm the Step 1 foundation before building the models.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.metrics.evaluation import accuracy, f1_macro, roc_auc
from src.utils.data_loaders import load_all
from src.utils.preprocessing import StandardScaler, flip_labels, train_test_split


def main() -> None:
    for ds in load_all():
        balance = ds.class_balance()
        minority = min(balance.values())
        print(f"=== {ds.name} ===")
        print(f"  shape            : {ds.n_samples} x {ds.n_features}")
        print(f"  classes          : {ds.n_classes} ({'binary' if ds.is_binary else 'multi'})")
        print(f"  class balance    : {{ {', '.join(f'{k}:{v:.4f}' for k, v in balance.items())} }}")
        print(f"  minority fraction: {minority:.4f}")

        X_tr, X_te, y_tr, y_te = train_test_split(ds.X, ds.y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        print(f"  train/test split : {X_tr.shape[0]}/{X_te.shape[0]}")
        print(f"  scaled train mean: {np.abs(X_tr_s.mean(axis=0)).max():.2e} (max |mean|)")

        # Exercise the metrics on a trivial majority-class predictor.
        majority = np.bincount(y_tr).argmax()
        y_pred = np.full_like(y_te, majority)
        print(f"  majority acc/f1  : {accuracy(y_te, y_pred):.3f} / {f1_macro(y_te, y_pred):.3f}")
        _ = flip_labels(y_tr, 0.1, random_state=42)  # smoke-test label flipping
        _ = X_te_s  # scaled test set ready for models
        print()

    print("Sanity check complete: all datasets load and preprocess correctly.")


if __name__ == "__main__":
    main()
