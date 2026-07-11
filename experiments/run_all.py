"""Single entry point that reproduces every experiment and figure."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import (  
    adaboost_scaling,
    rf_scaling,
    scaling,
)

EXPERIMENTS = {
    1: ("Baseline (tree/stump/sklearn)", scaling.run),
    2: ("AdaBoost scaling", adaboost_scaling.run),
    3: ("Random Forest scaling", rf_scaling.run),
}


def main(selected: list[int] | None = None) -> None:
    """Run the selected experiments (all by default)."""
    to_run = selected or list(EXPERIMENTS.keys())
    start = time.time()
    for num in to_run:
        if num not in EXPERIMENTS:
            print(f"[skip] unknown experiment {num}")
            continue
        _, fn = EXPERIMENTS[num]
        fn()
    print(f"\nAll requested experiments finished in {time.time() - start:.1f}s.")
    print("Figures written to ./figures/")


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:] if a.isdigit()]
    main(args or None)
