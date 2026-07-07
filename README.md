# Ensemble Methods: Boosting vs. Bagging

AI Academy — National AI Center · Machine Learning Final Project (Spring 2026)

From-scratch implementations (NumPy only) of a **Decision Tree (CART)**,
**AdaBoost (SAMME)**, and **Random Forest (bagging)**, together with an
**unsupervised analysis pipeline** (PCA, K-Means, DBSCAN), used to run a
controlled empirical study comparing boosting against bagging.

> No `sklearn.tree.DecisionTreeClassifier`, `sklearn.ensemble.*`, `xgboost`,
> `lightgbm`, or `catboost` are used as implementations. scikit-learn appears
> only as a sanity-check baseline and for `adjusted_rand_score` /
> `StandardScaler` where explicitly permitted by the brief.

## Setup

```bash
# activate the virtual environment (already created as env/)
#   Windows PowerShell:  .\env\Scripts\Activate.ps1
#   bash:                source env/Scripts/activate
pip install -r requirements.txt
```

## Datasets

Raw files live in `data/` (large ones are git-ignored; fetch with
`bash download_data.sh`).

| Dataset                | Task              | Why it was chosen                          |
| ---------------------- | ----------------- | ------------------------------------------ |
| Breast Cancer Wisconsin| Binary            | 30 features → high-dimensional, balanced   |
| Adult Income           | Binary            | Larger, mixed-type, moderate imbalance     |
| Covertype (subset)     | Multi-class (7)   | Minority class ≈0.47% → severe imbalance   |

## Reproducing everything

```bash
python experiments/run_all.py       # regenerates every figure into figures/
python experiments/sanity_check.py  # quick dataset/preprocessing check
```

All random operations are seeded (`random_state=42`) for full reproducibility.

## Repository layout

```
src/
  trees/decision_tree.py      # CART decision tree (weighted-sample aware)
  boosting/adaboost.py        # AdaBoost (SAMME) + DecisionStump
  bagging/random_forest.py    # Random Forest with OOB scoring
  unsupervised/{pca,kmeans,dbscan}.py
  utils/{data_loaders,preprocessing}.py
  metrics/evaluation.py       # accuracy, macro-F1, ROC-AUC (from scratch)
experiments/                  # one module per experiment + run_all.py
tests/                        # pytest suite (target ≥60% coverage)
report/                       # IEEE paper (report.tex → report.pdf)
slides/ , presentation/       # defense slide deck
```

## Testing & quality

```bash
pytest --cov=src --cov-report=term-missing
mypy src
```
