# legacy/

Earlier machine-learning scripts from the original repository, kept for reference. They **do not run** in the current state:

| File | Why it does not run |
|---|---|
| `solutions/*.py`, `ml_ultra_optimized.py`, `reproduction/reproduce_competition_per_category.py`, `reproduction/reproduce_with_nlp_features.py` | Read `data/code-comment-classification/{java,python,pharo}/input/*.csv` (the original per-language NLBSE data), which is not in the repository |
| `reproduction/reproduce_competition_baselines.py` | Imports `src.data`, `src.labels`, `src.utils` — these modules do not exist |

These scripts also solve 19 separate **binary** tasks on the fixed train/test split, so they cannot be compared directly with the repository's multi-label 5-fold results.

The machine-learning baseline used for comparison now is `scripts/ml_baseline.py` (TF-IDF + Logistic Regression, multi-label, same 5 folds as `src/train.py`).
