# Credit Card Fraud Detection

Production ML project for binary fraud classification on the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Exploratory Data Analysis

**Dataset:** 283,726 rows × 31 columns (after removing 1,081 duplicates).
Features are V1–V28 (PCA-anonymised), `Time`, `Amount`, and `Class` (target).
All columns are numeric (`float64`/`int64`). No missing values.

**Key findings:**

- **Severe class imbalance** — only 0.17% of transactions are fraud (~492 samples). Standard accuracy is meaningless here; evaluation must use precision-recall AUC or F1.
- **Strongest fraud signals** — V14, V12, and V10 show the largest distributional separation between classes and will dominate feature importance in tree-based models.
- **PCA features are uncorrelated** — V1–V28 are orthogonal by construction, so multicollinearity is not a concern and no dimensionality reduction is needed.
- **`Amount` is heavily right-skewed** (max ~$25k, mean ~$88) — log-transform before scaling is required to prevent the raw magnitude from distorting distance-based models.
- **Modeling implication** — class weighting (`class_weight="balanced"`) or SMOTE oversampling of the minority class is mandatory before training any classifier.
