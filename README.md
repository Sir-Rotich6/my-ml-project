# Credit Card Fraud Detection

> Real-time binary fraud classifier trained on 284K transactions — XGBoost + Optuna tuning achieves **AUPRC 0.81**, **precision 84%**, **recall 80%** on a dataset with only 0.17% fraud.

**[Live Demo →](https://credit-card-fraid-detection.streamlit.app)**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Results](#results)
4. [Feature Engineering](#feature-engineering)
5. [Tech Stack](#tech-stack)
6. [Setup & Installation](#setup--installation)
7. [How to Run](#how-to-run)
8. [Key Decisions & Lessons](#key-decisions--lessons)
9. [File Structure](#file-structure)

---

## Project Overview

### The Problem

Credit card fraud costs the global economy over $32B per year. Detecting it in real time is a hard ML problem because:

- **Severe class imbalance** — only 1 in 600 transactions is fraudulent (0.17%). A model that flags nothing achieves 99.83% accuracy, making accuracy a useless metric.
- **High cost of false negatives** — a missed fraud is real money lost. But too many false positives destroy cardholder trust.
- **Anonymised features** — the dataset uses PCA-transformed components (V1–V28), so domain intuition must be reconstructed from the data itself.

### Who Is This For?

The primary end users are **fraud analysts and risk engineers** at payment processors or banks who need a model they can slot into a real-time transaction scoring pipeline. The secondary audience is **hiring managers** reviewing this as a portfolio piece demonstrating production ML workflow end-to-end.

### The Data

| Property | Value |
|---|---|
| Source | [Kaggle — ULB Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Transactions | 283,726 (after removing 1,081 duplicates) |
| Fraud cases | 473 (0.17%) |
| Legitimate cases | 283,253 (99.83%) |
| Raw features | 31 (V1–V28 PCA components + Time, Amount, Class) |
| Engineered features | +14 domain and interaction features |
| Total model input | 41 features |

### What the Model Outputs

Given a transaction's feature vector, the model outputs a **fraud probability score in [0, 1]**. Scores above a chosen threshold (tuned to balance precision/recall for the business use case) trigger a fraud flag. The portfolio app exposes this as:

- An **interactive scoring widget** (drag sliders, get a live fraud probability gauge)
- A **REST endpoint** via FastAPI (`POST /predict`)

### Key Design Decision

**AUPRC was chosen as the primary optimization metric, not AUC-ROC.** On severely imbalanced datasets, AUC-ROC is optimistic because the large number of true negatives inflates the TPR-FPR curve. AUPRC (area under the precision-recall curve) is sensitive to the minority class and directly reflects the quality of fraud-specific predictions. Every model selection and hyperparameter optimization in this project targeted AUPRC.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Data Pipeline                      │
│                                                         │
│  creditcard.csv (284K rows)                             │
│        │                                                │
│        ▼                                                │
│  loader.py ──► cleaner.py ──► validator.py              │
│                    │               │                    │
│              remove dupes     schema + null checks      │
│                    │                                    │
│                    ▼                                    │
│             quality.py (quality gate — 5 checks)        │
│                    │                                    │
│              cleaned.csv                                │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Feature Engineering                    │
│                                                         │
│  engineering.py                                         │
│  ├── Domain features  (is_small_amount, is_night, ...)  │
│  ├── Statistical aggs (v_mean, v_l2_norm, v_max_abs)    │
│  └── Interactions     (amount_x_v14, v12_x_v14, ...)   │
│                    │                                    │
│              features.csv  (41 columns)                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   Model Training                        │
│                                                         │
│  baseline.py  ──────────────► Logistic Regression       │
│  compare.py   ──────────────► XGBoost (default)         │
│               ──────────────► LightGBM                  │
│               ──────────────► Random Forest             │
│  tuning.py    ──────────────► XGBoost + Optuna  ★       │
│                    │                                    │
│  MLflow tracks every run (params + metrics + artifacts) │
│                    │                                    │
│  tuned_model.pkl  production_model.pkl                  │
└────────────────────┬────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
┌──────────────────┐   ┌─────────────────┐
│  app/api.py      │   │ app/streamlit_  │
│  FastAPI REST    │   │ app.py          │
│  POST /predict   │   │ Portfolio UI    │
│  → fraud score   │   │ → 4-page app    │
└──────────────────┘   └─────────────────┘
```

---

## Results

### Model Comparison

All models evaluated on a held-out 20% stratified test split. CV columns are 5-fold stratified cross-validation on the training set.

| Model | AUPRC | AUC-ROC | Precision | Recall | F1 | CV AUPRC | CV Std | Train Time |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression *(baseline)* | 0.7644 | 0.9535 | 0.7217 | 0.7400 | 0.7307 | 0.7521 | ±0.031 | 3.2s |
| XGBoost (default) | 0.7891 | 0.9641 | 0.7843 | 0.7600 | 0.7720 | 0.7823 | ±0.020 | 45.3s |
| LightGBM | 0.7756 | 0.9612 | 0.7612 | 0.7800 | 0.7705 | 0.7698 | ±0.022 | 18.7s |
| Random Forest | 0.7634 | 0.9589 | 0.7423 | 0.7500 | 0.7461 | 0.7534 | ±0.027 | 87.4s |
| **XGBoost + Optuna** *(winner)* | **0.8106** | **0.9767** | **0.8444** | **0.8000** | **0.8216** | **0.8023** | **±0.016** | 312.4s |

### Improvement: Baseline → Winner

| Metric | Baseline | Winner | Absolute Gain |
|---|---|---|---|
| AUPRC | 0.7644 | 0.8106 | **+4.6pp** |
| AUC-ROC | 0.9535 | 0.9767 | **+2.3pp** |
| Precision | 0.7217 | 0.8444 | **+12.3pp** |
| Recall | 0.7400 | 0.8000 | **+6.0pp** |
| F1 | 0.7307 | 0.8216 | **+9.1pp** |
| CV Stability (std) | ±0.031 | ±0.016 | **2× more stable** |

### Why the Winner Won

The 30-trial Optuna search on AUPRC found that `max_depth=10` + `learning_rate=0.12` + `287 estimators` was the right combination for this feature space. Deeper trees capture the non-linear joint effects of PCA components that shallower defaults miss. The lowest CV standard deviation (±0.016) confirms the improvement is robust across data splits, not a lucky test-set fluke.

**Best hyperparameters found:**

```python
{
    "n_estimators":      287,
    "max_depth":         10,
    "learning_rate":     0.1206,
    "subsample":         0.7993,
    "colsample_bytree":  0.5780,
    "min_child_weight":  4,
    "gamma":             0.2904,
    "reg_alpha":         2.1423,
    "reg_lambda":        0.1013,
    "scale_pos_weight":  599.48,   # legit / fraud ratio
}
```

---

## Feature Engineering

14 features were added on top of the 30 raw PCA components. The top 15 by XGBoost importance are shown below.

### Engineered Feature Definitions

| Feature | Category | How It's Computed | Fraud Signal Rationale |
|---|---|---|---|
| `is_small_amount` | Domain | `Amount < 10` → binary | Card-testing micro-transactions: fraudsters probe stolen cards with small charges before escalating |
| `amount_log` | Domain | `log(1 + Amount)` | Normalises the heavy right-skew (max $25k vs mean $88); prevents raw magnitude dominating |
| `is_large_amount` | Domain | `Amount > 1000` → binary | High-value transactions are disproportionately targeted |
| `hour_of_day` | Domain | `(Time % 86400) / 3600` | Fraud peaks 00:00–06:00 when monitoring and dispute calls are lightest |
| `is_night` | Domain | `hour < 6 OR hour ≥ 22` → binary | Direct binary flag for the highest-risk time window |
| `is_round_amount` | Domain | `Amount % 1 == 0` → binary | Automated scripts charge exact round amounts; organic spend almost never does |
| `v_mean` | Statistical | Row mean of V1–V28 | Transactions deviating from zero across many PCA axes at once are statistically rare |
| `v_std` | Statistical | Row std of V1–V28 | High within-row variance means anomalies in multiple independent PCA directions |
| `v_l2_norm` | Statistical | `√(ΣVᵢ²)` | Euclidean distance from PCA origin; fraud clusters scatter further out |
| `v_max_abs` | Statistical | `max(|V1|…|V28|)` | Captures extreme single-component signals the mean averages away |
| `amount_x_v14` | Interaction | `amount_log × V14` | V14 is the top fraud signal (|r| ≈ 0.30); multiplied by amount isolates high-value fraud |
| `v12_x_v14` | Interaction | `V12 × V14` | Top-two fraud components together are a stronger combined signal than either alone |
| `night_x_large` | Interaction | `is_night × is_large_amount` | Joint flag for the highest-risk intersection: large charge at night |
| `v14_minus_v17` | Interaction | `V14 − V17` | Both components correlate negatively with fraud but in different PCA subspaces; their difference exposes a complementary signal |

### Top 15 Feature Importances (XGBoost Gain)

```
V14            ████████████████████  18.4%  ← dominant fraud PCA axis
V12            █████████████         12.3%
amount_x_v14   ██████████            9.9%   ← engineered
v12_x_v14      █████████             8.8%   ← engineered
V10            ████████              7.5%
V17            ███████               6.4%
v14_minus_v17  ██████                5.2%   ← engineered
amount_log     █████                 4.9%   ← engineered
V4             █████                 4.1%
V11            ████                  3.9%
v_max_abs      ████                  3.4%   ← engineered
V3             ███                   3.0%
V7             ███                   2.8%
Amount         ███                   2.4%
v_l2_norm      ██                    2.1%   ← engineered
```

5 of the top 15 features by gain are engineered — confirming that domain knowledge added signal that the raw PCA components did not carry alone.

---

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| **Python** | 3.9 | Runtime |
| **pandas** | latest | Data loading, cleaning, feature engineering |
| **NumPy** | latest | Vectorised feature computation |
| **scikit-learn** | latest | Preprocessing, evaluation metrics, baseline model |
| **XGBoost** | latest | Primary model — gradient boosting with scale_pos_weight |
| **LightGBM** | latest | Candidate model — leaf-wise boosting with is_unbalance |
| **Optuna** | latest | Bayesian hyperparameter optimisation (30 trials, AUPRC objective) |
| **MLflow** | latest | Experiment tracking — logs params, metrics, and model artifacts |
| **Streamlit** | latest | 4-page interactive portfolio app |
| **FastAPI** | latest | REST API endpoint (`POST /predict`) |
| **Plotly** | latest | Interactive charts in the Streamlit app |
| **pytest** | latest | Test suite — 85 tests across data, features, and models |
| **ruff** | latest | Linting (`src/`, `app/`) — enforces E/F/W rules |
| **Docker** | 29.x | Containerisation — python:3.9-slim + libgomp1 |
| **GitHub Actions** | — | CI/CD — parallel Test and Lint jobs on push/PR to main |
| **joblib** | latest | Model serialisation (`pickle`-compatible `.pkl` files) |

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- Git
- (Optional) Docker Desktop for containerised runs

### Clone and Install

```bash
git clone https://github.com/Sir-Rotich6/my-ml-project.git
cd my-ml-project

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Install the project package (so src/ is importable)
pip install -e .
```

### Get the Data

The raw dataset is not committed (284 MB). Download it from Kaggle and place it at `data/creditcard.csv`:

```bash
# Option A — Kaggle CLI
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip

# Option B — manual download
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place creditcard.csv in the data/ directory
```

> **No data? No problem.** The Streamlit app and tests fall back to synthetic demo data automatically if `data/creditcard.csv` is missing.

---

## How to Run

### 1. Full Training Pipeline

Runs the complete pipeline — clean → engineer features → train all models → tune winner → save results.

```bash
# Step 1: Clean raw data
python src/data/cleaner.py

# Step 2: Engineer features
python src/features/run_features.py

# Step 3: Train and compare all baseline models
python src/models/compare.py

# Step 4: Tune the winner with Optuna (30 trials, ~5 min)
python src/models/tuning.py

# Step 5: Evaluate final model
python src/models/evaluate.py
```

MLflow tracks every run. To view the experiment UI:

```bash
mlflow ui
# Open http://localhost:5000
```

### 2. Streamlit Portfolio App

```bash
streamlit run app/streamlit_app.py
# Open http://localhost:8501
```

The app has four pages:
- **Project Overview** — KPI cards, dataset stats, tech badges
- **Explore the Data** — interactive EDA, correlation heatmap, feature selector
- **Model Results** — comparison table, feature importance chart, confusion matrix, live prediction widget
- **How I Built This** — architecture diagram, 7-day build timeline, key decisions

### 3. FastAPI Prediction Endpoint

```bash
uvicorn app.api:app --reload
# Open http://localhost:8000/docs for the Swagger UI

# Example request
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"V14": -5.2, "V12": -3.1, "Amount": 9.99}'
```

### 4. Docker

```bash
# Build image
docker build -t my-ml-project .

# Run container
docker run -p 8501:8501 my-ml-project

# Or with docker-compose (mounts local data/ and models/ as volumes)
docker-compose up
```

### 5. Tests

```bash
# Run full test suite (85 tests)
pytest tests/ -v

# Run a specific test file
pytest tests/test_model.py -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=term-missing
```

### 6. Linting

```bash
ruff check src/ app/
```

---

## Key Decisions & Lessons

**1. AUPRC over AUC-ROC as the optimization target**
With 0.17% fraud, the ROC curve's large true-negative pool makes it misleadingly optimistic. AUPRC measures exactly what matters: how well the model ranks fraud cases above legitimate ones in the precision-recall space. Switching the Optuna objective from AUC-ROC to AUPRC produced a measurably better model on the test set — the difference was visible in fewer false positives at high-recall operating points.

**2. `scale_pos_weight` instead of SMOTE for class imbalance**
Early experiments used SMOTE (Synthetic Minority Oversampling Technique) applied before cross-validation. Cross-validation scores looked impressive, but the test-set performance was noticeably worse. The root cause: SMOTE generated synthetic fraud samples from the training fold and these blended into the validation fold's neighbourhood, inflating CV metrics. Switching to `scale_pos_weight = 599` (the exact legit/fraud ratio) handled imbalance correctly without any data leakage risk.

**3. Feature engineering added more value than raw hyperparameter tuning**
Before adding the 14 engineered features, the best model (default XGBoost) had AUPRC 0.74. After adding features, the same default XGBoost jumped to 0.789 — a bigger gain than tuning delivered. The lesson: investing in domain-informed features before reaching for hyperparameter search is almost always the higher-ROI move on tabular data.

**4. LightGBM was expected to win — it didn't**
LightGBM's leaf-wise growth, native imbalance handling, and 4× faster training made it the pre-experiment favourite. On this dataset, XGBoost's level-wise splitting consistently outperformed it by ~1.5pp AUPRC. The likely reason: PCA features have roughly equal variance (by design), so XGBoost's symmetric per-level splits align better with the feature geometry than LightGBM's aggressive asymmetric leaf growth.

**5. Deploying from `C:\Windows\System32\` taught me a hard lesson about project location**
The project was originally scaffolded inside `System32`, which blocked every file write, git operation, and Docker command without administrator elevation. Ninety percent of the friction in Day 6 came from working around Windows ACLs on a protected system directory. Always keep project directories under your user home.

---

## File Structure

```
my-ml-project/
│
├── .github/
│   └── workflows/
│       └── ci.yml              # CI: parallel Test (py3.9) + Lint (ruff) jobs
│
├── app/
│   ├── api.py                  # FastAPI REST endpoint — POST /predict
│   ├── dashboard.py            # Early prototype (superseded by streamlit_app.py)
│   └── streamlit_app.py        # 4-page portfolio Streamlit app (1,381 lines)
│
├── data/                       # Gitignored — download separately
│   ├── creditcard.csv          # Raw Kaggle dataset (284K rows)
│   ├── cleaned.csv             # After dedup + validation
│   ├── features.csv            # After feature engineering (41 columns)
│   └── model_results.json      # All model metrics — consumed by Streamlit app
│
├── models/                     # Gitignored — produced by training pipeline
│   ├── baseline.pkl            # Logistic Regression
│   ├── xgboost.pkl             # XGBoost default params
│   ├── lightgbm.pkl            # LightGBM
│   ├── randomforest.pkl        # Random Forest
│   ├── tuned_model.pkl         # XGBoost + Optuna (winner)
│   ├── production_model.pkl    # Alias for deployment
│   └── best_params.json        # Optuna best hyperparameters
│
├── notebooks/
│   └── eda.ipynb               # Exploratory data analysis
│
├── src/
│   ├── data/
│   │   ├── loader.py           # load_csv, print_shape, print_dtypes, print_missing
│   │   ├── cleaner.py          # Dedup, type casting, missing-value handling
│   │   ├── validator.py        # validate_schema, validate_no_nulls
│   │   └── quality.py          # check_data_quality — 5-check quality gate
│   │
│   ├── features/
│   │   ├── engineering.py      # create_features (14 new cols), select_features
│   │   └── run_features.py     # CLI entry point for feature pipeline
│   │
│   └── models/
│       ├── baseline.py         # evaluate_classification, evaluate_regression, load_data
│       ├── compare.py          # Train and compare all 4 candidate models
│       ├── tuning.py           # Optuna 30-trial AUPRC optimisation
│       ├── evaluate.py         # Final model evaluation + confusion matrix
│       ├── train.py            # MLflow-instrumented train() wrapper
│       └── predict.py          # Inference helper
│
├── tests/
│   ├── test_data.py            # 15 tests — loader, validator, quality
│   ├── test_data_quality.py    # 18 tests — quality gate pass/fail scenarios
│   ├── test_features.py        # 27 tests — column count, ranges, math identities
│   ├── test_model.py           # 15 tests — load, predict shape, probability range
│   └── test_models.py          # 10 tests — evaluate_classification, evaluate_regression
│
├── .dockerignore
├── .gitignore
├── Dockerfile                  # python:3.9-slim + libgomp1 + layered cache
├── docker-compose.yml          # Port 8501, data/ and models/ as volumes
├── pyproject.toml              # ruff config — E/F/W rules, line-length 120
├── requirements.txt            # All runtime + dev dependencies
└── setup.py                    # Editable install for src/ package
```

---

## CI/CD

Every push to `main` and every pull request triggers two parallel GitHub Actions jobs:

```
push / pull_request → main
         │
    ┌────┴────┐
    ▼         ▼
 Test job   Lint job
 (py 3.9)  (py 3.11)
    │         │
 pytest    ruff check
 tests/ -v src/ app/
    │         │
    └────┬────┘
         ▼
     ✅ merge
```

Configuration: `.github/workflows/ci.yml` — ruff rules in `pyproject.toml`.

---

## Dataset Citation

> Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi.
> *Calibrating Probability with Undersampling for Unbalanced Classification.*
> In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015.

Dataset: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
