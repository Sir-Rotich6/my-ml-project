import sys
import joblib
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parents[2] / "data"
MODELS_DIR = Path(__file__).parents[2] / "models"
TARGET_COL = "Class"
TASK = "classification"  # "classification" or "regression"
RANDOM_STATE = 42


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def evaluate_classification(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "auc_roc":   round(roc_auc_score(y_test, y_prob), 4),
    }
    return metrics


def evaluate_regression(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    metrics = {
        "mae":  round(float(mean_absolute_error(y_test, y_pred)), 4),
        "rmse": round(float(mean_squared_error(y_test, y_pred) ** 0.5), 4),
        "r2":   round(float(r2_score(y_test, y_pred)), 4),
    }
    return metrics


def print_metrics(metrics: dict) -> None:
    width = max(len(k) for k in metrics) + 2
    for name, value in metrics.items():
        print(f"  {name:<{width}}: {value}")


def run(features_path: Path = DATA_DIR / "features.csv") -> dict:
    if not features_path.exists():
        print(f"File not found: {features_path}")
        sys.exit(1)

    MODELS_DIR.mkdir(exist_ok=True)

    print(f"Loading {features_path.name}...")
    X, y = load_data(features_path)
    print(f"  {X.shape[0]:,} rows x {X.shape[1]} features  |  target: '{TARGET_COL}'")

    print("\nSplitting 80/20...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE,
        stratify=y if TASK == "classification" else None,
    )
    print(f"  Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    if TASK == "classification":
        fraud_train = y_train.sum()
        fraud_test = y_test.sum()
        print(f"  Fraud in train: {fraud_train} ({fraud_train/len(y_train)*100:.2f}%)")
        print(f"  Fraud in test:  {fraud_test} ({fraud_test/len(y_test)*100:.2f}%)")

    print("\nTraining baseline model...")
    if TASK == "classification":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ])
    else:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ])

    model.fit(X_train, y_train)
    print(f"  Model: StandardScaler + {model.steps[-1][1].__class__.__name__}")

    print("\nEvaluating on test set...")
    if TASK == "classification":
        metrics = evaluate_classification(model, X_test, y_test)
    else:
        metrics = evaluate_regression(model, X_test, y_test)
    print_metrics(metrics)

    model_path = MODELS_DIR / "baseline.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

    return metrics


if __name__ == "__main__":
    run()
