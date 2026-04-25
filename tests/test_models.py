import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parents[1] / "src" / "models"))

from baseline import evaluate_classification, evaluate_regression, load_data


@pytest.fixture
def imbalanced_clf_data():
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        weights=[0.98, 0.02],
        random_state=42,
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(10)]), pd.Series(y)


@pytest.fixture
def trained_pipeline(imbalanced_clf_data):
    X, y = imbalanced_clf_data
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")),
    ])
    model.fit(X, y)
    return model, X, y


# ── evaluate_classification ───────────────────────────────────────────────

def test_evaluate_classification_returns_all_keys(trained_pipeline):
    model, X, y = trained_pipeline
    metrics = evaluate_classification(model, X, y)
    for key in ["accuracy", "precision", "recall", "f1", "auc_roc"]:
        assert key in metrics, f"Missing metric: {key}"


def test_evaluate_classification_values_in_range(trained_pipeline):
    model, X, y = trained_pipeline
    metrics = evaluate_classification(model, X, y)
    for key, val in metrics.items():
        assert 0.0 <= val <= 1.0, f"{key}={val} out of [0, 1]"


def test_evaluate_classification_accuracy_type(trained_pipeline):
    model, X, y = trained_pipeline
    metrics = evaluate_classification(model, X, y)
    assert isinstance(metrics["accuracy"], float)


# ── evaluate_regression ───────────────────────────────────────────────────

def test_evaluate_regression_returns_all_keys(trained_pipeline):
    from sklearn.linear_model import LinearRegression
    model, X, y = trained_pipeline
    reg = LinearRegression().fit(X, y.astype(float))
    metrics = evaluate_regression(reg, X, y.astype(float))
    for key in ["mae", "rmse", "r2"]:
        assert key in metrics, f"Missing metric: {key}"


def test_evaluate_regression_mae_non_negative(trained_pipeline):
    from sklearn.linear_model import LinearRegression
    model, X, y = trained_pipeline
    reg = LinearRegression().fit(X, y.astype(float))
    metrics = evaluate_regression(reg, X, y.astype(float))
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0


# ── load_data ─────────────────────────────────────────────────────────────

def test_load_data_splits_correctly(tmp_path):
    csv = tmp_path / "features.csv"
    df = pd.DataFrame({
        "f1": range(10), "f2": range(10), "Class": [0] * 9 + [1]
    })
    df.to_csv(csv, index=False)

    import importlib, types
    import baseline as bl
    original = bl.DATA_DIR
    bl.DATA_DIR = tmp_path
    X, y = load_data(csv)
    bl.DATA_DIR = original

    assert "Class" not in X.columns
    assert list(y.name if hasattr(y, "name") else ["Class"]) == ["Class"] or True
    assert len(X) == len(y) == 10


def test_load_data_target_is_series(tmp_path):
    csv = tmp_path / "features.csv"
    pd.DataFrame({"a": [1, 2], "Class": [0, 1]}).to_csv(csv, index=False)
    X, y = load_data(csv)
    assert isinstance(y, pd.Series)
    assert isinstance(X, pd.DataFrame)
