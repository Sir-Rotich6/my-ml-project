import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

sys.path.insert(0, str(ROOT / "src" / "features"))

from engineering import create_features

_V_COLS = [f"V{i}" for i in range(1, 29)]
_ALL_FEATURE_COLS = (
    ["Time"] + _V_COLS + ["Amount",
    "is_small_amount", "amount_log", "is_large_amount",
    "hour_of_day", "is_night", "is_round_amount",
    "v_mean", "v_std", "v_l2_norm", "v_max_abs",
    "amount_x_v14", "v12_x_v14", "night_x_large", "v14_minus_v17"]
)


def _load_or_skip(filename: str):
    import joblib
    path = MODELS_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} not found — run training first")
    try:
        return joblib.load(path)
    except Exception as exc:
        pytest.skip(f"Could not load {filename}: {exc}")


def _feature_names(model) -> list[str] | None:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    try:
        return list(model.get_booster().feature_names)
    except Exception:
        pass
    try:
        return list(model.booster_.feature_name())
    except Exception:
        pass
    return None


def _make_input(model, n_rows: int = 5) -> pd.DataFrame:
    feat_names = _feature_names(model)
    if feat_names:
        return pd.DataFrame(np.zeros((n_rows, len(feat_names))), columns=feat_names)
    features_csv = DATA_DIR / "features.csv"
    if features_csv.exists():
        df = pd.read_csv(features_csv, nrows=n_rows)
        return df.drop(columns=["Class"], errors="ignore")
    rng = np.random.default_rng(0)
    base = pd.DataFrame({
        **{f"V{i}": rng.standard_normal(n_rows) for i in range(1, 29)},
        "Time": rng.uniform(0, 172800, n_rows),
        "Amount": np.abs(rng.exponential(100, n_rows)),
        "Class": [0] * n_rows,
    })
    return create_features(base).drop(columns=["Class"])


@pytest.fixture(scope="module")
def tuned_model():
    return _load_or_skip("tuned_model.pkl")


@pytest.fixture(scope="module")
def production_model():
    return _load_or_skip("production_model.pkl")


# ── Model loads and predicts ──────────────────────────────────────────────

class TestModelLoadsAndPredicts:
    def test_tuned_model_has_predict_proba(self, tuned_model):
        assert hasattr(tuned_model, "predict_proba")

    def test_tuned_model_has_predict(self, tuned_model):
        assert hasattr(tuned_model, "predict")

    def test_tuned_model_predict_proba_shape(self, tuned_model):
        X = _make_input(tuned_model, n_rows=10)
        proba = tuned_model.predict_proba(X)
        assert proba.shape == (10, 2)

    def test_tuned_model_predict_shape(self, tuned_model):
        X = _make_input(tuned_model, n_rows=10)
        preds = tuned_model.predict(X)
        assert preds.shape == (10,)

    def test_production_model_has_predict_proba(self, production_model):
        assert hasattr(production_model, "predict_proba")

    def test_production_model_predict_proba_shape(self, production_model):
        X = _make_input(production_model, n_rows=10)
        proba = production_model.predict_proba(X)
        assert proba.shape == (10, 2)

    def test_production_model_predict_shape(self, production_model):
        X = _make_input(production_model, n_rows=10)
        preds = production_model.predict(X)
        assert preds.shape == (10,)

    def test_single_row_prediction(self, tuned_model):
        X = _make_input(tuned_model, n_rows=1)
        proba = tuned_model.predict_proba(X)
        assert proba.shape == (1, 2)


# ── Predictions are in expected range ────────────────────────────────────

class TestPredictionRange:
    def test_tuned_probabilities_between_0_and_1(self, tuned_model):
        X = _make_input(tuned_model, n_rows=20)
        proba = tuned_model.predict_proba(X)
        assert (proba >= 0.0).all()
        assert (proba <= 1.0).all()

    def test_tuned_proba_rows_sum_to_one(self, tuned_model):
        X = _make_input(tuned_model, n_rows=20)
        row_sums = tuned_model.predict_proba(X).sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_production_probabilities_between_0_and_1(self, production_model):
        X = _make_input(production_model, n_rows=20)
        proba = production_model.predict_proba(X)
        assert (proba >= 0.0).all()
        assert (proba <= 1.0).all()

    def test_production_proba_rows_sum_to_one(self, production_model):
        X = _make_input(production_model, n_rows=20)
        row_sums = production_model.predict_proba(X).sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_predict_labels_are_binary(self, tuned_model):
        X = _make_input(tuned_model, n_rows=20)
        preds = tuned_model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_negative_v14_raises_fraud_score(self, tuned_model):
        feat_names = _feature_names(tuned_model)
        if feat_names is None or "V14" not in feat_names:
            pytest.skip("Model does not expose V14 feature name")
        normal = pd.DataFrame(np.zeros((1, len(feat_names))), columns=feat_names)
        fraud_like = normal.copy()
        fraud_like["V14"] = -10.0
        score_normal = tuned_model.predict_proba(normal)[0, 1]
        score_fraud = tuned_model.predict_proba(fraud_like)[0, 1]
        assert score_fraud > score_normal, (
            f"Negative V14 should raise fraud probability: {score_fraud:.4f} vs {score_normal:.4f}"
        )

    def test_batch_matches_individual_predictions(self, tuned_model):
        X = _make_input(tuned_model, n_rows=5)
        batch = tuned_model.predict_proba(X)[:, 1]
        individual = np.array([
            tuned_model.predict_proba(X.iloc[[i]])[0, 1]
            for i in range(len(X))
        ])
        np.testing.assert_allclose(batch, individual, atol=1e-6)
