import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src" / "features"))

from engineering import create_features, select_features


@pytest.fixture
def base_df():
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        **{f"V{i}": np.random.randn(n) for i in range(1, 29)},
        "Time":   np.linspace(0, 172800, n),
        "Amount": np.abs(np.random.exponential(100, n)),
        "Class":  np.random.choice([0, 1], n, p=[0.998, 0.002]),
    })
    return df


# ── create_features ───────────────────────────────────────────────────────

def test_create_features_adds_columns(base_df):
    result = create_features(base_df)
    assert result.shape[1] > base_df.shape[1]


def test_create_features_does_not_modify_input(base_df):
    cols_before = set(base_df.columns)
    create_features(base_df)
    assert set(base_df.columns) == cols_before


def test_domain_features_present(base_df):
    result = create_features(base_df)
    for col in ["is_small_amount", "amount_log", "is_large_amount",
                "hour_of_day", "is_night", "is_round_amount"]:
        assert col in result.columns, f"Missing: {col}"


def test_statistical_features_present(base_df):
    result = create_features(base_df)
    for col in ["v_mean", "v_std", "v_l2_norm", "v_max_abs"]:
        assert col in result.columns, f"Missing: {col}"


def test_interaction_features_present(base_df):
    result = create_features(base_df)
    for col in ["amount_x_v14", "v12_x_v14", "night_x_large", "v14_minus_v17"]:
        assert col in result.columns, f"Missing: {col}"


def test_is_small_amount_binary(base_df):
    result = create_features(base_df)
    assert set(result["is_small_amount"].unique()).issubset({0, 1})


def test_amount_log_non_negative(base_df):
    result = create_features(base_df)
    assert (result["amount_log"] >= 0).all()


def test_hour_of_day_range(base_df):
    result = create_features(base_df)
    assert result["hour_of_day"].between(0, 23).all()


def test_v_l2_norm_positive(base_df):
    result = create_features(base_df)
    assert (result["v_l2_norm"] > 0).all()


def test_no_nulls_introduced(base_df):
    result = create_features(base_df)
    assert result.isnull().sum().sum() == 0


# ── select_features ───────────────────────────────────────────────────────

def test_select_features_returns_correct_types(base_df):
    df_feat = create_features(base_df)
    selected, df_sel = select_features(df_feat, target_col="Class")
    assert isinstance(selected, list)
    assert isinstance(df_sel, pd.DataFrame)


def test_select_features_target_included(base_df):
    df_feat = create_features(base_df)
    _, df_sel = select_features(df_feat, target_col="Class")
    assert "Class" in df_sel.columns


def test_select_features_target_not_in_selected(base_df):
    df_feat = create_features(base_df)
    selected, _ = select_features(df_feat, target_col="Class")
    assert "Class" not in selected


def test_select_features_removes_high_corr(base_df):
    df_feat = create_features(base_df)
    selected, _ = select_features(df_feat, target_col="Class", corr_threshold=0.95)
    # v_std and v_l2_norm should be dropped (|r| > 0.95 with v_max_abs)
    assert not ("v_std" in selected and "v_l2_norm" in selected)


def test_select_features_all_numeric(base_df):
    df_feat = create_features(base_df)
    selected, df_sel = select_features(df_feat, target_col="Class")
    assert df_sel[selected].select_dtypes(include="number").shape[1] == len(selected)
