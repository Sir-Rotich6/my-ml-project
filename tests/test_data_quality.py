import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src" / "data"))

from quality import check_data_quality


@pytest.fixture
def clean_df():
    rng = np.random.default_rng(0)
    n = 500
    labels = np.zeros(n, dtype=int)
    labels[:30] = 1
    rng.shuffle(labels)
    return pd.DataFrame({
        "V1": rng.standard_normal(n),
        "V2": rng.standard_normal(n),
        "Amount": np.abs(rng.exponential(100, n)),
        "Class": labels,
    })


# ── Quality gate passes on clean data ────────────────────────────────────

class TestQualityGatePasses:
    def test_success_is_true(self, clean_df):
        result = check_data_quality(clean_df, target_col="Class")
        assert result["success"] is True

    def test_no_failures(self, clean_df):
        result = check_data_quality(clean_df, target_col="Class")
        assert result["failures"] == []

    def test_returns_all_top_level_keys(self, clean_df):
        result = check_data_quality(clean_df, target_col="Class")
        for key in ("success", "failures", "warnings", "statistics"):
            assert key in result

    def test_statistics_has_required_keys(self, clean_df):
        stats = check_data_quality(clean_df, target_col="Class")["statistics"]
        for key in ("total_rows", "total_columns", "checks_run",
                    "total_nulls_by_column", "null_rates_by_column",
                    "value_range_issues"):
            assert key in stats, f"Missing stats key: {key}"

    def test_row_count_is_correct(self, clean_df):
        stats = check_data_quality(clean_df)["statistics"]
        assert stats["total_rows"] == len(clean_df)

    def test_column_count_is_correct(self, clean_df):
        stats = check_data_quality(clean_df)["statistics"]
        assert stats["total_columns"] == len(clean_df.columns)

    def test_all_five_checks_run(self, clean_df):
        stats = check_data_quality(clean_df, target_col="Class")["statistics"]
        assert len(stats["checks_run"]) == 5

    def test_no_nulls_reported(self, clean_df):
        stats = check_data_quality(clean_df)["statistics"]
        assert stats["total_nulls_by_column"] == {}

    def test_target_distribution_present(self, clean_df):
        stats = check_data_quality(clean_df, target_col="Class")["statistics"]
        assert "target_distribution" in stats
        assert len(stats["target_distribution"]) == 2

    def test_schema_validation_with_required_cols(self, clean_df):
        result = check_data_quality(clean_df, required_columns=["V1", "Amount", "Class"])
        assert result["success"] is True
        assert result["failures"] == []


# ── Quality gate catches broken datasets ─────────────────────────────────

class TestQualityGateCatchesBrokenData:
    def test_too_few_rows_fails(self):
        tiny = pd.DataFrame({"A": range(10), "Class": [0] * 9 + [1]})
        result = check_data_quality(tiny, target_col="Class")
        assert not result["success"]
        assert any("Too few rows" in f for f in result["failures"])

    def test_exactly_99_rows_fails(self):
        df = pd.DataFrame({"A": range(99), "Class": [0] * 98 + [1]})
        result = check_data_quality(df, target_col="Class")
        assert not result["success"]

    def test_missing_required_column_fails(self, clean_df):
        result = check_data_quality(clean_df, required_columns=["V1", "nonexistent_col"])
        assert not result["success"]
        assert any("Missing required columns" in f for f in result["failures"])

    def test_high_null_rate_column_fails(self):
        df = pd.DataFrame({
            "A": [np.nan] * 400 + list(range(100)),
            "Class": [0] * 450 + [1] * 50,
        })
        result = check_data_quality(df)
        assert not result["success"]
        assert any("null rate" in f and "critical" in f for f in result["failures"])

    def test_missing_target_col_fails(self, clean_df):
        result = check_data_quality(clean_df, target_col="nonexistent_target")
        assert not result["success"]
        assert any("nonexistent_target" in f for f in result["failures"])

    def test_single_class_target_fails(self):
        df = pd.DataFrame({"A": range(200), "Class": [0] * 200})
        result = check_data_quality(df, target_col="Class")
        assert not result["success"]
        assert any("class" in f.lower() for f in result["failures"])

    def test_wrong_dtype_fails(self, clean_df):
        result = check_data_quality(
            clean_df,
            expected_dtypes={"V1": "object"},
        )
        assert not result["success"]
        assert any("dtype" in f for f in result["failures"])

    def test_multiple_broken_aspects_all_reported(self):
        df = pd.DataFrame({"A": range(5), "Class": [0] * 5})
        result = check_data_quality(df, target_col="Class",
                                    required_columns=["A", "missing_col"])
        assert len(result["failures"]) >= 2
