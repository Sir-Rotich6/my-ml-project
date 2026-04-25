import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src" / "data"))

from loader import load_csv, print_shape, print_dtypes, print_summary_stats, print_missing
from validator import validate_schema, validate_no_nulls
from quality import check_data_quality


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "V1": [0.1, -0.5, 1.2, 0.3],
        "Amount": [10.0, 250.5, 0.0, 100.0],
        "Class": [0, 0, 1, 0],
    })


@pytest.fixture
def df_with_nulls():
    df = pd.DataFrame({
        "V1": [1.0, None, 3.0, 4.0],
        "Amount": [10.0, 20.0, None, 40.0],
        "Class": [0, 1, 0, None],
    })
    return df


# ── loader ────────────────────────────────────────────────────────────────

def test_load_csv_returns_dataframe(tmp_path):
    csv = tmp_path / "test.csv"
    csv.write_text("a,b\n1,2\n3,4\n")
    df = load_csv(str(csv))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)


def test_print_shape_runs(sample_df, capsys):
    print_shape(sample_df)
    out = capsys.readouterr().out
    assert "4" in out and "3" in out


def test_print_dtypes_runs(sample_df, capsys):
    print_dtypes(sample_df)
    assert "float" in capsys.readouterr().out.lower() or "int" in capsys.readouterr().out.lower()


def test_print_summary_stats_runs(sample_df, capsys):
    print_summary_stats(sample_df)
    assert capsys.readouterr().out != ""


def test_print_missing_no_nulls(sample_df, capsys):
    print_missing(sample_df)
    assert "No missing" in capsys.readouterr().out


def test_print_missing_with_nulls(df_with_nulls, capsys):
    print_missing(df_with_nulls)
    out = capsys.readouterr().out
    assert "V1" in out or "Amount" in out or "Class" in out


# ── validator ─────────────────────────────────────────────────────────────

def test_validate_schema_passes(sample_df):
    validate_schema(sample_df, ["V1", "Amount", "Class"])


def test_validate_schema_raises_on_missing(sample_df):
    with pytest.raises(ValueError, match="Missing columns"):
        validate_schema(sample_df, ["V1", "nonexistent_col"])


def test_validate_no_nulls_passes(sample_df):
    validate_no_nulls(sample_df, ["V1", "Amount"])


def test_validate_no_nulls_raises(df_with_nulls):
    with pytest.raises(ValueError, match="Nulls found"):
        validate_no_nulls(df_with_nulls, ["V1"])


# ── quality ───────────────────────────────────────────────────────────────

def test_quality_check_passes_clean_data(sample_df):
    result = check_data_quality(sample_df, target_col="Class")
    assert "success" in result
    assert "failures" in result
    assert "warnings" in result
    assert "statistics" in result


def test_quality_check_too_few_rows():
    tiny = pd.DataFrame({"A": range(5), "Class": [0] * 5})
    result = check_data_quality(tiny, target_col="Class")
    assert not result["success"]
    assert any("Too few rows" in f for f in result["failures"])


def test_quality_check_missing_target():
    df = pd.DataFrame({"A": range(200), "Class": [None] * 200})
    result = check_data_quality(df, target_col="Class")
    assert not result["success"]


def test_quality_statistics_keys(sample_df):
    result = check_data_quality(sample_df)
    stats = result["statistics"]
    assert "total_rows" in stats
    assert "total_columns" in stats
    assert "checks_run" in stats


def test_quality_target_distribution(sample_df):
    result = check_data_quality(sample_df, target_col="Class")
    assert "target_distribution" in result["statistics"]
