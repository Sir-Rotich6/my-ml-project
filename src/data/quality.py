from __future__ import annotations

import pandas as pd


def check_data_quality(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
    expected_dtypes: dict[str, str] | None = None,
    numeric_bounds: dict[str, tuple[float, float]] | None = None,
    target_col: str | None = None,
) -> dict:
    failures: list[str] = []
    warnings: list[str] = []

    null_counts = df.isnull().sum()
    statistics: dict = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "total_nulls_by_column": {c: int(v) for c, v in null_counts.items() if v > 0},
        "null_rates_by_column": {c: round(df[c].isnull().mean() * 100, 2) for c in df.columns},
        "checks_run": [],
    }

    # Check 1: Schema validation
    statistics["checks_run"].append("schema_validation")
    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            failures.append(f"Missing required columns: {missing}")
    if expected_dtypes:
        for col, expected in expected_dtypes.items():
            if col in df.columns and not str(df[col].dtype).startswith(expected):
                failures.append(
                    f"Column '{col}': expected dtype '{expected}', got '{df[col].dtype}'"
                )

    # Check 2: Row count
    statistics["checks_run"].append("row_count")
    n = len(df)
    if n < 100:
        failures.append(f"Too few rows: {n} (minimum 100 required)")
    elif n < 1000:
        warnings.append(f"Low row count: {n} rows (recommend at least 1,000)")

    # Check 3: Null rates
    statistics["checks_run"].append("null_rates")
    for col in df.columns:
        rate = df[col].isnull().mean()
        if rate > 0.5:
            failures.append(f"Column '{col}' has {rate:.1%} null rate (critical: > 50%)")
        elif rate > 0.2:
            warnings.append(f"Column '{col}' has {rate:.1%} null rate (> 20%)")

    # Check 4: Value ranges
    statistics["checks_run"].append("value_ranges")
    numeric_cols = df.select_dtypes(include="number").columns
    range_issues: dict = {}

    if numeric_bounds:
        for col, (lo, hi) in numeric_bounds.items():
            if col in df.columns:
                n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
                if n_out:
                    warnings.append(
                        f"Column '{col}': {n_out} values outside expected range [{lo}, {hi}]"
                    )
                    range_issues[col] = {"out_of_range_count": n_out, "expected_range": [lo, hi]}
    else:
        for col in numeric_cols:
            series = df[col].dropna()
            std = series.std()
            if std == 0:
                warnings.append(f"Column '{col}' has zero variance (constant column)")
                range_issues[col] = {"issue": "zero_variance"}
                continue
            max_z = float(abs((series - series.mean()) / std).max())
            if max_z > 10:
                warnings.append(
                    f"Column '{col}' has extreme values (max |z-score| = {max_z:.1f})"
                )
                range_issues[col] = {"max_zscore": round(max_z, 2)}

    statistics["value_range_issues"] = range_issues

    # Check 5: Target distribution
    statistics["checks_run"].append("target_distribution")
    if target_col:
        if target_col not in df.columns:
            failures.append(f"Target column '{target_col}' not found in DataFrame")
        else:
            dist = df[target_col].value_counts(normalize=True)
            statistics["target_distribution"] = {str(k): round(float(v), 4) for k, v in dist.items()}
            if len(dist) < 2:
                failures.append(
                    f"Target '{target_col}' has only {len(dist)} class — need at least 2 for classification"
                )
            elif dist.min() < 0.05:
                warnings.append(
                    f"Target '{target_col}' is imbalanced: minority class is {dist.min():.1%} of data (< 5%)"
                )

    return {
        "success": len(failures) == 0,
        "failures": failures,
        "warnings": warnings,
        "statistics": statistics,
    }


if __name__ == "__main__":
    import sys
    import json
    from pathlib import Path
    from loader import load_csv

    data_dir = Path(__file__).parents[2] / "data"
    csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        sys.exit(1)

    for csv_path in csv_files:
        print(f"\n{'='*60}")
        print(f"Quality check: {csv_path.name}")
        print("=" * 60)

        df = load_csv(str(csv_path))
        result = check_data_quality(df, target_col="Class")

        status = "PASSED" if result["success"] else "FAILED"
        print(f"Status: {status}")

        if result["failures"]:
            print("\nFailures:")
            for f in result["failures"]:
                print(f"  [FAIL] {f}")

        if result["warnings"]:
            print("\nWarnings:")
            for w in result["warnings"]:
                print(f"  [WARN] {w}")

        print("\nStatistics:")
        stats = result["statistics"].copy()
        stats.pop("null_rates_by_column", None)
        print(json.dumps(stats, indent=2))
