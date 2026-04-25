import sys
from pathlib import Path

import pandas as pd

from quality import check_data_quality


def clean_data(
    df: pd.DataFrame,
    target_col: str | None = None,
    is_time_series: bool = False,
    categorical_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    df = df.copy()

    # Drop columns with >50% nulls
    null_rates = df.isnull().mean()
    drop_cols = null_rates[null_rates > 0.5].index.tolist()
    if drop_cols:
        print(f"Dropping columns with >50% nulls: {drop_cols}")
        df.drop(columns=drop_cols, inplace=True)

    # Drop rows where target is null
    if target_col and target_col in df.columns:
        before = len(df)
        df.dropna(subset=[target_col], inplace=True)
        dropped = before - len(df)
        if dropped:
            print(f"Dropped {dropped} rows with null target ('{target_col}')")

    # Handle remaining nulls
    non_target_cols = [c for c in df.columns if c != target_col]
    if is_time_series:
        df[non_target_cols] = df[non_target_cols].ffill()
        print("Forward-filled nulls in non-target columns (time series mode)")
    else:
        before = len(df)
        df.dropna(subset=non_target_cols, inplace=True)
        dropped = before - len(df)
        if dropped:
            print(f"Dropped {dropped} rows with nulls in non-target columns")

    # Remove exact duplicate rows
    before = len(df)
    df.drop_duplicates(keep="first", inplace=True)
    dupes = before - len(df)
    if dupes:
        print(f"Removed {dupes} duplicate rows")

    # Convert dtypes
    cat_cols = set(categorical_columns or [])
    for col in df.columns:
        if col in cat_cols:
            df[col] = df[col].astype(str)
        else:
            try:
                converted = pd.to_numeric(df[col], errors="raise")
                df[col] = converted
            except (ValueError, TypeError):
                df[col] = df[col].astype(str)

    # Save cleaned CSV
    out_path = Path(__file__).parents[2] / "data" / "cleaned.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved cleaned data to {out_path} ({len(df)} rows x {len(df.columns)} columns)")

    # Re-run quality gate
    quality_result = check_data_quality(df, target_col=target_col)

    return df, quality_result


if __name__ == "__main__":
    from loader import load_csv

    data_dir = Path(__file__).parents[2] / "data"
    csv_files = [f for f in data_dir.glob("*.csv") if f.name != "cleaned.csv"]

    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        sys.exit(1)

    for csv_path in csv_files:
        print(f"\n{'='*60}")
        print(f"Cleaning: {csv_path.name}")
        print("=" * 60)

        df_raw = load_csv(str(csv_path))
        rows_before, cols_before = df_raw.shape
        print(f"Before: {rows_before:,} rows x {cols_before} columns")

        df_clean, quality = clean_data(df_raw, target_col="Class")

        rows_after, cols_after = df_clean.shape
        print(f"After:  {rows_after:,} rows x {cols_after} columns")
        print(f"Removed: {rows_before - rows_after:,} rows, {cols_before - cols_after} columns")

        status = "PASSED" if quality["success"] else "FAILED"
        print(f"\nQuality gate: {status}")
        if quality["failures"]:
            for f in quality["failures"]:
                print(f"  [FAIL] {f}")
        if quality["warnings"]:
            for w in quality["warnings"]:
                print(f"  [WARN] {w}")
