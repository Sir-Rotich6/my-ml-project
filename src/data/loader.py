import sys
from pathlib import Path

import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def print_shape(df: pd.DataFrame) -> None:
    rows, cols = df.shape
    print(f"Shape: {rows} rows x {cols} columns")


def print_dtypes(df: pd.DataFrame) -> None:
    print("\nColumn names and data types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")


def print_summary_stats(df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        print("\nNo numeric columns found.")
        return
    stats = numeric.agg(["mean", "std", "min", "max"])
    print("\nSummary statistics (numeric columns):")
    print(stats.to_string())


def print_missing(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print("\nMissing value counts:")
    if missing.empty:
        print("  No missing values.")
        return
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"count": missing, "pct": pct})
    for col, row in report.iterrows():
        print(f"  {col}: {int(row['count'])} ({row['pct']}%)")


def inspect(df: pd.DataFrame) -> None:
    print_shape(df)
    print_dtypes(df)
    print_summary_stats(df)
    print_missing(df)


if __name__ == "__main__":
    data_dir = Path(__file__).parents[2] / "data"
    csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        sys.exit(1)

    for csv_path in csv_files:
        print(f"\n{'='*60}")
        print(f"File: {csv_path.name}")
        print("=" * 60)
        inspect(load_csv(str(csv_path)))
