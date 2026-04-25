import pandas as pd


def validate_schema(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def validate_no_nulls(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if df[col].isnull().any():
            raise ValueError(f"Nulls found in column: {col}")
