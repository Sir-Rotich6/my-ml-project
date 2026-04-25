import sys
import time
from pathlib import Path

import pandas as pd

from engineering import create_features, select_features

DATA_DIR = Path(__file__).parents[2] / "data"
TARGET_COL = "Class"


def main() -> None:
    cleaned = DATA_DIR / "cleaned.csv"
    if not cleaned.exists():
        print(f"File not found: {cleaned}")
        sys.exit(1)

    t0 = time.time()

    # Load
    print("Loading cleaned data...")
    df_raw = pd.read_csv(cleaned)
    print(f"  Loaded: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns  ({time.time() - t0:.2f}s)")

    # Engineer features
    print("\nEngineering features...")
    t1 = time.time()
    df_feat = create_features(df_raw)
    new_cols = [c for c in df_feat.columns if c not in df_raw.columns]
    print(f"  {len(new_cols)} new features added: {df_raw.shape[1]} -> {df_feat.shape[1]} columns  ({time.time() - t1:.2f}s)")

    # Select features
    print("\nSelecting features...")
    t2 = time.time()
    selected, df_selected = select_features(df_feat, target_col=TARGET_COL)
    print(f"  Selection complete  ({time.time() - t2:.2f}s)")

    # Save
    print("\nSaving features.csv...")
    t3 = time.time()
    out_path = DATA_DIR / "features.csv"
    df_selected.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}  ({time.time() - t3:.2f}s)")

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"  Before : {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")
    print(f"  After  : {df_selected.shape[0]:,} rows x {df_selected.shape[1]} columns")
    print(f"  Target : '{TARGET_COL}' included")
    print(f"\n  {len(selected)} feature(s) kept:")
    for col in selected:
        print(f"    {col}")
    print(f"\n  Total elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
