import sys
from pathlib import Path

import numpy as np
import pandas as pd

_V_COLS = [f"V{i}" for i in range(1, 29)]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    v = df[[c for c in _V_COLS if c in df.columns]]

    # ------------------------------------------------------------------ #
    # Category 1: Domain-specific features                                #
    # Derived from known fraud-detection patterns in the literature.      #
    # ------------------------------------------------------------------ #

    # Fraudsters test stolen cards with micro-transactions (<$10) before
    # escalating — small amounts are a well-documented card-testing signal.
    df["is_small_amount"] = (df["Amount"] < 10).astype(int)

    # Log transform normalises Amount's heavy right skew (max ~$25k, mean ~$88).
    # Prevents raw magnitude from dominating distance-based models.
    df["amount_log"] = np.log1p(df["Amount"])

    # High-value transactions (>$1000) are disproportionately targeted for fraud;
    # flagging them creates a direct high-risk indicator.
    df["is_large_amount"] = (df["Amount"] > 1_000).astype(int)

    # Dataset spans ~48 h of elapsed seconds; modding by 86400 recovers
    # approximate time-of-day. Fraud spikes between 00:00–06:00 when
    # real-time monitoring and customer call-backs are lightest.
    df["hour_of_day"] = (df["Time"] % 86_400 / 3_600).astype(int)
    df["is_night"] = ((df["hour_of_day"] < 6) | (df["hour_of_day"] >= 22)).astype(int)

    # Automated card-testing scripts typically charge round amounts (e.g. $10.00,
    # $50.00) — organic purchases almost never land on exact whole dollars.
    df["is_round_amount"] = ((df["Amount"] % 1 == 0) & (df["Amount"] > 0)).astype(int)

    # ------------------------------------------------------------------ #
    # Category 2: Statistical aggregations across PCA components          #
    # V1-V28 are orthogonal, so row-wise stats capture overall anomaly    #
    # magnitude without destroying the PCA structure.                     #
    # ------------------------------------------------------------------ #

    # Transactions that deviate from zero across many PCA dimensions at once
    # are statistically rare — high row-mean flags broad-spectrum anomalies.
    df["v_mean"] = v.mean(axis=1)

    # High within-row variance means the transaction is unusual in multiple
    # independent PCA directions simultaneously, a stronger fraud signature.
    df["v_std"] = v.std(axis=1)

    # L2 norm = Euclidean distance from the PCA origin. Legitimate transactions
    # cluster near zero; fraud transactions tend to scatter further out.
    df["v_l2_norm"] = np.sqrt((v ** 2).sum(axis=1))

    # The single strongest component signal — captures extreme values in any
    # one PCA direction that the mean would otherwise average away.
    df["v_max_abs"] = v.abs().max(axis=1)

    # ------------------------------------------------------------------ #
    # Category 3: Interaction features                                     #
    # Combinations where two features together carry more information      #
    # than either alone.                                                   #
    # ------------------------------------------------------------------ #

    # V14 is the most fraud-correlated PCA component (|r| ≈ 0.30).
    # Multiplying by log(Amount) isolates high-value transactions that also
    # trigger this dominant fraud-direction signal — the highest-risk segment.
    df["amount_x_v14"] = df["amount_log"] * df["V14"]

    # V12 and V14 are the top two fraud-correlated components. Their product
    # amplifies cases where both fire simultaneously, which is a stronger
    # combined indicator than either feature ranks individually.
    df["v12_x_v14"] = df["V12"] * df["V14"]

    # A large-value transaction at night combines two independent risk factors.
    # The joint binary flag targets the highest-risk intersection directly.
    df["night_x_large"] = df["is_night"] * df["is_large_amount"]

    # V14 and V17 are both negatively correlated with fraud but in different
    # PCA subspaces. Their difference exposes a complementary fraud signal
    # that the absolute values of either feature don't capture alone.
    df["v14_minus_v17"] = df["V14"] - df["V17"]

    return df


def select_features(
    df: pd.DataFrame,
    target_col: str = "Class",
    corr_threshold: float = 0.95,
    variance_threshold_pct: float = 0.01,
) -> tuple[list[str], pd.DataFrame]:
    numeric = df.select_dtypes(include="number")
    candidates = [c for c in numeric.columns if c != target_col]

    dropped_corr: list[tuple[str, str, float]] = []
    dropped_var: list[tuple[str, float, float]] = []

    # ---- Step 1: Remove highly correlated features ---------------------- #
    # Pairs with |r| > corr_threshold carry near-identical information.     #
    # Keeping the first encountered avoids redundancy and multicollinearity. #
    corr_matrix = df[candidates].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))

    to_drop_corr: set[str] = set()
    for col in upper.columns:
        if col in to_drop_corr:
            continue
        partners = upper.index[upper[col] > corr_threshold].tolist()
        for partner in partners:
            if partner not in to_drop_corr:
                dropped_corr.append((partner, col, round(float(upper.loc[partner, col]), 4)))
                to_drop_corr.add(partner)

    after_corr = [c for c in candidates if c not in to_drop_corr]

    # ---- Step 2: Remove near-zero-variance features --------------------- #
    # Features with variance < 1% of overall mean variance carry little     #
    # signal and can destabilise scaling and regularisation.                 #
    variances = df[after_corr].var()
    overall_var = variances.median()  # median is robust to scale outliers like Time (var ~2e9)
    min_var = variance_threshold_pct * overall_var

    to_drop_var: set[str] = set()
    for col, var in variances.items():
        if var < min_var:
            dropped_var.append((col, round(float(var), 6), round(float(min_var), 6)))
            to_drop_var.add(col)

    selected = [c for c in after_corr if c not in to_drop_var]

    # ---- Log results ---------------------------------------------------- #
    print(f"Feature selection: {len(candidates)} candidates -> {len(selected)} selected")

    if dropped_corr:
        print(f"\nDropped {len(dropped_corr)} features (correlation > {corr_threshold}):")
        for feat, kept, r in dropped_corr:
            print(f"  DROP '{feat}'  |r|={r}  with '{kept}'")
    else:
        print(f"\nNo features dropped for high correlation (threshold {corr_threshold})")

    if dropped_var:
        print(f"\nDropped {len(dropped_var)} features (variance < {variance_threshold_pct:.0%} of mean variance):")
        for feat, var, threshold in dropped_var:
            print(f"  DROP '{feat}'  var={var}  threshold={threshold}")
    else:
        print(f"\nNo features dropped for low variance (threshold {variance_threshold_pct:.0%} of mean var={overall_var:.4f})")

    return selected, df[selected + ([target_col] if target_col in df.columns else [])]


if __name__ == "__main__":
    data_dir = Path(__file__).parents[2] / "data"
    cleaned = data_dir / "cleaned.csv"

    if not cleaned.exists():
        print(f"File not found: {cleaned}")
        sys.exit(1)

    df_raw = pd.read_csv(cleaned)
    print(f"Before: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")

    df_feat = create_features(df_raw)
    new_cols = [c for c in df_feat.columns if c not in df_raw.columns]

    print(f"After:  {df_feat.shape[0]:,} rows x {df_feat.shape[1]} columns")
    print(f"\n{len(new_cols)} new features created:")
    for col in new_cols:
        s = df_feat[col]
        print(f"  {col:<20} min={s.min():>10.3f}  max={s.max():>10.3f}  mean={s.mean():>10.3f}")

    print(f"\n{'='*60}")
    print("Feature Selection")
    print("=" * 60)
    selected, df_selected = select_features(df_feat, target_col="Class")
    print(f"\nFinal feature set ({len(selected)} features):")
    for col in selected:
        print(f"  {col}")
