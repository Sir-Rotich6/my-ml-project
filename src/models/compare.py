import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

DATA_DIR = Path(__file__).parents[2] / "data"
MODELS_DIR = Path(__file__).parents[2] / "models"
TARGET_COL = "Class"
RANDOM_STATE = 42

MODEL_RATIONALE = {
    "XGBoost": (
        "Gradient boosting with scale_pos_weight handles extreme imbalance directly. "
        "Captures non-linear V-feature interactions. Robust to outliers (critical given "
        "high z-scores in V-features). Industry standard for tabular fraud detection."
    ),
    "LightGBM": (
        "Leaf-wise tree growth achieves better precision on rare classes than level-wise "
        "XGBoost. Fastest to train on 280k rows — important for 5-fold CV. Native "
        "is_unbalance flag handles the 0.17% fraud rate without manual weight calculation."
    ),
    "RandomForest": (
        "Bagging with balanced class weights produces well-calibrated probabilities — "
        "critical for AUPRC. Random feature subsets prevent any single V-feature from "
        "dominating. Provides diversity as a comparison point against boosting methods."
    ),
}


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def build_models(scale_pos_weight: float) -> dict:
    return {
        "XGBoost": XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            eval_metric="aucpr",
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            is_unbalance=True,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbosity=-1,
        ),
        "RandomForest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def main() -> pd.DataFrame:
    features_path = DATA_DIR / "features.csv"
    if not features_path.exists():
        print(f"File not found: {features_path}")
        sys.exit(1)

    MODELS_DIR.mkdir(exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    print("Loading features.csv...")
    X, y = load_data(features_path)
    n_fraud = int(y.sum())
    print(f"  {X.shape[0]:,} rows x {X.shape[1]} features")
    print(f"  Fraud: {n_fraud} ({y.mean()*100:.3f}%)")

    # ── Split ─────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"scale_pos_weight (XGBoost): {scale_pos_weight:.1f}")

    # ── Train & evaluate ──────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = build_models(scale_pos_weight)
    results = []

    for name, model in models.items():
        print(f"\n{'-'*55}")
        print(f"Model: {name}")
        print(f"Why:   {MODEL_RATIONALE[name]}")
        print(f"{'-'*55}")

        t0 = time.time()

        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring="average_precision",
            n_jobs=-1,
        )

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        test_auprc = average_precision_score(y_test, y_prob)
        elapsed = time.time() - t0

        model_path = MODELS_DIR / f"{name.lower()}.pkl"
        joblib.dump(model, model_path)

        print(f"  CV AUPRC : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        print(f"  Test AUPRC: {test_auprc:.4f}")
        print(f"  Time      : {elapsed:.1f}s")
        print(f"  Saved     : {model_path}")

        results.append({
            "Model":           name,
            "CV Mean (AUPRC)": round(cv_scores.mean(), 4),
            "CV Std":          round(cv_scores.std(), 4),
            "Test AUPRC":      round(test_auprc, 4),
            "Train Time (s)":  round(elapsed, 1),
        })

    # ── Comparison table ──────────────────────────────────────────────────
    df_results = pd.DataFrame(results).sort_values("Test AUPRC", ascending=False)

    print(f"\n{'='*65}")
    print("Model Comparison  (primary metric: AUPRC)")
    print("="*65)
    print(df_results.to_string(index=False))

    best = df_results.iloc[0]
    runner_up = df_results.iloc[1]
    margin = best["Test AUPRC"] - runner_up["Test AUPRC"]

    print(f"\nBest model : {best['Model']}")
    print(f"  Test AUPRC : {best['Test AUPRC']}  (+{margin:.4f} over {runner_up['Model']})")
    print(f"  CV AUPRC   : {best['CV Mean (AUPRC)']} +/- {best['CV Std']}")
    print(f"  Why best   : {MODEL_RATIONALE[best['Model']]}")

    return df_results


if __name__ == "__main__":
    main()
