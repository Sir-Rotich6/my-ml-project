import json
import sys
import tempfile
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = Path(__file__).parents[2] / "data"
MODELS_DIR = Path(__file__).parents[2] / "models"
TARGET_COL = "Class"
RANDOM_STATE = 42
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "fraud-detection"

# Fallback XGBoost params if tuning hasn't finished yet
_XGBOOST_DEFAULTS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbosity": 0,
    "eval_metric": "aucpr",
    "random_state": RANDOM_STATE,
}


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    return df.drop(columns=[TARGET_COL]), df[TARGET_COL]


def compute_metrics(model, X: pd.DataFrame, y: pd.Series, prefix: str) -> dict:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        f"{prefix}_accuracy":  round(float(accuracy_score(y, y_pred)), 4),
        f"{prefix}_auprc":     round(float(average_precision_score(y, y_prob)), 4),
        f"{prefix}_auc_roc":   round(float(roc_auc_score(y, y_prob)), 4),
        f"{prefix}_f1":        round(float(f1_score(y, y_pred, zero_division=0)), 4),
        f"{prefix}_precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
        f"{prefix}_recall":    round(float(recall_score(y, y_pred, zero_division=0)), 4),
    }


def log_model_artifact(model, run_id: str, name: str, save_path: Path) -> None:
    joblib.dump(model, save_path)
    mlflow.log_artifact(str(save_path), artifact_path="models")
    print(f"  Artifact saved: {save_path.name} (run_id={run_id[:8]}...)")


def print_metrics_block(metrics: dict) -> None:
    train = {k: v for k, v in metrics.items() if k.startswith("train_")}
    test  = {k: v for k, v in metrics.items() if k.startswith("test_")}
    header = f"  {'Metric':<20} {'Train':>8} {'Test':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for tk, tv in train.items():
        key = tk.replace("train_", "")
        test_key = f"test_{key}"
        print(f"  {key:<20} {tv:>8} {test.get(test_key, ''):>8}")


def build_model_configs(scale_pos_weight: float) -> list[dict]:
    # Load tuned params if available
    params_path = MODELS_DIR / "best_params.json"
    if params_path.exists():
        with open(params_path) as f:
            xgb_params = json.load(f)
        xgb_params["scale_pos_weight"] = scale_pos_weight
        source = "tuned (Optuna)"
    else:
        xgb_params = {**_XGBOOST_DEFAULTS, "scale_pos_weight": scale_pos_weight}
        source = "default (tuning not complete)"

    return [
        {
            "name": "baseline",
            "description": "StandardScaler + LogisticRegression",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]),
            "params": {
                "model_name": "LogisticRegression",
                "scaler": "StandardScaler",
                "max_iter": 1000,
                "solver": "lbfgs",
            },
            "save_path": MODELS_DIR / "baseline_mlflow.pkl",
        },
        {
            "name": "tuned_xgboost",
            "description": f"XGBoost — {source}",
            "model": XGBClassifier(**xgb_params),
            "params": {"model_name": "XGBoost", "param_source": source, **xgb_params},
            "save_path": MODELS_DIR / "tuned_model.pkl",
            "is_production": True,
        },
    ]


def main() -> None:
    features_path = DATA_DIR / "features.csv"
    if not features_path.exists():
        print(f"File not found: {features_path}")
        sys.exit(1)

    MODELS_DIR.mkdir(exist_ok=True)

    # ── MLflow setup ───────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
        print(f"MLflow tracking: {MLFLOW_TRACKING_URI}  experiment: '{EXPERIMENT_NAME}'")
    except Exception as e:
        print(f"[WARN] MLflow server not reachable ({e})")
        print("[WARN] Falling back to local file-based tracking (mlruns/)")
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment(EXPERIMENT_NAME)

    # ── Load & split ───────────────────────────────────────────────────────
    print("\nLoading features.csv...")
    X, y = load_data(features_path)
    print(f"  {X.shape[0]:,} rows x {X.shape[1]} features  |  fraud: {y.sum()} ({y.mean()*100:.3f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── Train & log each config ────────────────────────────────────────────
    configs = build_model_configs(scale_pos_weight)
    production_model = None

    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Run: {cfg['name']}  ({cfg['description']})")
        print("=" * 60)

        with mlflow.start_run(run_name=cfg["name"]):
            run_id = mlflow.active_run().info.run_id

            # Log params
            loggable = {
                k: str(v) if not isinstance(v, (int, float, str, bool)) else v
                for k, v in cfg["params"].items()
            }
            mlflow.log_params(loggable)

            # Train
            print("  Training...")
            cfg["model"].fit(X_train, y_train)

            # Metrics on train + test
            train_metrics = compute_metrics(cfg["model"], X_train, y_train, "train")
            test_metrics  = compute_metrics(cfg["model"], X_test,  y_test,  "test")
            all_metrics   = {**train_metrics, **test_metrics}
            mlflow.log_metrics(all_metrics)

            print_metrics_block(all_metrics)

            # Log artifact
            log_model_artifact(cfg["model"], run_id, cfg["name"], cfg["save_path"])

            if cfg.get("is_production"):
                production_model = cfg["model"]

    # ── Save production model ──────────────────────────────────────────────
    if production_model is not None:
        prod_path = MODELS_DIR / "production_model.pkl"
        joblib.dump(production_model, prod_path)
        print(f"\nProduction model saved to {prod_path}")

    print(f"\nAll runs logged. View at {MLFLOW_TRACKING_URI}")
    print("If the server isn't running, start it with:")
    print("  mlflow server --host 127.0.0.1 --port 5000")


if __name__ == "__main__":
    main()
