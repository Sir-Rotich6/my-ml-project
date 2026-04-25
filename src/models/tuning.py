import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = Path(__file__).parents[2] / "data"
MODELS_DIR = Path(__file__).parents[2] / "models"
TARGET_COL = "Class"
RANDOM_STATE = 42
N_TRIALS = 30
N_FOLDS = 5


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    return df.drop(columns=[TARGET_COL]), df[TARGET_COL]


def make_objective(X_train: pd.DataFrame, y_train: pd.Series, scale_pos_weight: float):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 20),
            "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight":  scale_pos_weight,
            "random_state":      RANDOM_STATE,
            "verbosity":         0,
            "eval_metric":       "aucpr",
        }

        model = XGBClassifier(**params)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring="average_precision",
            n_jobs=1,
        )
        mean_score = scores.mean()

        print(
            f"  Trial {trial.number:>3} | "
            f"AUPRC: {mean_score:.4f} +/- {scores.std():.4f} | "
            f"n_est={params['n_estimators']} depth={params['max_depth']} "
            f"lr={params['learning_rate']:.4f} subs={params['subsample']:.2f}"
        )
        return mean_score

    return objective


def print_metrics(y_test: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    metrics = {
        "auprc":     round(float(average_precision_score(y_test, y_prob)), 4),
        "auc_roc":   round(float(roc_auc_score(y_test, y_prob)), 4),
        "f1":        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
    }
    width = max(len(k) for k in metrics) + 2
    for name, val in metrics.items():
        print(f"  {name:<{width}}: {val}")
    return metrics


def main() -> None:
    features_path = DATA_DIR / "features.csv"
    if not features_path.exists():
        print(f"File not found: {features_path}")
        sys.exit(1)

    MODELS_DIR.mkdir(exist_ok=True)

    # ── Load & split ──────────────────────────────────────────────────────
    print("Loading features.csv...")
    X, y = load_data(features_path)
    print(f"  {X.shape[0]:,} rows x {X.shape[1]} features  |  fraud rate: {y.mean()*100:.3f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}  |  scale_pos_weight: {scale_pos_weight:.1f}")

    # ── Optuna study ──────────────────────────────────────────────────────
    print(f"\nRunning Optuna ({N_TRIALS} trials, {N_FOLDS}-fold CV, scoring=average_precision)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        make_objective(X_train, y_train, scale_pos_weight),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )

    best_trial = study.best_trial
    best_params = best_trial.params
    best_params["scale_pos_weight"] = scale_pos_weight
    best_params["random_state"] = RANDOM_STATE
    best_params["verbosity"] = 0
    best_params["eval_metric"] = "aucpr"

    print(f"\nBest trial: #{best_trial.number}  CV AUPRC: {best_trial.value:.4f}")
    print("Best hyperparameters:")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v}")

    # ── Save best params ──────────────────────────────────────────────────
    params_path = MODELS_DIR / "best_params.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest params saved to {params_path}")

    # ── Train final model on full training set ────────────────────────────
    print("\nTraining final model with best params on full training set...")
    final_model = XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)

    # ── Evaluate on test set ──────────────────────────────────────────────
    print("\nTest set evaluation:")
    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)[:, 1]
    metrics = print_metrics(y_test, y_pred, y_prob)

    # ── Baseline comparison ───────────────────────────────────────────────
    baseline_path = MODELS_DIR / "xgboost.pkl"
    if baseline_path.exists():
        baseline = joblib.load(baseline_path)
        baseline_auprc = average_precision_score(
            y_test, baseline.predict_proba(X_test)[:, 1]
        )
        delta = metrics["auprc"] - baseline_auprc
        sign = "+" if delta >= 0 else ""
        print(f"\n  vs. untuned XGBoost AUPRC {baseline_auprc:.4f}  ({sign}{delta:.4f})")

    # ── Save tuned model ──────────────────────────────────────────────────
    tuned_path = MODELS_DIR / "tuned_model.pkl"
    joblib.dump(final_model, tuned_path)
    print(f"\nTuned model saved to {tuned_path}")


if __name__ == "__main__":
    main()
