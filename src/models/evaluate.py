import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

DATA_DIR = Path(__file__).parents[2] / "data"
MODELS_DIR = Path(__file__).parents[2] / "models"
TARGET_COL = "Class"
RANDOM_STATE = 42
FIGURES_DIR = Path(__file__).parents[2] / "notebooks"


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    return df.drop(columns=[TARGET_COL]), df[TARGET_COL]


def load_model(path: Path):
    if not path.exists():
        print(f"[WARN] Model not found: {path}")
        return None
    return joblib.load(path)


def threshold_analysis(y_test: pd.Series, y_prob: np.ndarray) -> pd.DataFrame:
    thresholds = np.arange(0.1, 0.95, 0.05)
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rows.append({
            "threshold": round(t, 2),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
            "flagged":   int(y_pred.sum()),
        })
    return pd.DataFrame(rows)


def plot_pr_roc(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, model in models.items():
        if model is None:
            continue
        y_prob = model.predict_proba(X_test)[:, 1]

        # Precision-Recall
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        auprc = average_precision_score(y_test, y_prob)
        axes[0].plot(rec, prec, label=f"{name} (AUPRC={auprc:.3f})")

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        axes[1].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    axes[0].set_title("Precision-Recall Curve", fontsize=13)
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].legend()
    axes[0].axhline(y_test.mean(), linestyle="--", color="grey", label="Random")

    axes[1].set_title("ROC Curve", fontsize=13)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random")
    axes[1].legend()

    plt.tight_layout()
    out = FIGURES_DIR / "pr_roc_curves.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  Saved: {out.name}")
    plt.show()


def plot_confusion(model, name: str, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    if model is None:
        return
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"Confusion Matrix — {name}", fontsize=12)
    plt.tight_layout()
    out = FIGURES_DIR / f"confusion_{name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  Saved: {out.name}")
    plt.show()


def plot_feature_importance(model, feature_names: list[str], name: str, top_n: int = 20) -> None:
    estimator = model
    # Unwrap pipeline if needed
    if hasattr(model, "steps"):
        estimator = model.steps[-1][1]
    if not hasattr(estimator, "feature_importances_"):
        print(f"  [{name}] No feature_importances_ available — skipping.")
        return

    importances = pd.Series(estimator.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(8, top_n * 0.35 + 1))
    top.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"Top {top_n} Feature Importances — {name}", fontsize=12)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    out = FIGURES_DIR / f"feature_importance_{name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  Saved: {out.name}")
    plt.show()


def evaluate_model(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    if model is None:
        return {}
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "model":     name,
        "auprc":     round(float(average_precision_score(y_test, y_prob)), 4),
        "auc_roc":   round(float(roc_auc_score(y_test, y_prob)), 4),
        "f1":        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
    }


def main() -> None:
    features_path = DATA_DIR / "features.csv"
    if not features_path.exists():
        print(f"File not found: {features_path}")
        sys.exit(1)

    print("Loading data...")
    X, y = load_data(features_path)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Test set: {len(X_test):,} rows  |  Fraud: {y_test.sum()} ({y_test.mean()*100:.2f}%)")

    # Load models
    models = {
        "Baseline":      load_model(MODELS_DIR / "baseline.pkl"),
        "XGBoost":       load_model(MODELS_DIR / "xgboost.pkl"),
        "Tuned XGBoost": load_model(MODELS_DIR / "tuned_model.pkl"),
    }
    available = {k: v for k, v in models.items() if v is not None}
    print(f"  Loaded models: {list(available.keys())}")

    # ── Summary metrics table ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Test Set Metrics Summary")
    print("=" * 60)
    rows = [evaluate_model(name, m, X_test, y_test) for name, m in available.items()]
    df_summary = pd.DataFrame(rows).set_index("model")
    print(df_summary.to_string())

    # ── Threshold analysis on best model ──────────────────────────────────
    best_name = df_summary["auprc"].idxmax()
    best_model = available[best_name]
    y_prob_best = best_model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*60}")
    print(f"Threshold Analysis — {best_name}")
    print("=" * 60)
    df_thresh = threshold_analysis(y_test, y_prob_best)
    print(df_thresh.to_string(index=False))
    best_thresh_row = df_thresh.loc[df_thresh["f1"].idxmax()]
    print(f"\nBest threshold by F1: {best_thresh_row['threshold']}  "
          f"(P={best_thresh_row['precision']}  R={best_thresh_row['recall']}  "
          f"F1={best_thresh_row['f1']}  flags={int(best_thresh_row['flagged'])})")

    # ── Plots ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Generating plots...")
    print("=" * 60)
    plot_pr_roc(available, X_test, y_test)
    for name, model in available.items():
        plot_confusion(model, name, X_test, y_test)
    for name, model in available.items():
        plot_feature_importance(model, list(X_test.columns), name)


if __name__ == "__main__":
    main()
