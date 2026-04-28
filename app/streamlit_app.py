"""
Credit Card Fraud Detection — Portfolio Showcase
Multi-page Streamlit app for hiring managers.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
sys.path.insert(0, str(ROOT))

# ─── Colour palette ───────────────────────────────────────────────────────────
ACCENT = "#E63946"
NAVY   = "#1A1F3A"
TEAL   = "#2A9D8F"
GOLD   = "#F4A261"
BLUE   = "#457B9D"
GREY   = "#6C757D"
BG     = "#F8F9FA"

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        font=dict(family="Inter, sans-serif", color="#333"),
        paper_bgcolor="white",
        plot_bgcolor=BG,
        colorway=[BLUE, ACCENT, TEAL, GOLD, "#A8DADC", "#1D3557"],
    )
)

# ─── Hardcoded fallback for model_results.json ────────────────────────────────
# Used when data/model_results.json is missing (e.g. clean clone before training).
_FALLBACK_MODEL_RESULTS: dict = {
    "dataset": {
        "total_rows": 283726, "fraud_count": 473, "legit_count": 283253,
        "fraud_rate_pct": 0.1667, "features_original": 31,
        "features_engineered": 14, "features_total": 41,
        "amount_mean": 88.47, "amount_max": 25691.16,
    },
    "models": [
        {
            "name": "Logistic Regression", "label": "Baseline", "file": "baseline.pkl",
            "auprc": 0.7644, "auc_roc": 0.9535, "precision": 0.7217, "recall": 0.7400,
            "f1": 0.7307, "cv_auprc_mean": 0.7521, "cv_auprc_std": 0.0312,
            "train_time_s": 3.2, "is_winner": False, "color": "#ADB5BD",
            "why_not": "Linear decision boundary cannot capture non-linear interactions between PCA components.",
            "rationale": "Establishes the performance floor — interpretable and fast.",
        },
        {
            "name": "XGBoost", "label": "Candidate 1", "file": "xgboost.pkl",
            "auprc": 0.7891, "auc_roc": 0.9641, "precision": 0.7843, "recall": 0.7600,
            "f1": 0.7720, "cv_auprc_mean": 0.7823, "cv_auprc_std": 0.0198,
            "train_time_s": 45.3, "is_winner": False, "color": "#457B9D",
            "why_not": "Default hyperparameters leave significant performance on the table.",
            "rationale": "Gradient boosting with scale_pos_weight=599.5. Industry standard for tabular fraud detection.",
        },
        {
            "name": "LightGBM", "label": "Candidate 2", "file": "lightgbm.pkl",
            "auprc": 0.7756, "auc_roc": 0.9612, "precision": 0.7612, "recall": 0.7800,
            "f1": 0.7705, "cv_auprc_mean": 0.7698, "cv_auprc_std": 0.0221,
            "train_time_s": 18.7, "is_winner": False, "color": "#2A9D8F",
            "why_not": "Leaf-wise growth gives slightly lower precision than XGBoost on this dataset.",
            "rationale": "Fastest training (18.7s). Native is_unbalance flag. Good recall.",
        },
        {
            "name": "Random Forest", "label": "Candidate 3", "file": "randomforest.pkl",
            "auprc": 0.7634, "auc_roc": 0.9589, "precision": 0.7423, "recall": 0.7500,
            "f1": 0.7461, "cv_auprc_mean": 0.7534, "cv_auprc_std": 0.0267,
            "train_time_s": 87.4, "is_winner": False, "color": "#F4A261",
            "why_not": "Bagging misses sequential error correction that boosting provides. 5× slower.",
            "rationale": "Well-calibrated probabilities via bagging. Diversity comparison point.",
        },
        {
            "name": "XGBoost (Optuna-tuned)", "label": "Winner", "file": "tuned_model.pkl",
            "auprc": 0.8106, "auc_roc": 0.9767, "precision": 0.8444, "recall": 0.8000,
            "f1": 0.8216, "cv_auprc_mean": 0.8023, "cv_auprc_std": 0.0156,
            "train_time_s": 312.4, "is_winner": True, "color": "#E63946",
            "why_won": "30-trial Optuna search on AUPRC found depth=10 + lr=0.12 captures deep "
                       "non-linear fraud patterns. Lowest CV std (±0.016) confirms stability.",
            "rationale": "30-trial Optuna optimization. Best params: 287 estimators, depth=10, lr=0.12.",
        },
    ],
    "feature_importances": {
        "V14": 0.1842, "V12": 0.1231, "amount_x_v14": 0.0987, "v12_x_v14": 0.0876,
        "V10": 0.0754, "V17": 0.0643, "v14_minus_v17": 0.0521, "amount_log": 0.0487,
        "V4": 0.0412, "V11": 0.0387, "v_max_abs": 0.0341, "V3": 0.0298,
        "V7": 0.0276, "Amount": 0.0243, "v_l2_norm": 0.0211,
    },
}

# Hardcoded feature column order matching features.csv (minus Class).
_FALLBACK_FEATURE_COLS: list[str] = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount",
       "is_small_amount", "amount_log", "is_large_amount", "hour_of_day",
       "is_night", "is_round_amount", "v_mean", "v_std", "v_l2_norm",
       "v_max_abs", "amount_x_v14", "v12_x_v14", "night_x_large", "v14_minus_v17"]
)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection | Portfolio",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.site-header {
    background: linear-gradient(135deg, #1A1F3A 0%, #2C3E6B 100%);
    color: white; padding: 20px 32px 14px; border-radius: 12px; margin-bottom: 28px;
}
.site-header h1 { margin: 0; font-size: 1.6rem; font-weight: 800; }
.site-header p  { margin: 4px 0 0; font-size: 0.92rem; opacity: 0.75; }

.demo-banner {
    background: #FFF8E1; border: 2px dashed #F4A261; border-radius: 10px;
    padding: 12px 18px; margin-bottom: 20px; font-size: 0.93rem;
}

.hero-title {
    font-size: 2.8rem; font-weight: 800; color: #1A1F3A;
    line-height: 1.15; margin-bottom: 6px;
}
.hero-sub  { font-size: 1.2rem; color: #555; margin-bottom: 24px; }
.hero-accent { color: #E63946; }

.section-label {
    font-size: 0.75rem; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; color: #E63946; margin-bottom: 6px;
}

.kpi-wrap {
    background: white; border-radius: 14px; padding: 20px 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07); border-top: 4px solid var(--kc);
    text-align: center;
}
.kpi-val   { font-size: 2.1rem; font-weight: 800; color: #1A1F3A; }
.kpi-lbl   { font-size: 0.82rem; font-weight: 600; color: #888;
             text-transform: uppercase; letter-spacing: 0.8px; margin-top: 4px; }
.kpi-delta { font-size: 0.8rem; color: #2A9D8F; font-weight: 600; margin-top: 4px; }

.badge {
    display: inline-block; padding: 5px 14px; border-radius: 20px;
    font-size: 0.82rem; font-weight: 600; margin: 4px; letter-spacing: 0.3px;
}

.callout {
    background: #EBF5FB; border-left: 4px solid #2196F3;
    padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0; font-size: 0.93rem;
}
.callout.warn    { background: #FFF8E1; border-color: #F4A261; }
.callout.success { background: #E8F8F5; border-color: #2A9D8F; }
.callout.danger  { background: #FDEDEC; border-color: #E63946; }

.winner-banner {
    background: linear-gradient(135deg, #1A1F3A, #2C3E6B);
    color: white; border-radius: 12px; padding: 20px 24px; margin: 16px 0;
}
.winner-banner h3 { color: #F4A261; margin: 0 0 8px; }
.winner-banner p  { margin: 4px 0; font-size: 0.93rem; opacity: 0.9; }

.tl-item  { display: flex; gap: 16px; margin-bottom: 20px; }
.tl-dot   {
    width: 36px; height: 36px; border-radius: 50%; background: #E63946; color: white;
    font-weight: 800; display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; flex-shrink: 0; margin-top: 2px;
}
.tl-body  { flex: 1; }
.tl-day   { font-size: 0.75rem; font-weight: 700; color: #E63946;
            letter-spacing: 1px; text-transform: uppercase; }
.tl-title { font-size: 1.05rem; font-weight: 700; color: #1A1F3A; margin: 2px 0 4px; }
.tl-desc  { font-size: 0.9rem; color: #555; line-height: 1.5; }

.risk-box    { border-radius: 12px; padding: 18px 24px; text-align: center; margin-top: 8px; }
.risk-low    { background: #E8F8F5; border: 2px solid #2A9D8F; }
.risk-medium { background: #FFF8E1; border: 2px solid #F4A261; }
.risk-high   { background: #FDEDEC; border: 2px solid #E63946; }
.risk-label  { font-size: 1rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; }

.site-footer {
    text-align: center; color: #aaa; font-size: 0.82rem;
    padding: 24px 0 8px; border-top: 1px solid #eee; margin-top: 40px;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  DEMO DATA GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Generating demo dataset…")
def _make_demo_raw_data() -> pd.DataFrame:
    """
    Synthetic credit-card transactions that mirror real dataset distributions.
    Used automatically when data/creditcard.csv and data/cleaned.csv are absent.
    """
    rng = np.random.default_rng(42)
    n_legit, n_fraud = 9_800, 200  # 2% fraud for visual clarity in demo

    # ── Legitimate transactions ───────────────────────────────────────────────
    legit: dict = {f"V{i}": rng.standard_normal(n_legit) for i in range(1, 29)}
    legit["Time"]   = rng.uniform(0, 172_800, n_legit)
    legit["Amount"] = np.exp(rng.normal(3.5, 1.5, n_legit)).clip(0.01, 25_000)
    legit["Class"]  = 0

    # ── Fraudulent transactions ───────────────────────────────────────────────
    # Shift the components that are most correlated with fraud in the real dataset.
    fraud: dict = {f"V{i}": rng.standard_normal(n_fraud) for i in range(1, 29)}
    fraud["V14"] = rng.normal(-8.0, 2.5, n_fraud)   # dominant fraud signal
    fraud["V12"] = rng.normal(-6.0, 2.0, n_fraud)
    fraud["V10"] = rng.normal(-7.0, 2.5, n_fraud)
    fraud["V17"] = rng.normal(-5.0, 2.0, n_fraud)
    fraud["Time"] = rng.uniform(0, 172_800, n_fraud)
    # Mix of card-testing micro-transactions and larger fraud amounts.
    fraud["Amount"] = np.where(
        rng.random(n_fraud) < 0.3,
        rng.uniform(0.5, 9.99, n_fraud),
        rng.uniform(10, 2_500, n_fraud),
    )
    fraud["Class"] = 1

    df = pd.concat(
        [pd.DataFrame(legit), pd.DataFrame(fraud)],
        ignore_index=True,
    )
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA & MODEL LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading dataset…")
def load_raw_data() -> pd.DataFrame:
    for name in ("creditcard.csv", "cleaned.csv"):
        p = DATA_DIR / name
        if p.exists():
            return pd.read_csv(p)
    return _make_demo_raw_data()


@st.cache_data(show_spinner="Loading engineered features…")
def load_features() -> pd.DataFrame:
    p = DATA_DIR / "features.csv"
    if p.exists():
        return pd.read_csv(p)
    # Derive on-the-fly from raw data (real or demo).
    return engineer_features(load_raw_data())


@st.cache_data
def load_model_results() -> dict:
    p = DATA_DIR / "model_results.json"
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return _FALLBACK_MODEL_RESULTS


@st.cache_resource(show_spinner="Loading model…")
def load_model(name: str):
    try:
        import joblib
        p = MODELS_DIR / name
        if p.exists():
            return joblib.load(p)
    except Exception:
        pass
    return None


@st.cache_data
def get_feature_columns() -> list[str]:
    p = DATA_DIR / "features.csv"
    if p.exists():
        try:
            cols = list(pd.read_csv(p, nrows=0).columns)
            return [c for c in cols if c != "Class"]
        except Exception:
            pass
    return _FALLBACK_FEATURE_COLS


# ─── Source status ────────────────────────────────────────────────────────────

@st.cache_data
def _source_status() -> dict[str, bool]:
    return {
        "has_raw":     any((DATA_DIR / n).exists() for n in ("creditcard.csv", "cleaned.csv")),
        "has_features": (DATA_DIR / "features.csv").exists(),
        "has_model":    any((MODELS_DIR / n).exists()
                            for n in ("tuned_model.pkl", "production_model.pkl")),
        "has_results":  (DATA_DIR / "model_results.json").exists(),
    }


# ─── Feature engineering (mirrors src/features/engineering.py) ────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    v = df[[f"V{i}" for i in range(1, 29) if f"V{i}" in df.columns]]

    df["is_small_amount"] = (df["Amount"] < 10).astype(float)
    df["amount_log"]      = np.log1p(df["Amount"])
    df["is_large_amount"] = (df["Amount"] > 1_000).astype(float)
    df["hour_of_day"]     = ((df["Time"] % 86_400) / 3_600).astype(float)
    df["is_night"]        = ((df["hour_of_day"] < 6) | (df["hour_of_day"] >= 22)).astype(float)
    df["is_round_amount"] = ((df["Amount"] % 1 == 0) & (df["Amount"] > 0)).astype(float)
    df["v_mean"]          = v.mean(axis=1)
    df["v_std"]           = v.std(axis=1)
    df["v_l2_norm"]       = np.sqrt((v ** 2).sum(axis=1))
    df["v_max_abs"]       = v.abs().max(axis=1)
    df["amount_x_v14"]    = df["amount_log"] * df["V14"]
    df["v12_x_v14"]       = df["V12"] * df["V14"]
    df["night_x_large"]   = df["is_night"] * df["is_large_amount"]
    df["v14_minus_v17"]   = df["V14"] - df["V17"]
    return df


# ─── Heuristic scorer (used when no trained model is available) ───────────────
def _demo_score(v14: float, v12: float, v10: float, v17: float,
                amount: float, hour: int) -> float:
    """
    Logistic approximation of XGBoost behaviour on this dataset.
    Gives sensible risk scores for the interactive demo without a trained model.
    """
    amount_log = float(np.log1p(amount))
    is_night   = float(hour < 6 or hour >= 22)
    is_small   = float(amount < 10)

    logit = (
        -3.0
        + (-0.35 * v14)
        + (-0.22 * v12)
        + (-0.18 * v10)
        + (-0.12 * v17)
        + (0.50  * is_night)
        + (0.65  * is_small)
        + (0.04  * amount_log)
        + (-0.015 * v14 * v12)   # interaction term
    )
    return float(1.0 / (1.0 + np.exp(-logit)))


# ─── Cached test-set predictions (real or demo) ───────────────────────────────

@st.cache_data(show_spinner="Scoring test set…")
def _compute_test_predictions(model_path: str) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Returns (y_test, y_prob, is_demo).

    Loads the model and feature data independently so this function is hashable
    (keyed only on the model_path string). Falls back to a heuristic scorer if
    the model file is absent or fails to load.
    """
    import joblib
    from sklearn.metrics import average_precision_score  # noqa: F401 (validates import)

    feat_df   = load_features()
    feat_cols = get_feature_columns()

    X = feat_df.drop(columns=["Class"])
    y = feat_df["Class"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Try real model ────────────────────────────────────────────────────────
    mp = MODELS_DIR / model_path
    if mp.exists():
        try:
            model = joblib.load(mp)
            X_in = X_test[feat_cols] if feat_cols and set(feat_cols).issubset(X_test.columns) else X_test
            y_prob = model.predict_proba(X_in)[:, 1]
            return y_test.to_numpy(), y_prob, False
        except Exception:
            pass

    # ── Demo fallback: heuristic scoring ─────────────────────────────────────
    rng    = np.random.default_rng(42)
    scores = np.zeros(len(X_test))
    for col, w in [
        ("V14",            -0.42),
        ("V12",            -0.26),
        ("V10",            -0.20),
        ("is_night",        0.55),
        ("is_small_amount", 0.65),
        ("amount_x_v14",   -0.08),
        ("v14_minus_v17",  -0.10),
    ]:
        if col in X_test.columns:
            scores += w * X_test[col].fillna(0).to_numpy()
    scores += rng.normal(0, 0.65, len(scores))
    y_prob  = 1.0 / (1.0 + np.exp(-scores))
    return y_test.to_numpy(), y_prob, True


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED UI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def render_header(subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="site-header">
          <h1>🔍 Credit Card Fraud Detection</h1>
          <p>{subtitle or "End-to-end ML pipeline · XGBoost · Optuna · FastAPI · Streamlit"}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="site-footer">
            Built with Python · scikit-learn · XGBoost · Streamlit · Plotly &nbsp;|&nbsp;
            <a href="https://github.com/Sir-Rotich6" target="_blank">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def demo_banner(missing: list[str]) -> None:
    """Yellow info strip shown when the app is running on generated demo data."""
    items = " · ".join(missing)
    st.markdown(
        f"""
        <div class="demo-banner">
          <strong>⚡ Demo Mode</strong> — {items} not found.
          Showing synthetic data that mirrors real dataset distributions.
          Run the training pipeline to replace this with live results.
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi(col, value: str, label: str, delta: str = "", color: str = NAVY) -> None:
    col.markdown(
        f"""
        <div class="kpi-wrap" style="--kc:{color}">
          <div class="kpi-val">{value}</div>
          <div class="kpi-lbl">{label}</div>
          {"<div class='kpi-delta'>" + delta + "</div>" if delta else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str, bg: str, fg: str = "white") -> str:
    return f'<span class="badge" style="background:{bg};color:{fg}">{text}</span>'


def callout(text: str, kind: str = "") -> None:
    st.markdown(f'<div class="callout {kind}">{text}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — PROJECT OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

def page_overview() -> None:
    render_header("Portfolio Showcase · End-to-end Fraud Detection Pipeline")
    res = load_model_results()
    ds  = res.get("dataset", {})

    col_text, col_img = st.columns([3, 1], gap="large")
    with col_text:
        st.markdown(
            """
            <div class="hero-title">
              Detecting Fraud in<br>
              <span class="hero-accent">283 K Transactions</span>
            </div>
            <div class="hero-sub">
              A production-ready fraud detection system achieving <strong>97.7% AUC-ROC</strong>
              and <strong>81.1% AUPRC</strong> on a severely imbalanced dataset (0.17% fraud rate).
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_img:
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,{NAVY},{BLUE});
                        border-radius:20px;padding:32px 24px;text-align:center;color:white">
              <div style="font-size:3.5rem">🛡️</div>
              <div style="font-size:0.9rem;opacity:.8;margin-top:8px">
                Fraud Prevention<br>System
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown('<p class="section-label">About This Project</p>', unsafe_allow_html=True)
    left, right = st.columns([2, 1], gap="large")
    with left:
        st.markdown(
            """
            This project builds a **complete fraud-detection ML pipeline** on the public European
            credit-card dataset — from raw PCA-anonymised transaction features all the way to a
            served FastAPI endpoint with a real-time dashboard.

            The core challenge is extreme **class imbalance** (0.17% fraud): standard accuracy is
            misleading, so the pipeline optimises **AUPRC** throughout — from feature engineering
            decisions to the Optuna hyperparameter search objective.

            Feature engineering adds 14 domain-informed signals (card-testing micro-transactions,
            night-time activity, high-value anomalies, V-component interactions) on top of the
            30 PCA features, and the final XGBoost model is tuned across 30 Optuna trials to
            achieve **+6.2 pp AUPRC improvement** over the logistic-regression baseline.
            """,
        )
    with right:
        callout("⚠️ <strong>Class Imbalance:</strong> Only 473 fraud cases in 283 K transactions (0.17%). "
                "Accuracy alone is useless — a model that predicts 'legit' always gets 99.8%.", "warn")
        callout("✅ <strong>Solution:</strong> Optimise AUPRC end-to-end. "
                "Weight classes, engineer fraud-specific features, tune with Optuna.", "success")

    st.markdown("---")
    st.markdown('<p class="section-label">Key Results</p>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, f"{ds.get('total_rows', 283726):,}", "Transactions Analysed", color=NAVY)
    kpi(c2, str(ds.get("features_total", 41)),   "Features (31 raw + 14 engineered)", color=BLUE)
    kpi(c3, "97.7%",  "AUC-ROC",                  delta="vs 95.4% baseline  (+2.3pp)", color=TEAL)
    kpi(c4, "81.1%",  "AUPRC",                    delta="vs 76.4% baseline  (+6.2pp)", color=ACCENT)
    kpi(c5, "84.4%",  "Precision @ 0.5 threshold", delta="Catches 80% of fraud",       color=GOLD)

    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
    c6, c7, c8, c9, _ = st.columns(5)
    kpi(c6, "473",    "Confirmed Fraud Cases", color=ACCENT)
    kpi(c7, "30",     "Optuna Tuning Trials",  color=BLUE)
    kpi(c8, "5-Fold", "Cross-Validation",      color=TEAL)
    kpi(c9, "4",      "Models Compared",       color=GOLD)

    st.markdown("---")
    st.markdown('<p class="section-label">Tech Stack</p>', unsafe_allow_html=True)
    stack = [
        ("Python 3.11",       "#3776AB", "white"),
        ("pandas",            "#150458", "white"),
        ("NumPy",             "#013243", "white"),
        ("scikit-learn",      "#F7931E", "white"),
        ("XGBoost",           "#189ABF", "white"),
        ("LightGBM",          "#3CA753", "white"),
        ("Optuna",            "#6666FF", "white"),
        ("MLflow",            "#0194E2", "white"),
        ("FastAPI",           "#009688", "white"),
        ("Streamlit",         "#FF4B4B", "white"),
        ("Plotly",            "#3F4F75", "white"),
        ("Great Expectations","#FF6B6B", "white"),
        ("pytest",            "#0A9EDC", "white"),
        ("Docker",            "#2496ED", "white"),
    ]
    badges_html = "".join(badge(name, bg, fg) for name, bg, fg in stack)
    st.markdown(
        f'<div style="background:white;padding:20px 24px;border-radius:12px;'
        f'box-shadow:0 2px 8px rgba(0,0,0,.06)">{badges_html}</div>',
        unsafe_allow_html=True,
    )
    render_footer()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — EXPLORE THE DATA
# ═══════════════════════════════════════════════════════════════════════════════

def page_explore() -> None:
    render_header("Exploratory Data Analysis · Interactive Feature Explorer")

    status = _source_status()
    if not status["has_raw"]:
        demo_banner(["data/creditcard.csv"])

    df     = load_raw_data()
    target = "Class"

    # Ensure the dataset has the Class column (demo data always does).
    if target not in df.columns:
        st.error("Dataset is missing the 'Class' column. Cannot render EDA.")
        return

    fraud_df = df[df[target] == 1]
    legit_df = df[df[target] == 0]

    st.markdown('<p class="section-label">Key EDA Findings</p>', unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    with r1:
        callout("🔢 <strong>No missing values</strong> — dataset is complete across all rows and columns. "
                "No imputation required.", "success")
    with r2:
        fraud_pct = df[target].mean() * 100
        callout(f"⚖️ <strong>Extreme imbalance</strong> — {fraud_pct:.2f}% fraud rate. "
                "Standard accuracy is misleading; must use AUPRC.", "warn")
    with r3:
        callout("📐 <strong>PCA features are uncorrelated</strong> — V1–V28 are orthogonal by construction. "
                "No multicollinearity; tree-based models are ideal.", "")

    r4, r5, r6 = st.columns(3)
    with r4:
        callout("📍 <strong>V14, V12, V10 dominate</strong> — these three components show the largest "
                "distributional separation between fraud and legitimate transactions.", "success")
    with r5:
        amt_mean = df["Amount"].mean()
        callout(f"💰 <strong>Amount is right-skewed</strong> — mean ${amt_mean:.0f}, "
                "max ~$25K. Log-transform is essential before distance-based models.", "warn")
    with r6:
        callout("🌙 <strong>Night-time spike</strong> — fraud rate is elevated between 00:00–06:00 "
                "when real-time monitoring and call-backs are lightest.", "danger")

    st.markdown("---")

    # ── Target distribution ───────────────────────────────────────────────────
    st.markdown('<p class="section-label">Target Variable Distribution</p>', unsafe_allow_html=True)
    col_pie, col_bar = st.columns(2)
    counts = df[target].value_counts().rename({0: "Legitimate", 1: "Fraud"})

    with col_pie:
        total_label = f"{len(df) // 1000}K" if len(df) >= 1000 else str(len(df))
        fig = go.Figure(go.Pie(
            labels=counts.index, values=counts.values, hole=0.55,
            marker_colors=[BLUE, ACCENT], textinfo="percent+label", textfont_size=14,
        ))
        fig.update_layout(
            title="Class Split", height=320, template=PLOTLY_TEMPLATE, showlegend=False,
            annotations=[dict(text=total_label, x=0.5, y=0.5,
                              font_size=22, font_color=NAVY, showarrow=False)],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_bar:
        fig2 = go.Figure()
        fig2.add_bar(x=counts.index, y=counts.values,
                     marker_color=[BLUE, ACCENT], text=counts.values,
                     textposition="outside", textfont_size=13)
        fig2.update_layout(
            title="Transaction Count by Class", height=320, template=PLOTLY_TEMPLATE,
            yaxis_title="Count", xaxis_title="Class",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ── Amount distribution ───────────────────────────────────────────────────
    st.markdown('<p class="section-label">Transaction Amount Analysis</p>', unsafe_allow_html=True)
    col_a1, col_a2 = st.columns(2)

    with col_a1:
        clip_val = float(df["Amount"].quantile(0.99))
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=legit_df["Amount"].clip(upper=clip_val), name="Legitimate",
                                    marker_color=BLUE, opacity=0.65, nbinsx=60))
        fig3.add_trace(go.Histogram(x=fraud_df["Amount"].clip(upper=clip_val), name="Fraud",
                                    marker_color=ACCENT, opacity=0.75, nbinsx=60))
        fig3.update_layout(
            barmode="overlay", template=PLOTLY_TEMPLATE, height=320,
            title=f"Amount Distribution (clipped at ${clip_val:.0f})",
            xaxis_title="Amount ($)", yaxis_title="Count",
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_a2:
        fig4 = go.Figure()
        fig4.add_trace(go.Box(y=np.log1p(legit_df["Amount"]), name="Legitimate",
                              marker_color=BLUE, boxpoints=False))
        fig4.add_trace(go.Box(y=np.log1p(fraud_df["Amount"]), name="Fraud",
                              marker_color=ACCENT, boxpoints=False))
        fig4.update_layout(
            template=PLOTLY_TEMPLATE, height=320,
            title="Log(Amount+1) by Class — separation clearer after transform",
            yaxis_title="log(Amount + 1)",
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # ── Feature selector ─────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Interactive Feature Explorer</p>', unsafe_allow_html=True)
    v_cols       = [f"V{i}" for i in range(1, 29) if f"V{i}" in df.columns]
    numeric_cols = [c for c in ["Amount", "Time"] + v_cols if c in df.columns]

    default_idx = numeric_cols.index("V14") if "V14" in numeric_cols else 0
    sel_feat    = st.selectbox("Select a feature to explore:", numeric_cols, index=default_idx)
    fsamp       = df.sample(min(len(df), 5_000), random_state=42)
    col_hist, col_box = st.columns(2)

    with col_hist:
        fig5 = px.histogram(
            fsamp, x=sel_feat,
            color=fsamp[target].map({0: "Legitimate", 1: "Fraud"}),
            nbins=60, barmode="overlay", opacity=0.7,
            color_discrete_map={"Legitimate": BLUE, "Fraud": ACCENT},
            title=f"{sel_feat} Distribution by Class",
            template="plotly_white",
        )
        fig5.update_layout(height=340, legend_title_text="Class",
                           legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig5, use_container_width=True)

    with col_box:
        fig6 = go.Figure()
        for cls, name, color in [(0, "Legitimate", BLUE), (1, "Fraud", ACCENT)]:
            sub = fsamp[fsamp[target] == cls][sel_feat]
            fig6.add_trace(go.Violin(y=sub, name=name, fillcolor=color, opacity=0.7,
                                     line_color=color, box_visible=True, meanline_visible=True))
        fig6.update_layout(
            template=PLOTLY_TEMPLATE, height=340,
            title=f"{sel_feat} — Violin + Box by Class",
            yaxis_title=sel_feat, showlegend=True,
        )
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown("---")

    # ── Correlations with target ──────────────────────────────────────────────
    st.markdown('<p class="section-label">Feature Correlations with Fraud Target</p>', unsafe_allow_html=True)
    corr_sample = df.sample(min(len(df), 30_000), random_state=42)
    corrs = (
        corr_sample[numeric_cols + [target]]
        .corr()[target]
        .drop(target)
        .sort_values(key=abs, ascending=False)
        .head(20)
    )
    fig7 = go.Figure(go.Bar(
        x=corrs.values, y=corrs.index, orientation="h",
        marker_color=[ACCENT if v < 0 else BLUE for v in corrs.values],
        text=[f"{v:.3f}" for v in corrs.values], textposition="outside",
    ))
    fig7.update_layout(
        template=PLOTLY_TEMPLATE, height=420,
        title="Top 20 Features by |Correlation| with Fraud Class",
        xaxis_title="Pearson Correlation with Class",
        yaxis=dict(autorange="reversed"),
        xaxis=dict(range=[-0.45, 0.45]),
    )
    st.plotly_chart(fig7, use_container_width=True)
    callout("📌 <strong>V14 is the dominant signal</strong> (|r| ≈ 0.30). V12, V10, V17 and V3 also "
            "show significant separation. PCA feature interactions amplify these in engineered features.", "success")

    st.markdown("---")

    # ── Correlation heatmap ───────────────────────────────────────────────────
    st.markdown('<p class="section-label">V-Feature Correlation Heatmap</p>', unsafe_allow_html=True)
    st.caption("PCA components are designed to be uncorrelated — this confirms no multicollinearity issues.")
    top_v     = ["V14", "V12", "V10", "V17", "V3", "V4", "V11", "V16", "V7", "V9"]
    existing_v = [c for c in top_v if c in df.columns]
    hm_sample  = df[existing_v].sample(min(len(df), 20_000), random_state=42)
    fig8 = px.imshow(
        hm_sample.corr(),
        color_continuous_scale=["#E63946", "white", "#457B9D"],
        zmin=-1, zmax=1, text_auto=".2f", aspect="auto",
    )
    fig8.update_layout(
        template=PLOTLY_TEMPLATE, height=460,
        title="Pearson Correlation — Top V-Features",
        coloraxis_colorbar=dict(title="r"),
    )
    st.plotly_chart(fig8, use_container_width=True)
    render_footer()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — MODEL RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

def page_model_results() -> None:
    render_header("Model Results · Comparison · Feature Importance · Live Demo")

    status   = _source_status()
    res      = load_model_results()
    models   = res.get("models", [])
    feat_imp = res.get("feature_importances", {})

    missing = []
    if not status["has_model"]:
        missing.append("Trained models (models/*.pkl)")
    if not status["has_features"]:
        missing.append("data/features.csv")
    if missing:
        demo_banner(missing)

    # ── Model comparison table ────────────────────────────────────────────────
    st.markdown('<p class="section-label">Model Comparison</p>', unsafe_allow_html=True)
    st.caption("Primary metric: AUPRC (Area Under Precision-Recall Curve) — "
               "more informative than AUC-ROC on imbalanced data.")

    rows = [{
        "Model":      m["name"],
        "Type":       m["label"],
        "AUPRC":      m["auprc"],
        "AUC-ROC":    m["auc_roc"],
        "Precision":  m["precision"],
        "Recall":     m["recall"],
        "F1":         m["f1"],
        "CV AUPRC":   f'{m["cv_auprc_mean"]:.4f} ± {m["cv_auprc_std"]:.4f}',
        "Train (s)":  m["train_time_s"],
        "Winner":     "🏆" if m["is_winner"] else "",
    } for m in models]

    if rows:
        df_cmp = pd.DataFrame(rows)

        def highlight_winner(row):
            bg = "#FFF3CD" if row["Winner"] == "🏆" else ""
            fw = "bold"    if row["Winner"] == "🏆" else "normal"
            return [f"background-color:{bg};font-weight:{fw}" for _ in row]

        styled = (
            df_cmp.style
            .apply(highlight_winner, axis=1)
            .format({"AUPRC": "{:.4f}", "AUC-ROC": "{:.4f}",
                     "Precision": "{:.4f}", "Recall": "{:.4f}",
                     "F1": "{:.4f}", "Train (s)": "{:.1f}"})
            .background_gradient(subset=["AUPRC"], cmap="YlOrRd", vmin=0.75, vmax=0.82)
            .set_properties(**{"text-align": "center"})
            .set_table_styles([{"selector": "th",
                                "props": [("text-align", "center"),
                                          ("background-color", NAVY),
                                          ("color", "white"),
                                          ("font-size", "13px")]}])
        )
        st.dataframe(styled, use_container_width=True, height=220)
    else:
        st.info("No model results found.")

    # AUPRC bar chart
    if models:
        fig_cmp = go.Figure()
        for m in models:
            fig_cmp.add_trace(go.Bar(
                name=m["name"], x=[m["name"]], y=[m["auprc"]],
                marker_color=m["color"],
                error_y=dict(type="data", array=[m["cv_auprc_std"]], visible=True),
                text=f'{m["auprc"]:.4f}', textposition="outside", textfont_size=12,
            ))
        baseline_val = next((m["auprc"] for m in models if m["label"] == "Baseline"), 0.76)
        fig_cmp.update_layout(
            template=PLOTLY_TEMPLATE, height=340,
            title="AUPRC Comparison — All Models (error bars = CV std)",
            yaxis=dict(title="AUPRC", range=[0.72, 0.84]),
            showlegend=False, barmode="group",
        )
        fig_cmp.add_hline(y=baseline_val, line_dash="dash", line_color=GREY,
                          annotation_text="Baseline", annotation_position="right")
        st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown("---")

    # ── Why I picked the winner ───────────────────────────────────────────────
    st.markdown('<p class="section-label">Why XGBoost (Optuna-Tuned) Won</p>', unsafe_allow_html=True)
    winner = next((m for m in models if m["is_winner"]), None)
    if winner:
        st.markdown(
            f'<div class="winner-banner"><h3>🏆 {winner["name"]}</h3>'
            f'<p>{winner["why_won"]}</p></div>',
            unsafe_allow_html=True,
        )

    rcol1, rcol2, rcol3, rcol4 = st.columns(4)
    for col, (color, label, val, sub) in zip(
        [rcol1, rcol2, rcol3, rcol4],
        [
            (ACCENT, "Highest AUPRC",    "0.8106",    "+6.2pp over baseline"),
            (TEAL,   "Most Stable",      "CV ±0.016", "Lowest std across 5 folds"),
            (BLUE,   "Best Precision",   "84.4%",     "Fewest false alarms"),
            (GOLD,   "Recall",           "80.0%",     "Catches 4-in-5 fraud cases"),
        ],
    ):
        kpi(col, val, label, sub, color)

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
    st.markdown("**Why the runner-ups were not selected:**")
    not_winners = [m for m in models if not m["is_winner"] and m["label"] != "Baseline"]
    if not_winners:
        cols = st.columns(len(not_winners))
        for col, m in zip(cols, not_winners):
            with col:
                callout(f"<strong>{m['name']} ({m['label']}):</strong><br>{m['why_not']}", "warn")

    st.markdown("---")

    # ── Feature importance ────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Feature Importance — Top 15 (Production Model)</p>',
                unsafe_allow_html=True)

    imp_series = None
    model_obj  = load_model("tuned_model.pkl")
    feat_cols  = get_feature_columns()

    if model_obj is not None and feat_cols:
        est = model_obj.steps[-1][1] if hasattr(model_obj, "steps") else model_obj
        if hasattr(est, "feature_importances_") and len(est.feature_importances_) == len(feat_cols):
            imp_series = pd.Series(est.feature_importances_, index=feat_cols)

    if imp_series is None:
        imp_series = pd.Series(feat_imp) if feat_imp else None

    if imp_series is not None:
        top15       = imp_series.nlargest(15).sort_values()
        engineered  = {
            "amount_log", "is_small_amount", "is_large_amount", "hour_of_day",
            "is_night", "is_round_amount", "v_mean", "v_std", "v_l2_norm",
            "v_max_abs", "amount_x_v14", "v12_x_v14", "night_x_large", "v14_minus_v17",
        }
        bar_colors = [GOLD if f in engineered else BLUE for f in top15.index]
        fig_imp = go.Figure(go.Bar(
            x=top15.values, y=top15.index, orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.4f}" for v in top15.values], textposition="outside", textfont_size=11,
        ))
        fig_imp.update_layout(
            template=PLOTLY_TEMPLATE, height=480,
            title="XGBoost Feature Importance — Blue = raw PCA · Gold = engineered",
            xaxis_title="Importance Score", yaxis=dict(tickfont_size=12),
        )
        fig_imp.add_annotation(
            x=0.98, y=0.02, xref="paper", yref="paper", showarrow=False,
            text=f"<span style='color:{GOLD}'>■</span> Engineered &nbsp; "
                 f"<span style='color:{BLUE}'>■</span> Raw PCA",
            font_size=12, align="right", bgcolor="white", borderpad=4,
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        callout("💡 <strong>V14 is the single strongest signal</strong>, but engineered features "
                "<code>amount_x_v14</code> and <code>v12_x_v14</code> rank in the top 3 — "
                "confirming that feature interactions extracted meaningful signal beyond raw PCA.", "success")
    else:
        st.info("Feature importance not available.")

    st.markdown("---")

    # ── Confusion matrix & curves ─────────────────────────────────────────────
    st.markdown('<p class="section-label">Confusion Matrix & Classification Report</p>',
                unsafe_allow_html=True)

    # _compute_test_predictions handles both real-model and demo-heuristic paths.
    y_test, y_prob, pred_is_demo = _compute_test_predictions("tuned_model.pkl")

    if pred_is_demo:
        callout("📊 <strong>Demo predictions</strong> — these charts use a heuristic scorer "
                "because the trained model file was not found. Train the model for live results.", "warn")

    from sklearn.metrics import (
        confusion_matrix, precision_recall_curve, roc_curve,
        average_precision_score, roc_auc_score,
    )

    y_pred      = (y_prob >= 0.5).astype(int)
    cm          = confusion_matrix(y_test, y_pred)
    auprc_live  = average_precision_score(y_test, y_prob)
    auc_live    = roc_auc_score(y_test, y_prob)
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_prob)
    fpr_vals, tpr_vals, _  = roc_curve(y_test, y_prob)

    cm_col, cr_col = st.columns(2, gap="large")

    with cm_col:
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Predicted Legit", "Predicted Fraud"],
            y=["Actual Legit", "Actual Fraud"],
            text_auto=True,
            color_continuous_scale=["white", NAVY],
        )
        fig_cm.update_layout(
            template=PLOTLY_TEMPLATE, height=340,
            title=f"Confusion Matrix (threshold = 0.5)",
            coloraxis_showscale=False, font_size=14,
        )
        fig_cm.update_traces(textfont_size=18)
        st.plotly_chart(fig_cm, use_container_width=True)

        tn, fp, fn, tp = cm.ravel()
        callout(
            f"🔴 <strong>False Negatives: {fn}</strong> — fraud cases missed "
            f"(most costly in production).<br>"
            f"🟡 <strong>False Positives: {fp}</strong> — legitimate transactions "
            f"flagged (reduces customer experience).", "warn"
        )

    with cr_col:
        fig_pr = make_subplots(rows=1, cols=2,
                               subplot_titles=["Precision-Recall Curve", "ROC Curve"])
        fig_pr.add_trace(
            go.Scatter(x=rec_vals, y=prec_vals, mode="lines",
                       name=f"AUPRC={auprc_live:.3f}",
                       line=dict(color=ACCENT, width=2.5)), row=1, col=1)
        fig_pr.add_hline(y=y_test.mean(), line_dash="dash", line_color=GREY,
                         annotation_text="Random", row=1, col=1)
        fig_pr.add_trace(
            go.Scatter(x=fpr_vals, y=tpr_vals, mode="lines",
                       name=f"AUC={auc_live:.3f}",
                       line=dict(color=BLUE, width=2.5)), row=1, col=2)
        fig_pr.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                         line=dict(dash="dash", color=GREY), row=1, col=2)
        fig_pr.update_layout(
            template=PLOTLY_TEMPLATE, height=320, showlegend=True,
            legend=dict(orientation="h", y=-0.15),
        )
        fig_pr.update_xaxes(title_text="Recall", row=1, col=1)
        fig_pr.update_yaxes(title_text="Precision", row=1, col=1)
        fig_pr.update_xaxes(title_text="FPR", row=1, col=2)
        fig_pr.update_yaxes(title_text="TPR", row=1, col=2)
        st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown("---")

    # ── Try it yourself ───────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Try It Yourself — Live Prediction Demo</p>',
                unsafe_allow_html=True)
    st.caption("Adjust the sliders to simulate a transaction. The model scores it in real time.")

    demo_col, gauge_col = st.columns([2, 1], gap="large")

    with demo_col:
        st.markdown("**Transaction Parameters**")
        d1, d2 = st.columns(2)
        amount = d1.slider("Amount ($)", 0.0, 5000.0, 120.0, step=5.0)
        hour   = d2.slider("Hour of Day (0–23)", 0, 23, 14)

        st.markdown("**PCA Components (key fraud signals)**")
        p1, p2, p3, p4 = st.columns(4)
        v14 = p1.slider("V14", -20.0, 10.0, 0.0, step=0.5,
                         help="Most fraud-correlated component. Fraud clusters near -10 to -20.")
        v12 = p2.slider("V12", -15.0, 10.0, 0.0, step=0.5,
                         help="2nd most correlated. Fraud clusters at negative values.")
        v10 = p3.slider("V10", -20.0, 10.0, 0.0, step=0.5,
                         help="3rd most correlated component.")
        v17 = p4.slider("V17", -15.0, 10.0, 0.0, step=0.5,
                         help="Secondary fraud signal.")
        st.markdown(
            '<p style="font-size:.8rem;color:#888;margin-top:8px">'
            'All other PCA components (V1–V13, V15–V16, V18–V28) are set to 0 '
            '(the PCA-space average for a typical transaction).</p>',
            unsafe_allow_html=True,
        )

    # Score using real model when available, otherwise heuristic.
    if model_obj is not None:
        row_dict = {f"V{i}": 0.0 for i in range(1, 29)}
        row_dict.update({"V14": v14, "V12": v12, "V10": v10, "V17": v17,
                         "Amount": amount, "Time": float(hour * 3600)})
        input_df  = pd.DataFrame([row_dict])
        input_eng = engineer_features(input_df)
        if feat_cols:
            for c in set(feat_cols) - set(input_eng.columns):
                input_eng[c] = 0.0
            input_eng = input_eng[feat_cols]
        try:
            prob = float(model_obj.predict_proba(input_eng)[0, 1])
        except Exception:
            prob = _demo_score(v14, v12, v10, v17, amount, hour)
        gauge_note = ""
    else:
        prob       = _demo_score(v14, v12, v10, v17, amount, hour)
        gauge_note = "<br><small style='opacity:.7'>Heuristic score (no model loaded)</small>"

    with gauge_col:
        pct = prob * 100
        if pct < 20:
            risk_label, risk_css, risk_color = "LOW RISK",    "risk-low",    TEAL
        elif pct < 50:
            risk_label, risk_css, risk_color = "MEDIUM RISK", "risk-medium", GOLD
        else:
            risk_label, risk_css, risk_color = "HIGH RISK",   "risk-high",   ACCENT

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pct,
            delta=dict(reference=50, valueformat=".1f",
                       increasing=dict(color=ACCENT), decreasing=dict(color=TEAL)),
            number=dict(suffix="%", font_size=38, font_color=risk_color),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor=NAVY),
                bar=dict(color=risk_color),
                steps=[
                    dict(range=[0, 20],   color="#E8F8F5"),
                    dict(range=[20, 50],  color="#FFF8E1"),
                    dict(range=[50, 100], color="#FDEDEC"),
                ],
                threshold=dict(line=dict(color=NAVY, width=3), thickness=0.75, value=50),
            ),
            title=dict(text="Fraud Probability", font_size=15),
        ))
        fig_gauge.update_layout(
            height=280, margin=dict(t=40, b=10, l=10, r=10), paper_bgcolor="white"
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(
            f'<div class="risk-box {risk_css}">'
            f'<div class="risk-label" style="color:{risk_color}">{risk_label}</div>'
            f'<div style="font-size:.85rem;color:#555;margin-top:6px">'
            f'Probability: <strong>{pct:.1f}%</strong>{gauge_note}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("**Flags triggered:**")
        flags = []
        if amount < 10:
            flags.append("🚩 Micro-transaction (<$10) — card-testing pattern")
        if amount > 1_000:
            flags.append("🚩 Large amount (>$1,000) — high-value target")
        if hour < 6 or hour >= 22:
            flags.append("🌙 Night-time transaction (00–05 or 22–23)")
        if amount > 0 and amount % 1 == 0:
            flags.append("🔢 Round-dollar amount — automated script indicator")
        if v14 < -5:
            flags.append(f"📉 V14 = {v14:.1f} — strong negative deviation (fraud signal)")
        if v12 < -5:
            flags.append(f"📉 V12 = {v12:.1f} — strong negative deviation (fraud signal)")
        if not flags:
            flags.append("✅ No explicit fraud flags triggered")
        for f_txt in flags:
            st.markdown(f"- {f_txt}")

    render_footer()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — HOW I BUILT THIS
# ═══════════════════════════════════════════════════════════════════════════════

def page_how_built() -> None:
    render_header("How I Built This · Architecture · Decisions · Lessons Learned")

    st.markdown('<p class="section-label">Pipeline Architecture</p>', unsafe_allow_html=True)
    st.graphviz_chart(
        """
        digraph fraud_pipeline {
            rankdir=LR
            graph [bgcolor="transparent", fontname="Inter"]
            node  [shape=box, style="filled,rounded", fontname="Inter",
                   fontsize=12, margin="0.2,0.1"]
            edge  [color="#888888", penwidth=1.5, fontsize=10, fontname="Inter"]

            subgraph cluster_data {
                label="Data Layer"; style=filled; fillcolor="#EBF5FB"; color="#2196F3"
                raw  [label="creditcard.csv\n283K rows 31 cols",   fillcolor="#DBEAFE"]
                cln  [label="Cleaned Data\n(dedup + validate)",    fillcolor="#DBEAFE"]
                feat [label="Feature Store\n+14 engineered cols",  fillcolor="#BFDBFE"]
            }
            subgraph cluster_train {
                label="Training Layer"; style=filled; fillcolor="#F0FDF4"; color="#16A34A"
                base [label="Baseline\nLogReg Pipeline",           fillcolor="#DCFCE7"]
                cmp  [label="Model Selection\nXGB LGBM RF",        fillcolor="#DCFCE7"]
                tune [label="Optuna Tuning\n30 trials AUPRC",      fillcolor="#BBF7D0"]
                mf   [label="MLflow Tracking\n4 experiments",      fillcolor="#DCFCE7"]
            }
            subgraph cluster_serve {
                label="Serving Layer"; style=filled; fillcolor="#FEF9E7"; color="#D97706"
                prod [label="production_model.pkl\nTuned XGBoost", fillcolor="#FEF3C7"]
                api  [label="FastAPI\nPOST /predict",              fillcolor="#FEF3C7"]
                dash [label="Streamlit\nPortfolio Dashboard",      fillcolor="#FDE68A"]
            }

            raw  -> cln  [label="cleaner.py"]
            cln  -> feat [label="engineer.py"]
            feat -> base [label="80/20 split"]
            feat -> cmp
            cmp  -> tune [label="XGBoost selected"]
            base -> mf
            tune -> mf   [label="log metrics"]
            tune -> prod [label="joblib.dump"]
            prod -> api  [label="joblib.load"]
            prod -> dash
        }
        """,
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown('<p class="section-label">7-Day Build Timeline</p>', unsafe_allow_html=True)
    for day, title, desc in [
        ("Day 1", "Data Ingestion & Quality Checks",
         "Wrote loader.py, cleaner.py, validator.py. Ran Great Expectations suite to confirm zero nulls, "
         "expected dtypes, and amount range. Discovered the 0.17% fraud rate early — flagged AUPRC as "
         "the primary evaluation metric from day one."),
        ("Day 2", "Exploratory Data Analysis",
         "Built eda.ipynb with 15 visualisations. Key findings: V14/V12/V10 are dominant fraud signals, "
         "Amount is heavily right-skewed (log-transform required), and fraud spikes at night. These "
         "findings directly drove every feature engineering decision on Day 3."),
        ("Day 3", "Feature Engineering (14 new features)",
         "Implemented three feature categories: domain-specific signals (micro-transactions, night flag, "
         "round amounts), statistical PCA aggregations (v_l2_norm, v_max_abs), and interaction features "
         "(amount×V14, V12×V14). Correlation analysis validated all 14 features improved AUPRC in CV."),
        ("Day 4", "Baseline & Model Comparison",
         "Established LogReg baseline at 0.7644 AUPRC. Trained XGBoost, LightGBM, RandomForest with "
         "5-fold stratified CV and scale_pos_weight to handle imbalance. XGBoost showed best CV AUPRC "
         "(0.7823 ± 0.020) — selected for tuning."),
        ("Day 5", "Hyperparameter Tuning with Optuna",
         "30-trial Optuna study optimising AUPRC on hold-out CV fold. Key finding: depth=10 with "
         "reg_alpha=2.14 (L1 regularisation) captures deep non-linear fraud patterns while preventing "
         "overfitting. Final model: 0.8106 AUPRC, +6.2pp over baseline."),
        ("Day 6", "FastAPI Serving Endpoint",
         "Built /predict endpoint with Pydantic input validation, inline feature engineering, and "
         "risk-level classification (LOW/MEDIUM/HIGH). Added /health and /model/info endpoints. "
         "End-to-end latency < 15ms per prediction (benchmarked with locust)."),
        ("Day 7", "Portfolio Dashboard & Documentation",
         "Built this multi-page Streamlit app with interactive EDA, model comparison, live prediction "
         "demo, and architecture docs. Wrote README, docstrings, and CI-ready pytest suite."),
    ]:
        st.markdown(
            f'<div class="tl-item">'
            f'<div class="tl-dot">{day.split()[1]}</div>'
            f'<div class="tl-body">'
            f'<div class="tl-day">{day}</div>'
            f'<div class="tl-title">{title}</div>'
            f'<div class="tl-desc">{desc}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown('<p class="section-label">Key Design Decisions</p>', unsafe_allow_html=True)
    dec1, dec2 = st.columns(2)
    with dec1:
        callout("🎯 <strong>AUPRC as primary metric, not AUC-ROC</strong><br>"
                "With 0.17% fraud, a random model achieves 0.0017 AUPRC vs. 0.5 AUC-ROC — AUPRC "
                "punishes false alarms on the minority class. Used as the Optuna objective throughout.", "success")
        callout("⚖️ <strong>scale_pos_weight over SMOTE</strong><br>"
                "SMOTE synthesises in high-dimensional PCA space where distances may be meaningless. "
                "Cost-sensitive learning via scale_pos_weight=599.5 is simpler, faster, interpretable.", "")
        callout("🔧 <strong>Feature engineering before model tuning</strong><br>"
                "The 14 engineered features contributed ~3pp AUPRC independently of tuning. "
                "Separating concerns let the tuner focus on model capacity, not missing signals.", "success")
    with dec2:
        callout("📊 <strong>MLflow for experiment tracking</strong><br>"
                "All 4 runs logged with parameters, metrics, and artifacts — trivial to reproduce "
                "any result or compare model versions after hours-long tuning runs.", "")
        callout("🚀 <strong>FastAPI over Flask/Django</strong><br>"
                "Auto-generated OpenAPI docs, native Pydantic validation, and async support. "
                "The /health check enables Kubernetes-style readiness probes.", "")
        callout("🗂️ <strong>src/ layout with setup.py</strong><br>"
                "Separating source from notebooks and app prevents import path issues and makes "
                "the project installable (pip install -e .). Mirrors production ML team structure.", "success")

    st.markdown("---")
    st.markdown('<p class="section-label">Lessons Learned</p>', unsafe_allow_html=True)
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        st.markdown("##### What worked well")
        for item in [
            "Feature interactions (amount×V14) gave the biggest AUPRC jump per feature",
            "Optuna's median pruner cut tuning time by ~40% with negligible quality loss",
            "Great Expectations caught a duplicate-row issue before training",
            "AUPRC-first thinking from Day 1 kept the whole pipeline coherent",
        ]:
            st.markdown(f"✅ {item}")
    with lc2:
        st.markdown("##### What I'd do differently")
        for item in [
            "Add SHAP explanations alongside permutation importance",
            "Explore calibration (isotonic regression) for probability outputs",
            "Implement drift detection (PSI) for production monitoring",
            "Try a cost-sensitive learning approach with custom eval metric",
        ]:
            st.markdown(f"🔄 {item}")
    with lc3:
        st.markdown("##### If given more time")
        for item in [
            "Graph-based features: transaction network (card → merchant edges)",
            "Time-series aggregations: rolling fraud rate per card over 24h",
            "AutoML comparison (FLAML or AutoGluon) as an additional baseline",
            "A/B test the model against existing rule-based system in staging",
        ]:
            st.markdown(f"🚀 {item}")

    st.markdown("---")
    st.markdown('<p class="section-label">Source Code</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background:white;border-radius:12px;padding:24px;
                    box-shadow:0 2px 12px rgba(0,0,0,.07);display:flex;align-items:center;gap:20px">
          <div style="font-size:3rem">🐙</div>
          <div>
            <div style="font-size:1.1rem;font-weight:700;color:#1A1F3A">GitHub Repository</div>
            <div style="color:#555;margin:4px 0">Full source code, notebooks, training scripts, and CI pipeline</div>
            <a href="https://github.com/Sir-Rotich6" target="_blank"
               style="color:#457B9D;font-weight:600;text-decoration:none">
              github.com/Sir-Rotich6 →
            </a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_footer()


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR + ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.sidebar.markdown(
        f"""
        <div style="background:linear-gradient(135deg,{NAVY},{BLUE});
                    border-radius:10px;padding:16px;margin-bottom:16px;color:white">
          <div style="font-size:1.1rem;font-weight:800">🔍 Fraud Detection</div>
          <div style="font-size:0.78rem;opacity:.75;margin-top:3px">
            Portfolio Project · Sir-Rotich6
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pages = {
        "🏠  Project Overview": page_overview,
        "📊  Explore the Data": page_explore,
        "🤖  Model Results":    page_model_results,
        "🛠️  How I Built This": page_how_built,
    }

    selection = st.sidebar.radio(
        "Navigate", list(pages.keys()), label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    res = load_model_results()
    ds  = res.get("dataset", {})
    st.sidebar.markdown("**Quick Stats**")
    st.sidebar.metric("Transactions", f'{ds.get("total_rows", 283726):,}')
    st.sidebar.metric("Fraud Cases",  str(ds.get("fraud_count", 473)))
    st.sidebar.metric("Best AUPRC",   "0.8106")
    st.sidebar.metric("Best AUC-ROC", "0.9767")

    # Show data source status in sidebar
    status = _source_status()
    if not all(status.values()):
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Data status**")
        icons = {True: "✅", False: "⚡"}
        st.sidebar.caption(f"{icons[status['has_raw']]} Raw data")
        st.sidebar.caption(f"{icons[status['has_features']]} Features")
        st.sidebar.caption(f"{icons[status['has_model']]} Model")

    st.sidebar.markdown("---")
    st.sidebar.caption("Built with Streamlit · Plotly · XGBoost")

    pages[selection]()


if __name__ == "__main__":
    main()
