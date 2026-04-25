import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, roc_curve

DATA_DIR = Path(__file__).parents[1] / "data"
MODELS_DIR = Path(__file__).parents[1] / "models"
TARGET_COL = "Class"

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("Credit Card Fraud Detection — Model Dashboard")


@st.cache_data
def load_data():
    path = DATA_DIR / "features.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df


@st.cache_resource
def load_model(name: str):
    path = MODELS_DIR / name
    if not path.exists():
        return None
    return joblib.load(path)


df = load_data()

if df is None:
    st.error("features.csv not found. Run the feature pipeline first.")
    st.stop()

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")
model_options = {
    "Baseline (LogReg)": "baseline.pkl",
    "XGBoost (untuned)": "xgboost.pkl",
    "Tuned XGBoost":     "tuned_model.pkl",
    "Production Model":  "production_model.pkl",
}
selected_label = st.sidebar.selectbox("Model", list(model_options.keys()))
model = load_model(model_options[selected_label])
threshold = st.sidebar.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)
test_frac = st.sidebar.slider("Test split fraction", 0.1, 0.4, 0.2, 0.05)

if model is None:
    st.warning(f"Model file `{model_options[selected_label]}` not found in models/.")
    st.stop()

# ── Train/test split (same seed as training) ───────────────────────────────
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X, y, test_size=test_frac, random_state=42, stratify=y)
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

# ── KPI row ────────────────────────────────────────────────────────────────
from sklearn.metrics import precision_score, recall_score, f1_score
auprc   = average_precision_score(y_test, y_prob)
auc_roc = roc_auc_score(y_test, y_prob)
prec    = precision_score(y_test, y_pred, zero_division=0)
rec     = recall_score(y_test, y_pred, zero_division=0)
f1      = f1_score(y_test, y_pred, zero_division=0)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("AUPRC",     f"{auprc:.3f}")
c2.metric("AUC-ROC",   f"{auc_roc:.3f}")
c3.metric("Precision", f"{prec:.3f}")
c4.metric("Recall",    f"{rec:.3f}")
c5.metric("F1",        f"{f1:.3f}")

st.divider()

# ── Charts row 1 ───────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Precision-Recall Curve")
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec_vals, y=prec_vals, mode="lines", name=f"AUPRC={auprc:.3f}"))
    fig.add_hline(y=y_test.mean(), line_dash="dash", line_color="grey", annotation_text="Random")
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=350)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={auc_roc:.3f}"))
    fig2.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
    fig2.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=350)
    st.plotly_chart(fig2, use_container_width=True)

# ── Charts row 2 ───────────────────────────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("Score Distribution by Class")
    plot_df = pd.DataFrame({"prob": y_prob, "label": y_test.map({0: "Legit", 1: "Fraud"}).values})
    fig3 = px.histogram(plot_df, x="prob", color="label", nbins=60, barmode="overlay",
                        opacity=0.7, color_discrete_map={"Legit": "steelblue", "Fraud": "tomato"},
                        labels={"prob": "Fraud Probability"})
    fig3.add_vline(x=threshold, line_dash="dash", annotation_text=f"threshold={threshold}")
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("Threshold vs Metrics")
    thresholds = np.arange(0.05, 0.96, 0.05)
    rows = []
    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        rows.append({
            "threshold": round(t, 2),
            "precision": precision_score(y_test, yp, zero_division=0),
            "recall":    recall_score(y_test, yp, zero_division=0),
            "f1":        f1_score(y_test, yp, zero_division=0),
        })
    df_t = pd.DataFrame(rows)
    fig4 = go.Figure()
    for col_name, color in [("precision", "blue"), ("recall", "red"), ("f1", "green")]:
        fig4.add_trace(go.Scatter(x=df_t["threshold"], y=df_t[col_name],
                                  mode="lines", name=col_name, line=dict(color=color)))
    fig4.add_vline(x=threshold, line_dash="dash", annotation_text=f"current={threshold}")
    fig4.update_layout(xaxis_title="Threshold", yaxis_title="Score", height=350)
    st.plotly_chart(fig4, use_container_width=True)

# ── Feature importance ─────────────────────────────────────────────────────
st.subheader("Feature Importance (top 20)")
estimator = model
if hasattr(model, "steps"):
    estimator = model.steps[-1][1]

if hasattr(estimator, "feature_importances_"):
    imp = pd.Series(estimator.feature_importances_, index=X_test.columns)
    top20 = imp.nlargest(20).sort_values()
    fig5 = px.bar(top20, orientation="h", labels={"value": "Importance", "index": "Feature"})
    fig5.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig5, use_container_width=True)
else:
    st.info("Feature importances not available for this model type.")

# ── Raw test predictions ───────────────────────────────────────────────────
with st.expander("Test set predictions (sample)"):
    sample = X_test.copy()
    sample["true_label"] = y_test.values
    sample["fraud_prob"] = y_prob
    sample["predicted"] = y_pred
    st.dataframe(sample.head(100), use_container_width=True)
