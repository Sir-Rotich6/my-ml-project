import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pathlib import Path
from pydantic import BaseModel, Field

MODELS_DIR = Path(__file__).parents[1] / "models"
MODEL_PATH = MODELS_DIR / "production_model.pkl"

app = FastAPI(title="Fraud Detection API", version="1.0.0")

_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail=f"Model not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


class Transaction(BaseModel):
    Time: float = Field(..., description="Seconds elapsed since first transaction")
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float = Field(..., ge=0, description="Transaction amount in USD")


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_level: str


def engineer_features(tx: dict) -> pd.DataFrame:
    import math
    df = pd.DataFrame([tx])
    v_cols = [f"V{i}" for i in range(1, 29)]
    v = df[v_cols]

    df["is_small_amount"]  = (df["Amount"] < 10).astype(int)
    df["amount_log"]       = np.log1p(df["Amount"])
    df["is_large_amount"]  = (df["Amount"] > 1000).astype(int)
    df["hour_of_day"]      = (df["Time"] % 86400 / 3600).astype(int)
    df["is_night"]         = ((df["hour_of_day"] < 6) | (df["hour_of_day"] >= 22)).astype(int)
    df["is_round_amount"]  = ((df["Amount"] % 1 == 0) & (df["Amount"] > 0)).astype(int)
    df["v_mean"]           = v.mean(axis=1)
    df["v_max_abs"]        = v.abs().max(axis=1)
    df["amount_x_v14"]     = df["amount_log"] * df["V14"]
    df["v12_x_v14"]        = df["V12"] * df["V14"]
    df["v14_minus_v17"]    = df["V14"] - df["V17"]

    drop_cols = ["v_std", "v_l2_norm", "night_x_large"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    return df


def risk_label(prob: float) -> str:
    if prob >= 0.7:
        return "HIGH"
    if prob >= 0.3:
        return "MEDIUM"
    return "LOW"


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    model = get_model()
    features = engineer_features(transaction.model_dump())
    prob = float(model.predict_proba(features)[0, 1])
    return PredictionResponse(
        fraud_probability=round(prob, 4),
        is_fraud=prob >= 0.5,
        risk_level=risk_label(prob),
    )


@app.get("/model/info")
def model_info():
    model = get_model()
    return {
        "model_type": type(model).__name__,
        "model_path": str(MODEL_PATH),
    }
