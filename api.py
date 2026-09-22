"""
Fraud Detection Project: FastAPI Real-Time Inference Service
Loads the finalized model (fraud_model.joblib, produced by shap_explainability.py)
and serves real-time fraud predictions with per-transaction SHAP explanations.

Usage:
    uvicorn api:app --reload

Then test with:
    http://127.0.0.1:8000/docs   (interactive Swagger UI)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import shap
import numpy as np
import pandas as pd

MODEL_PATH = "fraud_model.joblib"

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud scoring with SHAP-based explanations.",
    version="1.0.0",
)

# ------------------------------------------------------------
# Load model, threshold, and feature list once at startup
# ------------------------------------------------------------
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
THRESHOLD = artifact["threshold"]
FEATURE_COLS = artifact["feature_cols"]
explainer = shap.TreeExplainer(model)

print(f"Loaded model. Threshold = {THRESHOLD}. Expecting {len(FEATURE_COLS)} features.")


# ------------------------------------------------------------
# Request/response schemas
# ------------------------------------------------------------
class Transaction(BaseModel):
    """
    Must contain exactly the engineered feature columns from transaction_features
    (tx_time, amount, hour_of_day, amount_percentile, avg_amount_last_50_tx, v1..v28).
    Example values below are illustrative, not real transactions.
    """
    tx_time: float = Field(..., example=50000)
    amount: float = Field(..., example=149.62)
    hour_of_day: float = Field(..., example=13)
    amount_percentile: float = Field(..., example=0.75)
    avg_amount_last_50_tx: float = Field(..., example=88.35)
    v1: float = 0.0
    v2: float = 0.0
    v3: float = 0.0
    v4: float = 0.0
    v5: float = 0.0
    v6: float = 0.0
    v7: float = 0.0
    v8: float = 0.0
    v9: float = 0.0
    v10: float = 0.0
    v11: float = 0.0
    v12: float = 0.0
    v13: float = 0.0
    v14: float = 0.0
    v15: float = 0.0
    v16: float = 0.0
    v17: float = 0.0
    v18: float = 0.0
    v19: float = 0.0
    v20: float = 0.0
    v21: float = 0.0
    v22: float = 0.0
    v23: float = 0.0
    v24: float = 0.0
    v25: float = 0.0
    v26: float = 0.0
    v27: float = 0.0
    v28: float = 0.0


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float
    feature_value: float


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_flagged: bool
    threshold_used: float
    top_contributing_features: list[FeatureContribution]


# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "Fraud Detection API is running.",
        "docs": "/docs",
        "threshold": THRESHOLD,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    # Build a single-row DataFrame in the exact column order the model expects
    try:
        input_dict = transaction.dict()
        row = pd.DataFrame([[input_dict[col] for col in FEATURE_COLS]], columns=FEATURE_COLS)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing expected feature: {e}")

    # Fraud probability
    proba = model.predict_proba(row)[0, 1]
    is_flagged = bool(proba >= THRESHOLD)

    # SHAP explanation for this single transaction
    shap_vals = explainer.shap_values(row)[0]
    contributions = pd.DataFrame({
        "feature": FEATURE_COLS,
        "shap_value": shap_vals,
        "feature_value": row.iloc[0].values,
    }).sort_values("shap_value", key=np.abs, ascending=False).head(5)

    top_features = [
        FeatureContribution(
            feature=r["feature"],
            shap_value=float(r["shap_value"]),
            feature_value=float(r["feature_value"]),
        )
        for _, r in contributions.iterrows()
    ]

    return PredictionResponse(
        fraud_probability=float(proba),
        is_flagged=is_flagged,
        threshold_used=THRESHOLD,
        top_contributing_features=top_features,
    )