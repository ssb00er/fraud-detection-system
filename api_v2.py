"""
Fraud Detection Project: FastAPI Real-Time Inference Service (v2)
Adds a /predict_explained endpoint that combines the model's SHAP-based
prediction with an LLM-generated plain-English explanation.

Usage:
    uvicorn api_v2:app --reload

Then test with:
    http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import shap
import numpy as np
import pandas as pd

from llm_explainer import generate_explanation

MODEL_PATH = "fraud_model.joblib"

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud scoring with SHAP explanations and LLM-generated summaries.",
    version="2.0.0",
)

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
THRESHOLD = artifact["threshold"]
FEATURE_COLS = artifact["feature_cols"]
explainer = shap.TreeExplainer(model)

print(f"Loaded model. Threshold = {THRESHOLD}. Expecting {len(FEATURE_COLS)} features.")


class Transaction(BaseModel):
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


class ExplainedPredictionResponse(PredictionResponse):
    plain_english_explanation: str


def score_transaction(transaction: Transaction):
    try:
        input_dict = transaction.dict()
        row = pd.DataFrame([[input_dict[col] for col in FEATURE_COLS]], columns=FEATURE_COLS)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing expected feature: {e}")

    proba = model.predict_proba(row)[0, 1]
    is_flagged = bool(proba >= THRESHOLD)

    shap_vals = explainer.shap_values(row)[0]
    contributions = pd.DataFrame({
        "feature": FEATURE_COLS,
        "shap_value": shap_vals,
        "feature_value": row.iloc[0].values,
    }).sort_values("shap_value", key=np.abs, ascending=False).head(5)

    top_features = [
        {
            "feature": r["feature"],
            "shap_value": float(r["shap_value"]),
            "feature_value": float(r["feature_value"]),
        }
        for _, r in contributions.iterrows()
    ]

    return proba, is_flagged, top_features


@app.get("/")
def root():
    return {"message": "Fraud Detection API v2 is running.", "docs": "/docs", "threshold": THRESHOLD}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    proba, is_flagged, top_features = score_transaction(transaction)
    return PredictionResponse(
        fraud_probability=float(proba),
        is_flagged=is_flagged,
        threshold_used=THRESHOLD,
        top_contributing_features=[FeatureContribution(**f) for f in top_features],
    )


@app.post("/predict_explained", response_model=ExplainedPredictionResponse)
def predict_explained(transaction: Transaction):
    proba, is_flagged, top_features = score_transaction(transaction)

    try:
        explanation = generate_explanation(
            fraud_probability=float(proba),
            threshold=THRESHOLD,
            is_flagged=is_flagged,
            top_features=top_features,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM explanation failed: {e}")

    return ExplainedPredictionResponse(
        fraud_probability=float(proba),
        is_flagged=is_flagged,
        threshold_used=THRESHOLD,
        top_contributing_features=[FeatureContribution(**f) for f in top_features],
        plain_english_explanation=explanation,
    )