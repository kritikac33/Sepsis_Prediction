"""Minimal FastAPI for v2 models (summary features)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_v2.npz"

INPUT_FEATURES = ("HR", "Temp", "Resp", "MAP", "O2Sat", "Lactate")
ABLATED_FEATURES = ["Unit1", "Unit2", "HospAdmTime"]


class VitalInput(BaseModel):
    HR: float
    Temp: float
    Resp: float
    MAP: float
    O2Sat: float
    Lactate: float


class PredictionResponse(BaseModel):
    model: str
    risk_score: float
    prediction: Literal["Low Risk", "Sepsis Risk"]


app = FastAPI(title="Sepsis V2 API", version="0.2.0")

# Load models at startup
@app.on_event("startup")
def load_models():
    if not FEATURES_PATH.exists():
        raise RuntimeError(f"Features missing: {FEATURES_PATH}")

    with np.load(FEATURES_PATH, allow_pickle=True) as data:
        all_feature_names = data["feature_names"].tolist()
        app.state.X_test = data["X_test"]
        app.state.y_test = data["y_test"]

    # Filter out ablated features (Unit1, Unit2, HospAdmTime)
    app.state.feature_names = [name for name in all_feature_names if not any(af in name for af in ABLATED_FEATURES)]

    models = {}
    for name in ["xgboost_v2.pkl", "logistic_regression_v2.pkl", "random_forest_v2.pkl", "gradient_boosting_v2.pkl", "adaboost_v2.pkl"]:
        path = MODELS_DIR / name
        if path.exists():
            with open(path, "rb") as f:
                models[name.replace("_v2.pkl", "")] = pickle.load(f)

    if not models:
        raise RuntimeError("No v2 models found")

    app.state.models = models
    print(f"Loaded {len(models)} v2 models with {len(app.state.feature_names)} features")


@app.get("/health")
def health():
    return {"status": "ok", "models": list(app.state.models.keys())}


@app.post("/predict", response_model=PredictionResponse)
def predict(vitals: VitalInput, model: Literal["xgboost", "logistic_regression", "random_forest", "gradient_boosting", "adaboost"] = "xgboost"):
    if model not in app.state.models:
        raise HTTPException(status_code=404, detail=f"Model {model} not found")

    model_obj = app.state.models[model]
    feature_names = app.state.feature_names

    # Build summary features from single snapshot (all stats = input value)
    x = np.zeros(len(feature_names), dtype=np.float32)
    vital_map = {"HR": vitals.HR, "Temp": vitals.Temp, "Resp": vitals.Resp, "MAP": vitals.MAP, "O2Sat": vitals.O2Sat, "Lactate": vitals.Lactate}

    for i, name in enumerate(feature_names):
        if name.endswith("_frac"):
            x[i] = 1.0  # measured
        else:
            # Check if this feature is for one of our vitals
            matched = False
            for vital_name, vital_value in vital_map.items():
                if name.startswith(vital_name + "_"):
                    x[i] = vital_value
                    matched = True
                    break
            if not matched:
                x[i] = 0.0  # median placeholder

    try:
        risk_score = float(model_obj.predict_proba(x.reshape(1, -1))[0, 1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    prediction = "Sepsis Risk" if risk_score > 0.1 else "Low Risk"

    return PredictionResponse(model=model, risk_score=risk_score, prediction=prediction)


@app.get("/evaluate")
def evaluate():
    """Evaluate all models on test set."""
    results = {}
    for name, model in app.state.models.items():
        y_pred = model.predict_proba(app.state.X_test)[:, 1]
        auc = roc_auc_score(app.state.y_test, y_pred)
        results[name] = {"test_roc_auc": float(auc)}
    return results
