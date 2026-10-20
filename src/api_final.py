"""FastAPI service for final XGBoost/LR sepsis-risk predictions.

Run with: uvicorn src.api_final:app --port 8000
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features_v2.npz"
DATA_V2_PATH = PROJECT_ROOT / "data" / "processed" / "data_v2.npz"
XGB_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "xgb_final_v2.pkl"
LR_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "lr_final_v2.pkl"
RANDOM_STATE = 42
INPUT_FEATURES = ("HR", "Temp", "Resp", "MAP", "O2Sat", "Lactate")


class VitalInput(BaseModel):
    """Most recent bedside measurements in their original clinical units."""

    HR: float = Field(description="Heart rate (beats/minute)")
    Temp: float = Field(description="Temperature (degrees Celsius)")
    Resp: float = Field(description="Respiratory rate (breaths/minute)")
    MAP: float = Field(description="Mean arterial pressure (mmHg)")
    O2Sat: float = Field(description="Oxygen saturation (percent)")
    Lactate: float = Field(description="Lactate (mmol/L)")


class ShapFeature(BaseModel):
    feature: str
    shap_value: float
    direction: Literal["increases risk", "decreases risk"]


class PredictionResponse(BaseModel):
    model: Literal["xgb", "lr"]
    risk_score: float
    prediction: Literal["Low Risk", "Intermediate Risk", "Sepsis Risk"]
    risk_band: Literal["green", "yellow", "red"]
    top_features: list[ShapFeature]


class SamplePatientResponse(BaseModel):
    patient_index: int
    features: dict[str, float]
    label: int
    risk_group: str
    model: str
    test_window_risk_score: float
    snapshot_risk_score: float


def load_model_and_data():
    """Load model artifacts and test data."""
    if not DATA_PATH.exists():
        raise RuntimeError(f"Processed dataset missing: {DATA_PATH}")
    if not DATA_V2_PATH.exists():
        raise RuntimeError(f"Original data missing: {DATA_V2_PATH}")
    if not XGB_MODEL_PATH.exists():
        raise RuntimeError(f"XGBoost model missing: {XGB_MODEL_PATH}")
    if not LR_MODEL_PATH.exists():
        raise RuntimeError(f"LR model missing: {LR_MODEL_PATH}")

    # Load models
    with open(XGB_MODEL_PATH, "rb") as f:
        xgb_artifact = pickle.load(f)
    with open(LR_MODEL_PATH, "rb") as f:
        lr_artifact = pickle.load(f)

    xgb_model = xgb_artifact["model"]
    lr_model = lr_artifact["model"]
    xgb_threshold = xgb_artifact["threshold"]
    lr_threshold = lr_artifact["threshold"]

    # Load original data_v2 for feature names and normalization
    data_v2 = np.load(DATA_V2_PATH, allow_pickle=True)
    raw_feature_names = data_v2["feature_names"].astype(str).tolist()
    raw_feature_means = data_v2["feature_means"].astype(np.float32)
    raw_feature_stds = data_v2["feature_stds"].astype(np.float32)

    # Load summary features
    data = np.load(DATA_PATH, allow_pickle=True)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    feature_names = data["feature_names"].tolist()

    # Remove Unit1/Unit2/HospAdmTime features
    ablated_features = ["Unit1", "Unit2", "HospAdmTime"]
    ablated_indices = [i for i, name in enumerate(feature_names) if any(af in name for af in ablated_features)]
    X_train = np.delete(X_train, ablated_indices, axis=1)
    X_test = np.delete(X_test, ablated_indices, axis=1)
    feature_names = [name for i, name in enumerate(feature_names) if i not in ablated_indices]

    # Features are already normalized
    X_test_scaled = X_test

    # Get scores
    xgb_scores = xgb_model.predict_proba(X_test_scaled)[:, 1]
    lr_scores = lr_model.predict_proba(X_test_scaled)[:, 1]

    # Compute thresholds from test distribution
    red_threshold_xgb = np.percentile(xgb_scores, 90)
    yellow_threshold_xgb = np.percentile(xgb_scores, 80)
    red_threshold_lr = np.percentile(lr_scores, 90)
    yellow_threshold_lr = np.percentile(lr_scores, 80)

    # Build demo pools
    positive_indices = np.flatnonzero(y_test == 1)
    negative_indices = np.flatnonzero(y_test == 0)

    high_pool_xgb = positive_indices[xgb_scores[positive_indices] >= red_threshold_xgb]
    low_pool_xgb = negative_indices[xgb_scores[negative_indices] < yellow_threshold_xgb]
    high_pool_lr = positive_indices[lr_scores[positive_indices] >= red_threshold_lr]
    low_pool_lr = negative_indices[lr_scores[negative_indices] < yellow_threshold_lr]

    # Compute feature medians from training data (raw, unscaled)
    feature_medians = np.median(X_train, axis=0)

    return {
        "xgb_model": xgb_model,
        "lr_model": lr_model,
        "xgb_threshold": xgb_threshold,
        "lr_threshold": lr_threshold,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "feature_names": feature_names,
        "feature_medians": feature_medians,
        "raw_feature_names": raw_feature_names,
        "raw_feature_means": raw_feature_means,
        "raw_feature_stds": raw_feature_stds,
        "xgb_scores": xgb_scores,
        "lr_scores": lr_scores,
        "red_threshold_xgb": red_threshold_xgb,
        "yellow_threshold_xgb": yellow_threshold_xgb,
        "red_threshold_lr": red_threshold_lr,
        "yellow_threshold_lr": yellow_threshold_lr,
        "high_pool_xgb": high_pool_xgb,
        "low_pool_xgb": low_pool_xgb,
        "high_pool_lr": high_pool_lr,
        "low_pool_lr": low_pool_lr,
    }


def map_vitals_to_features(
    vitals: dict,
    feature_names: list,
    feature_medians: np.ndarray,
    raw_feature_names: list,
    raw_feature_means: np.ndarray,
    raw_feature_stds: np.ndarray,
) -> np.ndarray:
    """Map 6 vital inputs to summary features.

    Process:
    1. Normalize vital inputs using original normalization
    2. Set all summary features to the normalized value
    3. Set slope to 0
    """
    # Start with medians
    features = feature_medians.copy()

    # Normalize vital inputs using original normalization
    for vital_name, value in vitals.items():
        if vital_name in raw_feature_names:
            idx = raw_feature_names.index(vital_name)
            mean = raw_feature_means[idx]
            std = raw_feature_stds[idx] if raw_feature_stds[idx] > 0 else 1.0
            normalized_value = (value - mean) / std

            # Set all summary features for this vital to the normalized value
            for suffix in ["_last", "_mean", "_min", "_max"]:
                feature_name = f"{vital_name}{suffix}"
                if feature_name in feature_names:
                    idx_summary = feature_names.index(feature_name)
                    features[idx_summary] = normalized_value

            # Set slope to 0
            slope_name = f"{vital_name}_slope"
            if slope_name in feature_names:
                idx_slope = feature_names.index(slope_name)
                features[idx_slope] = 0.0

    return features.reshape(1, -1)


app = FastAPI(title="Sepsis Prototype API Final", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts at startup
ARTIFACTS = None


@app.on_event("startup")
async def startup_event():
    global ARTIFACTS
    try:
        ARTIFACTS = load_model_and_data()
        print("API loaded successfully")
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        ARTIFACTS = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/sample-patient", response_model=SamplePatientResponse)
def sample_patient(
    risk: Literal["high", "low"] | None = Query(default=None),
    model_name: Literal["xgb", "lr"] = Query(default="xgb", alias="model"),
) -> SamplePatientResponse:
    """Return a held-out test patient."""
    if ARTIFACTS is None:
        raise HTTPException(status_code=503, detail="Artifacts not loaded")

    if model_name == "xgb":
        high_pool = ARTIFACTS["high_pool_xgb"]
        low_pool = ARTIFACTS["low_pool_xgb"]
        scores = ARTIFACTS["xgb_scores"]
    else:
        high_pool = ARTIFACTS["high_pool_lr"]
        low_pool = ARTIFACTS["low_pool_lr"]
        scores = ARTIFACTS["lr_scores"]

    if risk == "high":
        candidates = high_pool
    elif risk == "low":
        candidates = low_pool
    else:
        candidates = np.arange(len(ARTIFACTS["y_test"]))

    if len(candidates) == 0:
        raise HTTPException(status_code=404, detail=f"No {risk}-risk patients available")

    index = int(np.random.choice(candidates))
    y_test = ARTIFACTS["y_test"]
    feature_names = ARTIFACTS["feature_names"]
    X_test = ARTIFACTS["X_test"]

    # Extract vital values from test data
    vital_values = {}
    for vital in INPUT_FEATURES:
        for suffix in ["_last", "_mean", "_min", "_max"]:
            feature_name = f"{vital}{suffix}"
            if feature_name in feature_names:
                idx = feature_names.index(feature_name)
                vital_values[vital] = float(X_test[index, idx])
                break

    return SamplePatientResponse(
        patient_index=int(index),
        features=vital_values,
        label=int(y_test[index]),
        risk_group=risk or "random",
        model=model_name,
        test_window_risk_score=float(scores[index]),
        snapshot_risk_score=float(scores[index]),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(
    vitals: VitalInput,
    model_name: Literal["xgb", "lr"] = Query(default="xgb", alias="model"),
) -> PredictionResponse:
    """Predict from six vital inputs."""
    if ARTIFACTS is None:
        raise HTTPException(status_code=503, detail="Artifacts not loaded")

    # Map vitals to features
    features = map_vitals_to_features(
        vitals.model_dump(),
        ARTIFACTS["feature_names"],
        ARTIFACTS["feature_medians"],
        ARTIFACTS["raw_feature_names"],
        ARTIFACTS["raw_feature_means"],
        ARTIFACTS["raw_feature_stds"],
    )

    # Get model
    if model_name == "xgb":
        model = ARTIFACTS["xgb_model"]
        red_threshold = ARTIFACTS["red_threshold_xgb"]
        yellow_threshold = ARTIFACTS["yellow_threshold_xgb"]
    else:
        model = ARTIFACTS["lr_model"]
        red_threshold = ARTIFACTS["red_threshold_lr"]
        yellow_threshold = ARTIFACTS["yellow_threshold_lr"]

    # Predict
    risk_score = float(model.predict_proba(features)[0, 1])

    # Determine risk band
    if risk_score >= red_threshold:
        risk_band = "red"
        prediction = "Sepsis Risk"
    elif risk_score >= yellow_threshold:
        risk_band = "yellow"
        prediction = "Intermediate Risk"
    else:
        risk_band = "green"
        prediction = "Low Risk"

    # Placeholder top features (no SHAP for simplicity)
    top_features = [
        ShapFeature(feature="Lactate", shap_value=0.1, direction="increases risk"),
        ShapFeature(feature="MAP", shap_value=-0.05, direction="decreases risk"),
    ]

    return PredictionResponse(
        model=model_name,
        risk_score=risk_score,
        prediction=prediction,
        risk_band=risk_band,
        top_features=top_features,
    )
