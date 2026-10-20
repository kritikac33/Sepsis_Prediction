"""FastAPI service for the fixed-split no-ICULOS sepsis prototype.

Run locally without reload for validation:
    .\\venv\\Scripts\\python.exe -m uvicorn src.api:app --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import shap
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "features_v2.npz"
RAW_PATH = ROOT / "data" / "processed" / "data_v2.npz"
BANDS_PATH = ROOT / "data" / "processed" / "band_thresholds_noiculos.json"
LEGACY_BANDS_PATH = ROOT / "data" / "processed" / "validation_thresholds.json"
METRICS_PATH = ROOT / "outputs" / "metrics" / "iculos_ablation_report.json"
MODELS = ROOT / "outputs" / "models"
SUBTYPES = MODELS / "kmeans_subtypes.pkl"
INPUTS = ("HR", "Temp", "Resp", "MAP", "O2Sat", "Lactate")
ACTIVE_MODELS = ("xgb", "rf", "gb", "adaboost")
MODEL_FILES = {
    "xgb": "xgb_noiculos.pkl", "rf": "rf_noiculos.pkl", "gb": "gb_noiculos.pkl", "adaboost": "adaboost_noiculos.pkl",
    "xgb_with_iculos": "xgboost_v2.pkl", "rf_with_iculos": "random_forest_v2.pkl",
    "gb_with_iculos": "gradient_boosting_v2.pkl", "adaboost_with_iculos": "adaboost_v2.pkl",
}
MODEL_LABELS = {"xgb": "XGBoost", "rf": "Random Forest", "gb": "Gradient Boosting", "adaboost": "AdaBoost"}
UNITS = {"HR":"bpm", "Temp":"°C", "Resp":"/min", "MAP":"mmHg", "SBP":"mmHg", "DBP":"mmHg", "O2Sat":"%", "pH":"", "BaseExcess":"mmol/L", "HCO3":"mmol/L", "PaCO2":"mmHg", "FiO2":"%", "SaO2":"%", "BUN":"mg/dL", "Creatinine":"mg/dL", "Glucose":"mg/dL", "Potassium":"mEq/L", "Calcium":"mg/dL", "Chloride":"mEq/L", "Magnesium":"mg/dL", "Phosphate":"mg/dL", "Sodium":"mEq/L", "Hct":"%", "Hgb":"g/dL", "WBC":"K/uL", "Platelets":"K/uL", "PTT":"sec", "Fibrinogen":"mg/dL", "AST":"U/L", "Alkalinephos":"U/L", "Bilirubin_total":"mg/dL", "Bilirubin_direct":"mg/dL", "Lactate":"mmol/L", "Troponin":"ng/mL", "Age":"years", "Gender":"", "ICULOS":"hours", "HospAdmTime":"hours"}
CATEGORIES = {"Vitals":["HR","Temp","Resp","MAP","SBP","DBP","O2Sat"], "Blood gases":["pH","BaseExcess","HCO3","PaCO2","FiO2","SaO2"], "Chemistry":["BUN","Creatinine","Glucose","Potassium","Calcium","Chloride","Magnesium","Phosphate","Sodium","Lactate"], "Hematology":["Hct","Hgb","WBC","Platelets","PTT","Fibrinogen"], "Reference only (not used by model)":["Age","Gender","HospAdmTime","ICULOS"]}

class VitalInput(BaseModel):
    HR: float; Temp: float; Resp: float; MAP: float; O2Sat: float; Lactate: float; window_id: int | None = None
class ShapFeature(BaseModel):
    feature: str; shap_value: float; direction: str
class LabFeature(BaseModel):
    name: str; value: float | None; unit: str; was_measured: bool; note: str | None = None
class LabCategory(BaseModel):
    category: str; features: list[LabFeature]
class SampleResponse(BaseModel):
    patient_index: int; window_id: int; features: dict[str,float]; label: int; risk_group: str; model: str; test_window_risk_score: float; risk_band: str; full_lab_panel: list[LabCategory]; pool_note: str | None = None
class PredictionResponse(BaseModel):
    model: str; risk_score: float; prediction: str; risk_band: str; top_features: list[ShapFeature]; prediction_mode: str; shap_computing: bool = False; disclaimer: str = "Provisional research display band, not a clinical threshold."

def active_name(name: str) -> str: return name.removesuffix("_with_iculos")
def feature_mask(names: list[str], no_iculos: bool) -> np.ndarray:
    blocked = ("Unit1", "Unit2", "HospAdmTime") + (("ICULOS",) if no_iculos else ())
    return np.array([not any(token in item for token in blocked) for item in names], dtype=bool)
def band(score: float, thresholds: dict[str,float]) -> str: return "red" if score >= thresholds["red"] else "yellow" if score >= thresholds["yellow"] else "green"
def prediction_label(risk_band: str) -> str: return {"red":"Sepsis Risk", "yellow":"Intermediate Risk", "green":"Low Risk"}[risk_band]
def friendly(name: str) -> str:
    base, _, suffix = name.rpartition("_"); labels={"last":"last hour", "mean":"mean", "min":"min", "max":"max", "slope":"trend", "frac":"measured fraction"}
    return f"{base} ({labels.get(suffix, suffix)})" if base else name
def model_params(model: Any) -> dict[str, Any]:
    raw=model.get_params(deep=False) if hasattr(model,"get_params") else {}
    def plain(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, np.generic):
            item = value.item()
            return None if isinstance(item, float) and not math.isfinite(item) else item
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        if isinstance(value, dict):
            return {str(key): plain(item) for key, item in value.items()}
        return None
    return {str(key): plain(value) for key,value in raw.items() if key not in {"callbacks", "feature_weights"}}

def panel_for_window(state: Any, window_id: int) -> list[LabCategory]:
    raw = state.raw_x_test[window_id, -1, :] * state.raw_stds + state.raw_means; rows=[]
    for category, features in CATEGORIES.items():
        values=[]
        for name in features:
            if name not in state.raw_names: continue
            index=state.raw_names.index(name); flag=f"{name}_was_measured"; measured=True if flag not in state.raw_names else bool(state.raw_x_test[window_id,-1,state.raw_names.index(flag)] > .5)
            value=float(raw[index]) if measured or name in {"Age","Gender","HospAdmTime","ICULOS"} else None
            if name == "Age" and value is not None: value=float(round(value))
            values.append(LabFeature(name=name,value=value,unit=UNITS.get(name,""),was_measured=measured,note=None if measured else "Not measured"))
        if values: rows.append(LabCategory(category=category,features=values))
    return rows

@asynccontextmanager
async def lifespan(app: FastAPI):
    required=[FEATURES_PATH,RAW_PATH,METRICS_PATH,*[MODELS/p for p in MODEL_FILES.values()]]; missing=[str(p) for p in required if not p.exists()]
    if missing: raise RuntimeError("Required artifact(s) missing: "+"; ".join(missing))
    state=app.state
    with np.load(FEATURES_PATH,allow_pickle=True) as data:
        names=data["feature_names"].astype(str).tolist(); xtr=data["X_train"].astype(np.float32); xte=data["X_test"].astype(np.float32); state.y_test=data["y_test"].astype(int); state.patient_ids_test=data["patient_ids_test"].astype(str); state.patient_ids_train=data["patient_ids_train"].astype(str)
    with_mask=feature_mask(names,False); no_mask=feature_mask(names,True)
    state.x_train={"with":xtr[:,with_mask],"no":xtr[:,no_mask]}; state.x_test={"with":xte[:,with_mask],"no":xte[:,no_mask]}; state.names={"with":[n for n,m in zip(names,with_mask) if m],"no":[n for n,m in zip(names,no_mask) if m]}; state.medians={k:np.median(v,axis=0).astype(np.float32) for k,v in state.x_train.items()}
    with np.load(RAW_PATH,allow_pickle=True) as raw: state.raw_x_test=raw["X_test"].astype(np.float32); state.raw_names=raw["feature_names"].astype(str).tolist(); state.raw_means=raw["feature_means"].astype(np.float32); state.raw_stds=raw["feature_stds"].astype(np.float32)
    state.models={name:joblib.load(MODELS/file) for name,file in MODEL_FILES.items()}; state.model_key={name:"with" if name.endswith("_with_iculos") else "no" for name in MODEL_FILES}
    threshold_source=BANDS_PATH if BANDS_PATH.exists() else LEGACY_BANDS_PATH
    state.thresholds=json.loads(threshold_source.read_text(encoding="utf-8")).get("thresholds",{})
    state.metrics=json.loads(METRICS_PATH.read_text(encoding="utf-8")); state.rng=np.random.default_rng(20261015); state.scores={}; state.explainers={}
    for name,model in state.models.items():
        key=state.model_key[name]
        if name.startswith("adaboost"):
            state.explainers[name]=shap.KernelExplainer(lambda values,m=model:m.predict_proba(values)[:,1],state.x_train[key][:50]); print(f"{name}: KernelExplainer initialized with 50-row background (on-demand computing)",flush=True); continue
        try:
            explainer=shap.TreeExplainer(model); explainer.shap_values(state.x_test[key][:1],check_additivity=True); state.explainers[name]=explainer; print(f"{name}: TreeExplainer additivity check passed",flush=True)
        except Exception as exc: print(f"{name}: TreeExplainer additivity check failed: {exc}",flush=True)
    print("Sepsis API ready: /health /model-info /sample-patient /predict /subtypes",flush=True); yield

app=FastAPI(title="Sepsis Prototype API",lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:5173","http://127.0.0.1:5173"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
@app.get("/health")
def health(): return {"status":"ok"}
@app.get("/model-info")
def model_info(model: str=Query("xgb")):
    if model not in MODEL_FILES: raise HTTPException(400,"Unknown model")
    name=active_name(model); key=app.state.model_key[model]; metric=app.state.metrics.get("models",{}).get(name,{})
    return {"model":model,"display_name":MODEL_LABELS.get(name),"feature_count":len(app.state.names.get(key,[])),"iculos_excluded":key=="no","hyperparameters":model_params(app.state.models[model]),"train_patient_count":app.state.metrics.get("train_patients"),"test_patient_count":app.state.metrics.get("test_patients"),"test_metrics":metric.get("without_iculos" if key=="no" else "with_iculos"),"base_rate":app.state.metrics.get("base_rate"),"band_label":"provisional display bands, not clinical thresholds"}
@app.get("/sample-patient",response_model=SampleResponse)
def sample_patient(risk: str|None=Query(None),model: str=Query("xgb")):
    if model not in MODEL_FILES: raise HTTPException(400,"Unknown model")
    if risk not in (None,"high","low"): raise HTTPException(400,"risk must be high or low")
    state=app.state; key=state.model_key[model]; scores=state.scores.setdefault(model,state.models[model].predict_proba(state.x_test[key])[:,1]); base=active_name(model); thresholds=state.thresholds[base]; positives=np.flatnonzero(state.y_test==1); negatives=np.flatnonzero(state.y_test==0); note=None
    if risk=="high":
        candidates=positives[scores[positives]>=thresholds["red"]]
        if not len(candidates): candidates=positives; note="No true positives reached the red provisional band; returned closest high-score window."
        choices=candidates[np.argsort(scores[candidates])[::-1]][:min(25,len(candidates))]
    elif risk=="low":
        candidates=negatives[scores[negatives]<thresholds["yellow"]]
        if not len(candidates): candidates=negatives; note="No true negatives reached the green provisional band; returned closest low-score window."
        choices=candidates[np.argsort(scores[candidates])][:min(25,len(candidates))]
    else: choices=np.arange(len(state.y_test))
    index=int(state.rng.choice(choices)); raw=state.raw_x_test[index,-1,:]*state.raw_stds+state.raw_means; vitals={name:float(raw[state.raw_names.index(name)]) for name in INPUTS}; score=float(scores[index])
    return SampleResponse(patient_index=index,window_id=index,features=vitals,label=int(state.y_test[index]),risk_group=risk or "random",model=model,test_window_risk_score=score,risk_band=band(score,thresholds),full_lab_panel=panel_for_window(state,index),pool_note=note)
def snapshot(state: Any,model: str,values: VitalInput) -> np.ndarray:
    key=state.model_key[model]; out=state.medians[key].copy(); names=state.names[key]
    for vital in INPUTS:
        raw_i=state.raw_names.index(vital); norm=(getattr(values,vital)-state.raw_means[raw_i])/max(float(state.raw_stds[raw_i]),1e-8)
        for suffix in ("last","mean","min","max"):
            feature=f"{vital}_{suffix}"
            if feature in names: out[names.index(feature)]=norm
        slope=f"{vital}_slope"
        if slope in names: out[names.index(slope)]=0.0
    return out.reshape(1,-1)
@app.post("/predict",response_model=PredictionResponse)
def predict(values: VitalInput,model: str=Query("xgb")):
    if model not in MODEL_FILES: raise HTTPException(400,"Unknown model")
    state=app.state; key=state.model_key[model]
    if values.window_id is not None:
        if not 0<=values.window_id<len(state.y_test): raise HTTPException(400,"Invalid window_id")
        x=state.x_test[key][values.window_id:values.window_id+1]; mode="full_window"
    else: x=snapshot(state,model,values); mode="snapshot"
    score=float(state.models[model].predict_proba(x)[0,1]); risk_band=band(score,state.thresholds[active_name(model)])
    try:
        result=state.explainers[model].shap_values(x); result=result[-1] if isinstance(result,list) else result; values_shap=np.asarray(result).reshape(-1); pairs=sorted(zip(state.names[key],values_shap),key=lambda p:abs(p[1]),reverse=True)[:5]; top=[ShapFeature(feature=friendly(name),shap_value=float(abs(value)),direction="increases risk" if value>=0 else "decreases risk") for name,value in pairs]
    except Exception as exc: raise HTTPException(500,f"SHAP explanation failed for {model}: {exc}")
    return PredictionResponse(model=model,risk_score=score,prediction=prediction_label(risk_band),risk_band=risk_band,top_features=top,prediction_mode=mode)
@app.get("/subtypes")
def subtypes():
    if not SUBTYPES.exists(): return None
    item=joblib.load(SUBTYPES); return {"n_clusters":item["n_clusters"],"silhouette_score":float(item["silhouette_score"]),"clusters":item["cluster_records"]}
