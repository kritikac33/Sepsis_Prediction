"""Compute one model's train-only GroupKFold OOF display bands.

Run once per model, then run with --combine.  Individual result files are new
artifacts so an interrupted Windows terminal command cannot cause test labels
to be used as a fallback.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "features_v2.npz"
MODELS = ROOT / "outputs" / "models"
OUT = ROOT / "outputs" / "metrics"
FINAL = ROOT / "data" / "processed" / "band_thresholds_noiculos.json"
PATHS = {"xgb": "xgb_noiculos.pkl", "rf": "rf_noiculos.pkl", "gb": "gb_noiculos.pkl", "adaboost": "adaboost_noiculos.pkl"}

def run(name: str) -> None:
    with np.load(DATA, allow_pickle=True) as data:
        names = data["feature_names"].astype(str).tolist()
        keep = np.array([not any(token in feature for token in ("Unit1", "Unit2", "HospAdmTime", "ICULOS")) for feature in names])
        x, y, groups = data["X_train"][:, keep], data["y_train"].astype(int), data["patient_ids_train"].astype(str)
    model = joblib.load(MODELS / PATHS[name])
    scores = np.empty(len(y), dtype=float)
    for number, (fit, valid) in enumerate(GroupKFold(n_splits=3).split(x, y, groups), 1):
        fitted = clone(model)
        fitted.fit(x[fit], y[fit])
        scores[valid] = fitted.predict_proba(x[valid])[:, 1]
        print(f"{name}: fold {number}/3 complete", flush=True)
    OUT.mkdir(exist_ok=True)
    record = {"model": name, "yellow": float(np.percentile(scores, 75)), "red": float(np.percentile(scores, 90)), "n_train_windows": int(len(y)), "n_train_patients": int(len(np.unique(groups)))}
    (OUT / f"oof_bands_{name}_noiculos.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record), flush=True)

def combine() -> None:
    records = {name: json.loads((OUT / f"oof_bands_{name}_noiculos.json").read_text(encoding="utf-8")) for name in PATHS}
    payload = {"label": "provisional display bands, not clinical thresholds", "method": "3-fold GroupKFold out-of-fold predictions on train patients only; yellow >= 75th percentile, red >= 90th percentile.", "feature_set": "ICULOS-derived, Unit1, Unit2, and HospAdmTime columns excluded", "thresholds": {name: {"yellow": r["yellow"], "red": r["red"]} for name, r in records.items()}}
    FINAL.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {FINAL}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("model", nargs="?", choices=list(PATHS)); p.add_argument("--combine", action="store_true"); a=p.parse_args()
    combine() if a.combine else run(a.model)
