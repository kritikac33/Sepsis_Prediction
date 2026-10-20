"""Train the requested no-ICULOS GradientBoosting model in resumable chunks.

This is operationally equivalent to fitting the saved 50-tree estimator with
warm_start=True, but keeps each Windows command under the terminal time limit.
It never alters an existing model.  The checkpoint is a new temporary artifact.
"""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "features_v2.npz"
MODELS = ROOT / "outputs" / "models"
CHECKPOINT = MODELS / "gb_noiculos_build_checkpoint.pkl"
FINAL = MODELS / "gb_noiculos.pkl"


def main() -> None:
    with np.load(DATA, allow_pickle=True) as data:
        names = data["feature_names"].astype(str).tolist()
        keep = np.array([not any(token in name for token in ("Unit1", "Unit2", "HospAdmTime", "ICULOS")) for name in names])
        x, y = data["X_train"][:, keep], data["y_train"].astype(int)
    template = joblib.load(MODELS / "gradient_boosting_v2.pkl")
    target = template.get_params()["n_estimators"]
    if FINAL.exists():
        print(f"Final model already exists: {FINAL.name}")
        return
    if CHECKPOINT.exists():
        model = joblib.load(CHECKPOINT)
        start = model.n_estimators
    else:
        model = clone(template)
        model.set_params(warm_start=True, n_estimators=10)
        start = 0
    next_count = min(start + 10, target)
    model.set_params(warm_start=True, n_estimators=next_count)
    print(f"Fitting GB estimators {start + 1}-{next_count} of {target} on {x.shape}...", flush=True)
    model.fit(x, y)
    if next_count >= target:
        model.set_params(warm_start=False)
        joblib.dump(model, FINAL)
        print(f"Complete: saved {FINAL.name}")
    else:
        joblib.dump(model, CHECKPOINT)
        print(f"Checkpoint complete at {next_count}/{target}")


if __name__ == "__main__":
    main()
