"""Retrain final XGBoost and LR models WITHOUT Unit1/Unit2/HospAdmTime using existing split."""

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_v2.npz"
SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "split.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models"

RANDOM_STATE = 42
ABLATED_FEATURES = ["Unit1", "Unit2", "HospAdmTime"]
INPUT_FEATURES = ("HR", "Temp", "Resp", "MAP", "O2Sat", "Lactate")
MONOTONIC_DIRECTIONS = {"HR": 1, "Temp": 1, "Resp": 1, "MAP": -1, "O2Sat": -1, "Lactate": 1}

start = time.time()
print(f"[{time.time()-start:.1f}s] Loading features and split...")

with np.load(FEATURES_PATH, allow_pickle=True) as data:
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    feature_names = data["feature_names"].tolist()
    patient_ids_train = data["patient_ids_train"]
    patient_ids_test = data["patient_ids_test"]

print(f"[{time.time()-start:.1f}s] Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(feature_names)}")

# Remove ablated features
ablated_indices = [i for i, name in enumerate(feature_names) if any(af in name for af in ABLATED_FEATURES)]
X_train = np.delete(X_train, ablated_indices, axis=1)
X_test = np.delete(X_test, ablated_indices, axis=1)
feature_names = [name for i, name in enumerate(feature_names) if i not in ablated_indices]

print(f"[{time.time()-start:.1f}s] After removing {ABLATED_FEATURES}: {len(feature_names)} features")

# Build monotonic constraints for summary features
monotone_constraints = []
for name in feature_names:
    for vital, direction in MONOTONIC_DIRECTIONS.items():
        if name.startswith(vital + "_"):
            monotone_constraints.append(direction)
            break
    else:
        monotone_constraints.append(0)

monotone_constraints = tuple(monotone_constraints)
print(f"[{time.time()-start:.1f}s] Monotonic constraints on {sum(monotone_constraints) != 0} features")

# Train XGBoost
print(f"[{time.time()-start:.1f}s] Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=30,
    max_depth=3,
    learning_rate=0.1,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    monotone_constraints=monotone_constraints,
)
xgb_model.fit(X_train, y_train)

xgb_test_proba = xgb_model.predict_proba(X_test)[:, 1]
xgb_roc_auc = roc_auc_score(y_test, xgb_test_proba)
xgb_pr_auc = average_precision_score(y_test, xgb_test_proba)

print(f"[{time.time()-start:.1f}s] XGBoost Test ROC-AUC: {xgb_roc_auc:.4f}, PR-AUC: {xgb_pr_auc:.4f}")

# Train Logistic Regression
print(f"[{time.time()-start:.1f}s] Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
lr_model.fit(X_train, y_train)

lr_test_proba = lr_model.predict_proba(X_test)[:, 1]
lr_roc_auc = roc_auc_score(y_test, lr_test_proba)
lr_pr_auc = average_precision_score(y_test, lr_test_proba)

print(f"[{time.time()-start:.1f}s] LR Test ROC-AUC: {lr_roc_auc:.4f}, PR-AUC: {lr_pr_auc:.4f}")

# Save models
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(xgb_model, OUTPUT_DIR / "xgb_v2.pkl")
joblib.dump(lr_model, OUTPUT_DIR / "lr_v2.pkl")

# Store feature names for API
xgb_model.selected_feature_names_ = np.array(feature_names)
lr_model.selected_feature_names_ = np.array(feature_names)

print(f"[{time.time()-start:.1f}s] Saved models to {OUTPUT_DIR}")

# Compare with "with" version if exists
with_path = OUTPUT_DIR / "xgboost_v2.pkl"
if with_path.exists():
    print(f"[{time.time()-start:.1f}s] Comparing with version that includes Unit1/Unit2/HospAdmTime...")
    with_model = joblib.load(with_path)
    with_proba = with_model.predict_proba(X_test)[:, 1]
    with_roc_auc = roc_auc_score(y_test, with_proba)
    with_pr_auc = average_precision_score(y_test, with_proba)
    print(f"[{time.time()-start:.1f}s] With Unit1/Unit2/HospAdmTime: ROC-AUC {with_roc_auc:.4f}, PR-AUC {with_pr_auc:.4f}")
    print(f"[{time.time()-start:.1f}s] Without Unit1/Unit2/HospAdmTime: ROC-AUC {xgb_roc_auc:.4f}, PR-AUC {xgb_pr_auc:.4f}")
    
    roc_diff = abs(xgb_roc_auc - with_roc_auc)
    pr_diff = abs(xgb_pr_auc - with_pr_auc)
    
    if roc_diff < 0.01 and pr_diff < 0.01:
        print(f"[{time.time()-start:.1f}s] DECISION: Use WITHOUT version (difference < 0.01 AUC)")
    else:
        print(f"[{time.time()-start:.1f}s] DECISION: WITH version has higher AUC by > 0.01")
else:
    print(f"[{time.time()-start:.1f}s] No 'with' version found for comparison")

print(f"[{time.time()-start:.1f}s] DONE")
