"""Compute F1, precision, recall at best-F1 threshold from OOF predictions."""

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_v2.npz"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "f1_metrics.json"

RANDOM_STATE = 42
N_FOLDS = 5
ABLATED_FEATURES = ["Unit1", "Unit2", "HospAdmTime"]
INPUT_FEATURES = ("HR", "Temp", "Resp", "MAP", "O2Sat", "Lactate")
MONOTONIC_DIRECTIONS = {"HR": 1, "Temp": 1, "Resp": 1, "MAP": -1, "O2Sat": -1, "Lactate": 1}

start = time.time()
print(f"[{time.time()-start:.1f}s] Loading data...")

with np.load(FEATURES_PATH, allow_pickle=True) as data:
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    feature_names = data["feature_names"].tolist()
    patient_ids_train = data["patient_ids_train"]

# Remove ablated features
ablated_indices = [i for i, name in enumerate(feature_names) if any(af in name for af in ABLATED_FEATURES)]
X_train = np.delete(X_train, ablated_indices, axis=1)
X_test = np.delete(X_test, ablated_indices, axis=1)
feature_names = [name for i, name in enumerate(feature_names) if i not in ablated_indices]

# Build monotonic constraints
monotone_constraints = []
for name in feature_names:
    for vital, direction in MONOTONIC_DIRECTIONS.items():
        if name.startswith(vital + "_"):
            monotone_constraints.append(direction)
            break
    else:
        monotone_constraints.append(0)
monotone_constraints = tuple(monotone_constraints)

print(f"[{time.time()-start:.1f}s] Computing OOF predictions for threshold tuning...")

# XGBoost OOF
gkf = GroupKFold(n_splits=N_FOLDS)
xgb_oof = np.zeros(len(X_train))
for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=patient_ids_train)):
    xgb_fold = xgb.XGBClassifier(
        n_estimators=30, max_depth=3, learning_rate=0.1,
        objective="binary:logistic", eval_metric="auc",
        random_state=RANDOM_STATE, n_jobs=-1,
        monotone_constraints=monotone_constraints,
    )
    xgb_fold.fit(X_train[train_idx], y_train[train_idx])
    xgb_oof[val_idx] = xgb_fold.predict_proba(X_train[val_idx])[:, 1]

# Find best threshold for XGBoost
thresholds = np.arange(0.01, 0.5, 0.01)
best_f1 = 0
best_threshold = 0.5
for thresh in thresholds:
    preds = (xgb_oof >= thresh).astype(int)
    f1 = f1_score(y_train, preds, pos_label=1, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

print(f"[{time.time()-start:.1f}s] XGBoost best threshold: {best_threshold:.2f}, OOF F1: {best_f1:.4f}")

# Train final XGBoost on full train set
xgb_final = xgb.XGBClassifier(
    n_estimators=30, max_depth=3, learning_rate=0.1,
    objective="binary:logistic", eval_metric="auc",
    random_state=RANDOM_STATE, n_jobs=-1,
    monotone_constraints=monotone_constraints,
)
xgb_final.fit(X_train, y_train)
xgb_test_proba = xgb_final.predict_proba(X_test)[:, 1]
xgb_test_preds = (xgb_test_proba >= best_threshold).astype(int)
xgb_test_f1 = f1_score(y_test, xgb_test_preds, pos_label=1, zero_division=0)
xgb_test_precision = precision_score(y_test, xgb_test_preds, pos_label=1, zero_division=0)
xgb_test_recall = recall_score(y_test, xgb_test_preds, pos_label=1, zero_division=0)

print(f"[{time.time()-start:.1f}s] XGBoost test F1: {xgb_test_f1:.4f}, Precision: {xgb_test_precision:.4f}, Recall: {xgb_test_recall:.4f}")

# Logistic Regression OOF
lr_oof = np.zeros(len(X_train))
for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=patient_ids_train)):
    lr_fold = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr_fold.fit(X_train[train_idx], y_train[train_idx])
    lr_oof[val_idx] = lr_fold.predict_proba(X_train[val_idx])[:, 1]

# Find best threshold for LR
best_f1_lr = 0
best_threshold_lr = 0.5
for thresh in thresholds:
    preds = (lr_oof >= thresh).astype(int)
    f1 = f1_score(y_train, preds, pos_label=1, zero_division=0)
    if f1 > best_f1_lr:
        best_f1_lr = f1
        best_threshold_lr = thresh

print(f"[{time.time()-start:.1f}s] LR best threshold: {best_threshold_lr:.2f}, OOF F1: {best_f1_lr:.4f}")

# Train final LR on full train set
lr_final = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr_final.fit(X_train, y_train)
lr_test_proba = lr_final.predict_proba(X_test)[:, 1]
lr_test_preds = (lr_test_proba >= best_threshold_lr).astype(int)
lr_test_f1 = f1_score(y_test, lr_test_preds, pos_label=1, zero_division=0)
lr_test_precision = precision_score(y_test, lr_test_preds, pos_label=1, zero_division=0)
lr_test_recall = recall_score(y_test, lr_test_preds, pos_label=1, zero_division=0)

print(f"[{time.time()-start:.1f}s] LR test F1: {lr_test_f1:.4f}, Precision: {lr_test_precision:.4f}, Recall: {lr_test_recall:.4f}")

# Get F1 at default 0.5 for comparison
xgb_default_preds = (xgb_test_proba >= 0.5).astype(int)
xgb_default_f1 = f1_score(y_test, xgb_default_preds, pos_label=1, zero_division=0)
lr_default_preds = (lr_test_proba >= 0.5).astype(int)
lr_default_f1 = f1_score(y_test, lr_default_preds, pos_label=1, zero_division=0)

print(f"[{time.time()-start:.1f}s] XGBoost F1 at 0.5: {xgb_default_f1:.4f}")
print(f"[{time.time()-start:.1f}s] LR F1 at 0.5: {lr_default_f1:.4f}")
print(f"[{time.time()-start:.1f}s] NOTE: RF, GB, AdaBoost F1 ~0 reflects default 0.5 cutoff, not zero skill")

results = {
    "xgboost": {
        "best_threshold": float(best_threshold),
        "test_f1": float(xgb_test_f1),
        "test_precision": float(xgb_test_precision),
        "test_recall": float(xgb_test_recall),
        "f1_at_0.5": float(xgb_default_f1),
    },
    "logistic_regression": {
        "best_threshold": float(best_threshold_lr),
        "test_f1": float(lr_test_f1),
        "test_precision": float(lr_test_precision),
        "test_recall": float(lr_test_recall),
        "f1_at_0.5": float(lr_default_f1),
    },
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"[{time.time()-start:.1f}s] Saved to {OUTPUT_PATH}")
print(f"[{time.time()-start:.1f}s] DONE")
