"""Train final XGBoost and LR models without Unit1/Unit2/HospAdmTime."""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
import xgboost as xgb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "features_v2.npz"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models"
RANDOM_STATE = 42

ABLATED_FEATURES = ["Unit1", "Unit2", "HospAdmTime"]

# XGBoost hyperparameters from specification (no monotone constraints for now)
XGB_PARAMS = {
    "n_estimators": 30,
    "max_depth": 2,
    "min_child_weight": 20,
    "reg_lambda": 10,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "learning_rate": 0.05,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def main() -> None:
    start_time = time.time()

    # Load data
    data = np.load(INPUT_PATH, allow_pickle=True)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_names = data["feature_names"].tolist()

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Original features: {len(feature_names)}")

    # Remove Unit1, Unit2, HospAdmTime features
    ablated_indices = [i for i, name in enumerate(feature_names) if any(af in name for af in ABLATED_FEATURES)]
    print(f"Removing {len(ablated_indices)} features: {[feature_names[i] for i in ablated_indices]}")
    X_train = np.delete(X_train, ablated_indices, axis=1)
    X_test = np.delete(X_test, ablated_indices, axis=1)
    feature_names = [name for i, name in enumerate(feature_names) if i not in ablated_indices]

    print(f"Features after dropping: {len(feature_names)}")

    # Features are already normalized from data_v2, no additional scaling needed
    X_train_scaled = X_train
    X_test_scaled = X_test
    scaler = None  # No scaler needed

    # Train XGBoost
    print("\n=== Training XGBoost ===")
    xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_train_xgb = xgb_model.predict_proba(X_train_scaled)[:, 1]
    y_pred_test_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    train_auc_xgb = roc_auc_score(y_train, y_pred_train_xgb)
    test_auc_xgb = roc_auc_score(y_test, y_pred_test_xgb)
    test_pr_auc_xgb = average_precision_score(y_test, y_pred_test_xgb)

    # F1 at best threshold on validation (20% of train)
    n_val = len(X_train_scaled) // 5
    X_val, X_tr = X_train_scaled[:n_val], X_train_scaled[n_val:]
    y_val, y_tr = y_train[:n_val], y_train[n_val:]

    xgb_val = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_val.fit(X_tr, y_tr)
    y_pred_val = xgb_val.predict_proba(X_val)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_val)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    y_pred_test_binary = (y_pred_test_xgb >= best_threshold).astype(int)
    test_f1_xgb = f1_score(y_test, y_pred_test_binary)
    test_precision_xgb = precision_score(y_test, y_pred_test_binary, zero_division=0)
    test_recall_xgb = recall_score(y_test, y_pred_test_binary, zero_division=0)

    print(f"Train ROC-AUC: {train_auc_xgb:.4f}")
    print(f"Test ROC-AUC: {test_auc_xgb:.4f}")
    print(f"Test PR-AUC: {test_pr_auc_xgb:.4f}")
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"Test F1: {test_f1_xgb:.4f}, Precision: {test_precision_xgb:.4f}, Recall: {test_recall_xgb:.4f}")

    # Save XGBoost model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "xgb_final_v2.pkl", "wb") as f:
        pickle.dump({"model": xgb_model, "threshold": best_threshold}, f)
    print(f"Saved XGBoost model to {OUTPUT_DIR / 'xgb_final_v2.pkl'}")

    # Train Logistic Regression (L2)
    print("\n=== Training Logistic Regression (L2) ===")
    lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
    lr_model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_train_lr = lr_model.predict_proba(X_train_scaled)[:, 1]
    y_pred_test_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    train_auc_lr = roc_auc_score(y_train, y_pred_train_lr)
    test_auc_lr = roc_auc_score(y_test, y_pred_test_lr)
    test_pr_auc_lr = average_precision_score(y_test, y_pred_test_lr)

    # F1 at best threshold on validation
    lr_val = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
    lr_val.fit(X_tr, y_tr)
    y_pred_val_lr = lr_val.predict_proba(X_val)[:, 1]

    precisions_lr, recalls_lr, thresholds_lr = precision_recall_curve(y_val, y_pred_val_lr)
    f1_scores_lr = 2 * (precisions_lr * recalls_lr) / (precisions_lr + recalls_lr + 1e-10)
    best_idx_lr = np.argmax(f1_scores_lr)
    best_threshold_lr = thresholds_lr[best_idx_lr] if best_idx_lr < len(thresholds_lr) else 0.5

    y_pred_test_binary_lr = (y_pred_test_lr >= best_threshold_lr).astype(int)
    test_f1_lr = f1_score(y_test, y_pred_test_binary_lr)
    test_precision_lr = precision_score(y_test, y_pred_test_binary_lr, zero_division=0)
    test_recall_lr = recall_score(y_test, y_pred_test_binary_lr, zero_division=0)

    print(f"Train ROC-AUC: {train_auc_lr:.4f}")
    print(f"Test ROC-AUC: {test_auc_lr:.4f}")
    print(f"Test PR-AUC: {test_pr_auc_lr:.4f}")
    print(f"Best threshold: {best_threshold_lr:.4f}")
    print(f"Test F1: {test_f1_lr:.4f}, Precision: {test_precision_lr:.4f}, Recall: {test_recall_lr:.4f}")

    # Save LR model
    with open(OUTPUT_DIR / "lr_final_v2.pkl", "wb") as f:
        pickle.dump({"model": lr_model, "threshold": best_threshold_lr}, f)
    print(f"Saved LR model to {OUTPUT_DIR / 'lr_final_v2.pkl'}")

    # Prevalence baseline
    prevalence = np.mean(y_test)
    print(f"\nTest prevalence: {prevalence:.4f}")
    print(f"Random classifier AUC: 0.5000")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")

    # Update progress log
    with open(PROJECT_ROOT / "outputs" / "progress_log.md", "a") as f:
        f.write(f"| Step 0: List files | Found features_v2.npz, existing models | {elapsed:.1f} |\n")
        f.write(f"| Step 1: Train final models | XGB AUC={test_auc_xgb:.4f}, LR AUC={test_auc_lr:.4f} | {time.time() - start_time:.1f} |\n")


if __name__ == "__main__":
    main()
