"""Quick ensemble of XGBoost and LR using rank-normalized probabilities."""

import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             roc_auc_score)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_v2.npz"
XGB_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "xgb_final_v2.pkl"
LR_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "lr_final_v2.pkl"


def main():
    start_time = time.time()

    # Load data
    data = np.load(FEATURES_PATH, allow_pickle=True)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    patient_ids_train = data["patient_ids_train"]
    feature_names = data["feature_names"].tolist()

    # Remove ablated features
    ablated = [i for i, n in enumerate(feature_names) if any(af in n for af in ["Unit1", "Unit2", "HospAdmTime"])]
    X_train = np.delete(X_train, ablated, axis=1)
    X_test = np.delete(X_test, ablated, axis=1)

    # Load models
    with open(XGB_MODEL_PATH, "rb") as f:
        xgb_artifact = pickle.load(f)
    with open(LR_MODEL_PATH, "rb") as f:
        lr_artifact = pickle.load(f)

    xgb_model = xgb_artifact["model"]
    lr_model = lr_artifact["model"]

    # Get predictions
    xgb_train_pred = xgb_model.predict_proba(X_train)[:, 1]
    lr_train_pred = lr_model.predict_proba(X_train)[:, 1]
    xgb_test_pred = xgb_model.predict_proba(X_test)[:, 1]
    lr_test_pred = lr_model.predict_proba(X_test)[:, 1]

    # Rank-normalize predictions
    def rank_normalize(pred):
        from scipy.stats import rankdata
        return rankdata(pred) / len(pred)

    xgb_train_rank = rank_normalize(xgb_train_pred)
    lr_train_rank = rank_normalize(lr_train_pred)
    xgb_test_rank = rank_normalize(xgb_test_pred)
    lr_test_rank = rank_normalize(lr_test_pred)

    # Ensemble by averaging ranks
    ensemble_train_pred = (xgb_train_rank + lr_train_rank) / 2
    ensemble_test_pred = (xgb_test_rank + lr_test_rank) / 2

    # Choose best-F1 threshold on 20% validation slice of TRAIN
    train_idx, val_idx = train_test_split(
        np.arange(len(y_train)), test_size=0.2, random_state=42, stratify=y_train
    )
    ensemble_val_ens = ensemble_train_pred[val_idx]
    y_val_ens = y_train[val_idx]

    precisions, recalls, thresholds = precision_recall_curve(y_val_ens, ensemble_val_ens)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    # Evaluate on test
    test_auc = roc_auc_score(y_test, ensemble_test_pred)
    test_pr_auc = average_precision_score(y_test, ensemble_test_pred)
    y_pred_test_binary = (ensemble_test_pred >= best_threshold).astype(int)
    test_f1 = f1_score(y_test, y_pred_test_binary, zero_division=0)

    print(f"=== Ensemble Results ===")
    print(f"Test ROC-AUC: {test_auc:.4f}")
    print(f"Test PR-AUC: {test_pr_auc:.4f}")
    print(f"Test F1 (threshold={best_threshold:.4f}): {test_f1:.4f}")

    # Compare to individual models
    xgb_test_auc = roc_auc_score(y_test, xgb_test_pred)
    lr_test_auc = roc_auc_score(y_test, lr_test_pred)

    print(f"\nXGBoost Test AUC: {xgb_test_auc:.4f}")
    print(f"LR Test AUC: {lr_test_auc:.4f}")
    print(f"Ensemble Test AUC: {test_auc:.4f}")

    if test_auc > max(xgb_test_auc, lr_test_auc):
        print("Ensemble improved AUC!")
    else:
        print("Ensemble did not improve AUC.")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")

    # Update progress log
    with open(PROJECT_ROOT / "outputs" / "progress_log.md", "a") as f:
        f.write(f"| PART 4: Ensemble | Test AUC={test_auc:.4f} (vs XGB {xgb_test_auc:.4f}, LR {lr_test_auc:.4f}) | {elapsed:.1f} |\n")


if __name__ == "__main__":
    main()
