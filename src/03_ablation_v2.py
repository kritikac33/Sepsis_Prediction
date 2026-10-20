"""Train XGBoost with and without Unit1/Unit2/HospAdmTime using patient-level GroupKFold."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
import xgboost as xgb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "features_v2.npz"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "phase3_ablation.json"
RANDOM_STATE = 42
N_FOLDS = 5

ABLATED_FEATURES = ["Unit1", "Unit2", "HospAdmTime"]


def train_and_evaluate(X: np.ndarray, y: np.ndarray, patient_ids: np.ndarray, feature_names: list[str]) -> dict:
    """Train XGBoost with patient-level GroupKFold and return metrics."""
    gkf = GroupKFold(n_splits=N_FOLDS)

    roc_aucs = []
    pr_aucs = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=patient_ids)):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_fold_train, y_fold_train)
        y_pred = model.predict_proba(X_fold_val)[:, 1]

        roc_auc = roc_auc_score(y_fold_val, y_pred)
        pr_auc = average_precision_score(y_fold_val, y_pred)

        roc_aucs.append(roc_auc)
        pr_aucs.append(pr_auc)
        print(f"  Fold {fold_idx + 1}: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")

    return {
        "roc_auc_mean": float(np.mean(roc_aucs)),
        "roc_auc_std": float(np.std(roc_aucs)),
        "pr_auc_mean": float(np.mean(pr_aucs)),
        "pr_auc_std": float(np.std(pr_aucs)),
        "roc_aucs": [float(x) for x in roc_aucs],
        "pr_aucs": [float(x) for x in pr_aucs],
    }


def main() -> None:
    data = np.load(INPUT_PATH, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    patient_ids_train = data["patient_ids_train"]
    feature_names = data["feature_names"].tolist()

    print(f"Train samples: {len(X_train)}")
    print(f"Features: {len(feature_names)}")

    X_combined = X_train
    y_combined = y_train
    patient_ids_combined = patient_ids_train

    print("\n=== With Unit1, Unit2, HospAdmTime ===")
    metrics_with = train_and_evaluate(X_combined, y_combined, patient_ids_combined, feature_names)

    ablated_indices = [i for i, name in enumerate(feature_names) if any(af in name for af in ABLATED_FEATURES)]
    X_without = np.delete(X_combined, ablated_indices, axis=1)
    feature_names_without = [name for i, name in enumerate(feature_names) if i not in ablated_indices]

    print(f"\n=== Without Unit1, Unit2, HospAdmTime ===")
    print(f"Features removed: {len(ablated_indices)}, remaining: {len(feature_names_without)}")
    metrics_without = train_and_evaluate(X_without, y_combined, patient_ids_combined, feature_names_without)

    results = {
        "with_features": {
            "features_included": ABLATED_FEATURES,
            "metrics": metrics_with,
        },
        "without_features": {
            "features_excluded": ABLATED_FEATURES,
            "metrics": metrics_without,
        },
        "recommendation": "",
    }

    if metrics_with["roc_auc_mean"] > metrics_without["roc_auc_mean"]:
        results["recommendation"] = "Keep Unit1, Unit2, HospAdmTime - higher ROC-AUC"
    elif metrics_without["roc_auc_mean"] > metrics_with["roc_auc_mean"]:
        results["recommendation"] = "Remove Unit1, Unit2, HospAdmTime - higher ROC-AUC without them"
    else:
        if metrics_with["pr_auc_mean"] > metrics_without["pr_auc_mean"]:
            results["recommendation"] = "Keep Unit1, Unit2, HospAdmTime - higher PR-AUC (ROC-AUC tied)"
        else:
            results["recommendation"] = "Remove Unit1, Unit2, HospAdmTime - higher PR-AUC (ROC-AUC tied)"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {OUTPUT_PATH}")
    print(f"\nRecommendation: {results['recommendation']}")


if __name__ == "__main__":
    main()
