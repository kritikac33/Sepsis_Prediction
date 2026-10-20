"""Train multiple models on summary features (without Unit1/Unit2/HospAdmTime)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
import xgboost as xgb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "features_v2.npz"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "phase4_models.json"
RANDOM_STATE = 42
N_FOLDS = 2

ABLATED_FEATURES = ["Unit1", "Unit2", "HospAdmTime"]


def train_model(model, X: np.ndarray, y: np.ndarray, patient_ids: np.ndarray) -> dict:
    """Train a model with patient-level GroupKFold and return metrics."""
    gkf = GroupKFold(n_splits=N_FOLDS)
    roc_aucs = []
    pr_aucs = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=patient_ids)):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]

        model.fit(X_fold_train, y_fold_train)
        y_pred = model.predict_proba(X_fold_val)[:, 1]

        roc_aucs.append(roc_auc_score(y_fold_val, y_pred))
        pr_aucs.append(average_precision_score(y_fold_val, y_pred))

    return {
        "roc_auc_mean": float(np.mean(roc_aucs)),
        "roc_auc_std": float(np.std(roc_aucs)),
        "pr_auc_mean": float(np.mean(pr_aucs)),
        "pr_auc_std": float(np.std(pr_aucs)),
    }


def main() -> None:
    data = np.load(INPUT_PATH, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    patient_ids_train = data["patient_ids_train"]
    feature_names = data["feature_names"].tolist()

    # Remove Unit1, Unit2, HospAdmTime features
    ablated_indices = [i for i, name in enumerate(feature_names) if any(af in name for af in ABLATED_FEATURES)]
    X_train = np.delete(X_train, ablated_indices, axis=1)
    feature_names = [name for i, name in enumerate(feature_names) if i not in ablated_indices]

    print(f"Train samples: {len(X_train)}")
    print(f"Features (after dropping Unit1/Unit2/HospAdmTime): {len(feature_names)}")

    results = {}

    # XGBoost
    print("\n=== XGBoost ===")
    xgb_model = xgb.XGBClassifier(
        n_estimators=30,
        max_depth=3,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    results["xgboost"] = train_model(xgb_model, X_train, y_train, patient_ids_train)
    print(f"ROC-AUC: {results['xgboost']['roc_auc_mean']:.4f} ± {results['xgboost']['roc_auc_std']:.4f}")
    print(f"PR-AUC: {results['xgboost']['pr_auc_mean']:.4f} ± {results['xgboost']['pr_auc_std']:.4f}")

    # Logistic Regression
    print("\n=== Logistic Regression ===")
    lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
    results["logistic_regression"] = train_model(lr_model, X_train, y_train, patient_ids_train)
    print(f"ROC-AUC: {results['logistic_regression']['roc_auc_mean']:.4f} ± {results['logistic_regression']['roc_auc_std']:.4f}")
    print(f"PR-AUC: {results['logistic_regression']['pr_auc_mean']:.4f} ± {results['logistic_regression']['pr_auc_std']:.4f}")

    # Random Forest (regularized)
    print("\n=== Random Forest ===")
    rf_model = RandomForestClassifier(
        n_estimators=30,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    results["random_forest"] = train_model(rf_model, X_train, y_train, patient_ids_train)
    print(f"ROC-AUC: {results['random_forest']['roc_auc_mean']:.4f} ± {results['random_forest']['roc_auc_std']:.4f}")
    print(f"PR-AUC: {results['random_forest']['pr_auc_mean']:.4f} ± {results['random_forest']['pr_auc_std']:.4f}")

    # Gradient Boosting
    print("\n=== Gradient Boosting ===")
    gb_model = GradientBoostingClassifier(
        n_estimators=30,
        max_depth=3,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
    )
    results["gradient_boosting"] = train_model(gb_model, X_train, y_train, patient_ids_train)
    print(f"ROC-AUC: {results['gradient_boosting']['roc_auc_mean']:.4f} ± {results['gradient_boosting']['roc_auc_std']:.4f}")
    print(f"PR-AUC: {results['gradient_boosting']['pr_auc_mean']:.4f} ± {results['gradient_boosting']['pr_auc_std']:.4f}")

    # AdaBoost
    print("\n=== AdaBoost ===")
    ada_model = AdaBoostClassifier(
        n_estimators=30,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
    )
    results["adaboost"] = train_model(ada_model, X_train, y_train, patient_ids_train)
    print(f"ROC-AUC: {results['adaboost']['roc_auc_mean']:.4f} ± {results['adaboost']['roc_auc_std']:.4f}")
    print(f"PR-AUC: {results['adaboost']['pr_auc_mean']:.4f} ± {results['adaboost']['pr_auc_std']:.4f}")

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
