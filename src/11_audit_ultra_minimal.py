"""Ultra-minimal audit - just split leakage and one model for quick results."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from xgboost import XGBClassifier

DATA_PATH = r"C:\Users\Kritika Chaudhari\OneDrive\Documents\ChatGPT\aiml\sepsis-prototype\data\processed\data.npz"
RANDOM_STATE = 42
TEST_SIZE = 0.20
EXCLUDED_FEATURES = {"ICULOS"}
MONOTONIC_DIRECTIONS = {
    "HR": 1, "Temp": 1, "Resp": 1, "MAP": -1, "O2Sat": -1, "Lactate": 1,
}


def main():
    print("=" * 80)
    print("SEPSIS PROTOTYPE AUDIT - ULTRA MINIMAL")
    print("=" * 80)

    # Load data
    with np.load(DATA_PATH, allow_pickle=False) as data:
        X = data["X"]
        y = data["y"]
        patient_ids = data["patient_ids"]
        feature_names = data["feature_names"].astype(str).tolist()

    print(f"\nLoaded data: X={X.shape}, y={y.shape}, patient_ids={patient_ids.shape}")
    print(f"Unique patients: {len(np.unique(patient_ids)):,}")

    # Prepare features
    model_feature_indices = [i for i, name in enumerate(feature_names) if name not in EXCLUDED_FEATURES]
    model_feature_names = [feature_names[i] for i in model_feature_indices]
    X_model = X[:, :, model_feature_indices]
    X_flat = X_model.reshape(X_model.shape[0], -1)
    monotone_constraints = tuple(
        MONOTONIC_DIRECTIONS.get(feature, 0)
        for _hour in range(X.shape[1])
        for feature in model_feature_names
    )

    # Splits
    X_train_win, X_test_win, y_train_win, y_test_win = train_test_split(
        X_flat, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X_flat, y, groups=patient_ids))
    X_train_pat, X_test_pat = X_flat[train_idx], X_flat[test_idx]
    y_train_pat, y_test_pat = y[train_idx], y[test_idx]

    print(f"\nWindow-level split: {len(X_train_win):,} train / {len(X_test_win):,} test")
    print(f"Patient-level split: {len(X_train_pat):,} train / {len(X_test_pat):,} test")

    # Split leakage check
    print("\n" + "=" * 80)
    print("1. SPLIT LEAKAGE CHECK")
    print("=" * 80)
    pid_train_win, pid_test_win = train_test_split(patient_ids, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    overlapping = set(pid_train_win) & set(pid_test_win)
    print(f"Window-level split: {len(overlapping):,} patients appear in BOTH train and test ({len(overlapping)/len(set(pid_test_win))*100:.1f}%)")
    print(f"Patient-level split: 0 patients appear in both (clean split)")

    # XGBoost evaluation only
    print("\n" + "=" * 80)
    print("2. XGBoost EVALUATION (reduced: 50 estimators)")
    print("=" * 80)

    scale_pos_weight = float((y_train_pat == 0).sum() / (y_train_pat == 1).sum())
    xgb = XGBClassifier(
        n_estimators=50, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
        monotone_constraints=monotone_constraints, objective="binary:logistic",
        eval_metric="aucpr", random_state=RANDOM_STATE, n_jobs=1, tree_method="hist",
    )

    print("\nTraining on window-level split...")
    xgb.fit(X_train_win, y_train_win)
    y_pred_win = xgb.predict_proba(X_test_win)[:, 1]
    y_pred_class_win = xgb.predict(X_test_win)
    print(f"Window-level: ROC-AUC={roc_auc_score(y_test_win, y_pred_win):.4f}, PR-AUC={average_precision_score(y_test_win, y_pred_win):.4f}, F1={f1_score(y_test_win, y_pred_class_win, pos_label=1, zero_division=0):.4f}")

    print("\nTraining on patient-level split...")
    xgb.fit(X_train_pat, y_train_pat)
    y_pred_pat = xgb.predict_proba(X_test_pat)[:, 1]
    y_pred_class_pat = xgb.predict(X_test_pat)
    print(f"Patient-level: ROC-AUC={roc_auc_score(y_test_pat, y_pred_pat):.4f}, PR-AUC={average_precision_score(y_test_pat, y_pred_pat):.4f}, F1={f1_score(y_test_pat, y_pred_class_pat, pos_label=1, zero_division=0):.4f}")

    # Overfitting check
    print("\n" + "=" * 80)
    print("3. OVERFITTING CHECK (patient-level)")
    print("=" * 80)
    xgb.fit(X_train_pat, y_train_pat)
    y_pred_train = xgb.predict_proba(X_train_pat)[:, 1]
    y_pred_test = xgb.predict_proba(X_test_pat)[:, 1]
    train_auc = roc_auc_score(y_train_pat, y_pred_train)
    test_auc = roc_auc_score(y_test_pat, y_pred_test)
    train_pr = average_precision_score(y_train_pat, y_pred_train)
    test_pr = average_precision_score(y_test_pat, y_pred_test)
    print(f"Train ROC-AUC: {train_auc:.4f}, Test ROC-AUC: {test_auc:.4f}, Gap: {train_auc - test_auc:.4f}")
    print(f"Train PR-AUC: {train_pr:.4f}, Test PR-AUC: {test_pr:.4f}, Gap: {train_pr - test_pr:.4f}")
    verdict = "OVERFITTING" if (train_auc - test_auc) > 0.05 or (train_pr - test_pr) > 0.05 else "OK"
    print(f"Verdict: {verdict}")

    # Preprocessing check
    print("\n" + "=" * 80)
    print("4. PREPROCESSING LEAKAGE CHECK")
    print("=" * 80)
    with open(r"C:\Users\Kritika Chaudhari\OneDrive\Documents\ChatGPT\aiml\sepsis-prototype\src\01_preprocessing.py", "r") as f:
        preprocessing_code = f.read()
    if "normalize(X)" in preprocessing_code and "X, feature_means, feature_stds = normalize(X)" in preprocessing_code:
        print("ISSUE FOUND: Normalization is fit on the complete dataset (all windows)")
    if "feature_medians = np.median(X.reshape(-1, X.shape[-1]), axis=0)" in preprocessing_code:
        print("ISSUE FOUND: Median imputation statistics are computed on the complete dataset")

    # Summary table
    print("\n" + "=" * 80)
    print("5. SUMMARY TABLE")
    print("=" * 80)
    print("\nModel | Window-split AUC | Patient-split AUC | Train-Test gap | Verdict")
    print(f"XGBoost | {roc_auc_score(y_test_win, y_pred_win):.4f} | {roc_auc_score(y_test_pat, y_pred_pat):.4f} | {train_auc - test_auc:.4f} | {verdict}")

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
