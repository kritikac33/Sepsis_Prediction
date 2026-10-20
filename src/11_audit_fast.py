"""Fast audit - reduced parameters for quick diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "data.npz"
RANDOM_STATE = 42
TEST_SIZE = 0.20
EXCLUDED_FEATURES = {"ICULOS"}
MONOTONIC_DIRECTIONS = {
    "HR": 1, "Temp": 1, "Resp": 1, "MAP": -1, "O2Sat": -1, "Lactate": 1,
}


def load_data():
    with np.load(DATA_PATH, allow_pickle=False) as data:
        X = data["X"]
        y = data["y"]
        patient_ids = data["patient_ids"]
        feature_names = data["feature_names"].astype(str).tolist()
    return X, y, patient_ids, feature_names


def prepare_features(X, feature_names):
    model_feature_indices = [i for i, name in enumerate(feature_names) if name not in EXCLUDED_FEATURES]
    model_feature_names = [feature_names[i] for i in model_feature_indices]
    X_model = X[:, :, model_feature_indices]
    X_flat = X_model.reshape(X_model.shape[0], -1)
    monotone_constraints = tuple(
        MONOTONIC_DIRECTIONS.get(feature, 0)
        for _hour in range(X.shape[1])
        for feature in model_feature_names
    )
    return X_flat, model_feature_names, monotone_constraints


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        probabilities = model.decision_function(X_test)
    return {
        "ROC-AUC": float(roc_auc_score(y_test, probabilities)),
        "PR-AUC": float(average_precision_score(y_test, probabilities)),
        "F1 (positive)": float(f1_score(y_test, predictions, pos_label=1, zero_division=0)),
        "Precision": float(precision_score(y_test, predictions, pos_label=1, zero_division=0)),
        "Recall": float(recall_score(y_test, predictions, pos_label=1, zero_division=0)),
    }


def evaluate_model_train_test(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    train_prob = model.predict_proba(X_train)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_train)
    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    return {
        "train": {
            "ROC-AUC": float(roc_auc_score(y_train, train_prob)),
            "PR-AUC": float(average_precision_score(y_train, train_prob)),
            "F1 (positive)": float(f1_score(y_train, train_pred, pos_label=1, zero_division=0)),
        },
        "test": {
            "ROC-AUC": float(roc_auc_score(y_test, test_prob)),
            "PR-AUC": float(average_precision_score(y_test, test_prob)),
            "F1 (positive)": float(f1_score(y_test, test_pred, pos_label=1, zero_division=0)),
        },
    }


def main():
    print("=" * 80)
    print("SEPSIS PROTOTYPE AUDIT - FAST VERSION (reduced parameters)")
    print("=" * 80)

    X, y, patient_ids, feature_names = load_data()
    print(f"\nLoaded data: X={X.shape}, y={y.shape}, patient_ids={patient_ids.shape}")
    print(f"Unique patients: {len(np.unique(patient_ids)):,}")

    X_flat, model_feature_names, monotone_constraints = prepare_features(X, feature_names)

    # Window-level split
    X_train_win, X_test_win, y_train_win, y_test_win = train_test_split(
        X_flat, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Patient-level split
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X_flat, y, groups=patient_ids))
    X_train_pat, X_test_pat = X_flat[train_idx], X_flat[test_idx]
    y_train_pat, y_test_pat = y[train_idx], y[test_idx]

    print(f"\nWindow-level split: {len(X_train_win):,} train / {len(X_test_win):,} test")
    print(f"Patient-level split: {len(X_train_pat):,} train / {len(X_test_pat):,} test")

    # Check split leakage
    print("\n" + "=" * 80)
    print("1. SPLIT LEAKAGE CHECK")
    print("=" * 80)
    pid_train_win, pid_test_win = train_test_split(patient_ids, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    overlapping = set(pid_train_win) & set(pid_test_win)
    print(f"Window-level split: {len(overlapping):,} patients appear in BOTH train and test ({len(overlapping)/len(set(pid_test_win))*100:.1f}%)")
    print(f"Patient-level split: 0 patients appear in both (clean split)")

    # Evaluate all models with REDUCED parameters
    print("\n" + "=" * 80)
    print("2. PATIENT-LEVEL RE-EVALUATION (reduced parameters for speed)")
    print("=" * 80)

    results = {}

    # XGBoost - reduced estimators
    print("\n--- XGBoost (reduced: 100 estimators, no calibration) ---")
    scale_pos_weight = float((y_train_pat == 0).sum() / (y_train_pat == 1).sum())
    xgb = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
        monotone_constraints=monotone_constraints, objective="binary:logistic",
        eval_metric="aucpr", random_state=RANDOM_STATE, n_jobs=1, tree_method="hist",
    )
    xgb_win = evaluate_model(xgb, X_train_win, X_test_win, y_train_win, y_test_win)
    xgb_pat = evaluate_model(xgb, X_train_pat, X_test_pat, y_train_pat, y_test_pat)
    print(f"Window-level: ROC-AUC={xgb_win['ROC-AUC']:.4f}, PR-AUC={xgb_win['PR-AUC']:.4f}, F1={xgb_win['F1 (positive)']:.4f}")
    print(f"Patient-level: ROC-AUC={xgb_pat['ROC-AUC']:.4f}, PR-AUC={xgb_pat['PR-AUC']:.4f}, F1={xgb_pat['F1 (positive)']:.4f}")
    results["XGBoost"] = {"window_level": xgb_win, "patient_level": xgb_pat}

    # Random Forest - reduced estimators
    print("\n--- Random Forest (reduced: 50 estimators) ---")
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=RANDOM_STATE, n_jobs=1, class_weight="balanced")
    rf_win = evaluate_model(rf, X_train_win, X_test_win, y_train_win, y_test_win)
    rf_pat = evaluate_model(rf, X_train_pat, X_test_pat, y_train_pat, y_test_pat)
    print(f"Window-level: ROC-AUC={rf_win['ROC-AUC']:.4f}, PR-AUC={rf_win['PR-AUC']:.4f}, F1={rf_win['F1 (positive)']:.4f}")
    print(f"Patient-level: ROC-AUC={rf_pat['ROC-AUC']:.4f}, PR-AUC={rf_pat['PR-AUC']:.4f}, F1={rf_pat['F1 (positive)']:.4f}")
    results["RandomForest"] = {"window_level": rf_win, "patient_level": rf_pat}

    # Gradient Boosting - already reduced
    print("\n--- Gradient Boosting (50 estimators) ---")
    gb = GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=RANDOM_STATE)
    gb_win = evaluate_model(gb, X_train_win, X_test_win, y_train_win, y_test_win)
    gb_pat = evaluate_model(gb, X_train_pat, X_test_pat, y_train_pat, y_test_pat)
    print(f"Window-level: ROC-AUC={gb_win['ROC-AUC']:.4f}, PR-AUC={gb_win['PR-AUC']:.4f}, F1={gb_win['F1 (positive)']:.4f}")
    print(f"Patient-level: ROC-AUC={gb_pat['ROC-AUC']:.4f}, PR-AUC={gb_pat['PR-AUC']:.4f}, F1={gb_pat['F1 (positive)']:.4f}")
    results["GradientBoosting"] = {"window_level": gb_win, "patient_level": gb_pat}

    # AdaBoost - already reduced
    print("\n--- AdaBoost (100 estimators) ---")
    tree = DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE)
    ada = AdaBoostClassifier(estimator=tree, n_estimators=100, random_state=RANDOM_STATE)
    ada_win = evaluate_model(ada, X_train_win, X_test_win, y_train_win, y_test_win)
    ada_pat = evaluate_model(ada, X_train_pat, X_test_pat, y_train_pat, y_test_pat)
    print(f"Window-level: ROC-AUC={ada_win['ROC-AUC']:.4f}, PR-AUC={ada_win['PR-AUC']:.4f}, F1={ada_win['F1 (positive)']:.4f}")
    print(f"Patient-level: ROC-AUC={ada_pat['ROC-AUC']:.4f}, PR-AUC={ada_pat['PR-AUC']:.4f}, F1={ada_pat['F1 (positive)']:.4f}")
    results["AdaBoost"] = {"window_level": ada_win, "patient_level": ada_pat}

    # Overfitting diagnostics
    print("\n" + "=" * 80)
    print("3. OVERFITTING / UNDERFITTING DIAGNOSTICS (patient-level split)")
    print("=" * 80)

    overfitting_results = {}

    print("\n--- XGBoost ---")
    xgb_metrics = evaluate_model_train_test(xgb, X_train_pat, X_test_pat, y_train_pat, y_test_pat)
    train_auc, test_auc = xgb_metrics["train"]["ROC-AUC"], xgb_metrics["test"]["ROC-AUC"]
    train_pr, test_pr = xgb_metrics["train"]["PR-AUC"], xgb_metrics["test"]["PR-AUC"]
    print(f"Train ROC-AUC: {train_auc:.4f}, Test ROC-AUC: {test_auc:.4f}, Gap: {train_auc - test_auc:.4f}")
    print(f"Train PR-AUC: {train_pr:.4f}, Test PR-AUC: {test_pr:.4f}, Gap: {train_pr - test_pr:.4f}")
    verdict = "OVERFITTING" if (train_auc - test_auc) > 0.05 or (train_pr - test_pr) > 0.05 else "OK"
    print(f"Verdict: {verdict}")
    overfitting_results["XGBoost"] = {"metrics": xgb_metrics, "verdict": verdict}

    print("\n--- Random Forest ---")
    rf_metrics = evaluate_model_train_test(rf, X_train_pat, X_test_pat, y_train_pat, y_test_pat)
    train_auc, test_auc = rf_metrics["train"]["ROC-AUC"], rf_metrics["test"]["ROC-AUC"]
    train_pr, test_pr = rf_metrics["train"]["PR-AUC"], rf_metrics["test"]["PR-AUC"]
    print(f"Train ROC-AUC: {train_auc:.4f}, Test ROC-AUC: {test_auc:.4f}, Gap: {train_auc - test_auc:.4f}")
    print(f"Train PR-AUC: {train_pr:.4f}, Test PR-AUC: {test_pr:.4f}, Gap: {train_pr - test_pr:.4f}")
    verdict = "OVERFITTING" if (train_auc - test_auc) > 0.05 or (train_pr - test_pr) > 0.05 else "OK"
    print(f"Verdict: {verdict}")
    overfitting_results["RandomForest"] = {"metrics": rf_metrics, "verdict": verdict}

    print("\n--- Gradient Boosting ---")
    gb_metrics = evaluate_model_train_test(gb, X_train_pat, X_test_pat, y_train_pat, y_test_pat)
    train_auc, test_auc = gb_metrics["train"]["ROC-AUC"], gb_metrics["test"]["ROC-AUC"]
    train_pr, test_pr = gb_metrics["train"]["PR-AUC"], gb_metrics["test"]["PR-AUC"]
    print(f"Train ROC-AUC: {train_auc:.4f}, Test ROC-AUC: {test_auc:.4f}, Gap: {train_auc - test_auc:.4f}")
    print(f"Train PR-AUC: {train_pr:.4f}, Test PR-AUC: {test_pr:.4f}, Gap: {train_pr - test_pr:.4f}")
    verdict = "OVERFITTING" if (train_auc - test_auc) > 0.05 or (train_pr - test_pr) > 0.05 else "OK"
    print(f"Verdict: {verdict}")
    overfitting_results["GradientBoosting"] = {"metrics": gb_metrics, "verdict": verdict}

    print("\n--- AdaBoost ---")
    ada_metrics = evaluate_model_train_test(ada, X_train_pat, X_test_pat, y_train_pat, y_test_pat)
    train_auc, test_auc = ada_metrics["train"]["ROC-AUC"], ada_metrics["test"]["ROC-AUC"]
    train_pr, test_pr = ada_metrics["train"]["PR-AUC"], ada_metrics["test"]["PR-AUC"]
    print(f"Train ROC-AUC: {train_auc:.4f}, Test ROC-AUC: {test_auc:.4f}, Gap: {train_auc - test_auc:.4f}")
    print(f"Train PR-AUC: {train_pr:.4f}, Test PR-AUC: {test_pr:.4f}, Gap: {train_pr - test_pr:.4f}")
    verdict = "OVERFITTING" if (train_auc - test_auc) > 0.05 or (train_pr - test_pr) > 0.05 else "OK"
    print(f"Verdict: {verdict}")
    overfitting_results["AdaBoost"] = {"metrics": ada_metrics, "verdict": verdict}

    # Baseline comparison
    print("\n" + "=" * 80)
    print("4. BASELINE SANITY")
    print("=" * 80)

    print("\n--- Dummy Classifier (majority class) ---")
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy_metrics = evaluate_model(dummy, X_train_pat, X_test_pat, y_train_pat, y_test_pat)
    print(f"ROC-AUC: {dummy_metrics['ROC-AUC']:.4f}, PR-AUC: {dummy_metrics['PR-AUC']:.4f}, F1: {dummy_metrics['F1 (positive)']:.4f}")

    print("\n--- Logistic Regression ---")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
    lr_metrics = evaluate_model(lr, X_train_pat, X_test_pat, y_train_pat, y_test_pat)
    print(f"ROC-AUC: {lr_metrics['ROC-AUC']:.4f}, PR-AUC: {lr_metrics['PR-AUC']:.4f}, F1: {lr_metrics['F1 (positive)']:.4f}")

    # Preprocessing leakage check
    print("\n" + "=" * 80)
    print("5. LEAKAGE IN FEATURES (PREPROCESSING)")
    print("=" * 80)
    with open(Path(__file__).resolve().parents[1] / "src" / "01_preprocessing.py", "r") as f:
        preprocessing_code = f.read()
    if "normalize(X)" in preprocessing_code and "X, feature_means, feature_stds = normalize(X)" in preprocessing_code:
        print("ISSUE FOUND: Normalization is fit on the complete dataset (all windows)")
    if "feature_medians = np.median(X.reshape(-1, X.shape[-1]), axis=0)" in preprocessing_code:
        print("ISSUE FOUND: Median imputation statistics are computed on the complete dataset")

    # Final verdict table
    print("\n" + "=" * 80)
    print("6. FINAL VERDICT TABLE")
    print("=" * 80)

    verdict_table = []
    for model_name in ["XGBoost", "RandomForest", "GradientBoosting", "AdaBoost"]:
        window_auc = results[model_name]["window_level"]["ROC-AUC"]
        patient_auc = results[model_name]["patient_level"]["ROC-AUC"]
        train_auc = overfitting_results[model_name]["metrics"]["train"]["ROC-AUC"]
        test_auc = overfitting_results[model_name]["metrics"]["test"]["ROC-AUC"]
        gap = train_auc - test_auc
        verdict = overfitting_results[model_name]["verdict"]
        verdict_table.append({
            "Model": model_name,
            "Window-split AUC": window_auc,
            "Patient-split AUC": patient_auc,
            "Train-Test gap": gap,
            "Verdict": verdict,
        })

    df = pd.DataFrame(verdict_table)
    print("\n" + df.to_string(index=False, float_format="%.4f"))

    for row in verdict_table:
        if row["Window-split AUC"] - row["Patient-split AUC"] > 0.10:
            print(f"\nWARNING: {row['Model']} patient-level AUC ({row['Patient-split AUC']:.4f}) is much lower than window-level ({row['Window-split AUC']:.4f})")
            print(f"  This indicates significant data leakage in the window-level split.")
            print(f"  Report the patient-level numbers: {row['Patient-split AUC']:.4f}")

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE (fast version - reduced parameters, no permutation test)")
    print("=" * 80)


if __name__ == "__main__":
    main()
