"""Comprehensive audit for data leakage, overfitting, and underfitting.

This script performs a read-and-measure pass to audit the sepsis prototype:
1. Checks split leakage (window vs patient-level)
2. Re-evaluates all models with patient-level splits
3. Diagnoses overfitting/underfitting
4. Checks preprocessing leakage
5. Runs baseline comparisons
6. Generates learning/validation curves
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib"))

import matplotlib.pyplot as plt
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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, learning_curve, validation_curve
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "data.npz"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
PLOT_DIR = PROJECT_ROOT / "outputs" / "plots"
AUDIT_REPORT_PATH = PROJECT_ROOT / "outputs" / "audit_report.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
EXCLUDED_FEATURES = {"ICULOS"}
MONOTONIC_DIRECTIONS = {
    "HR": 1,
    "Temp": 1,
    "Resp": 1,
    "MAP": -1,
    "O2Sat": -1,
    "Lactate": 1,
}


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load preprocessed data with patient IDs."""
    with np.load(DATA_PATH, allow_pickle=False) as data:
        X = data["X"]
        y = data["y"]
        patient_ids = data["patient_ids"]
        feature_names = data["feature_names"].astype(str).tolist()
    return X, y, patient_ids, feature_names


def check_split_leakage(X: np.ndarray, y: np.ndarray, patient_ids: np.ndarray) -> dict:
    """Check if train/test split is by window or by patient."""
    print("\n" + "=" * 80)
    print("1. SPLIT LEAKAGE CHECK")
    print("=" * 80)

    from sklearn.model_selection import train_test_split

    # Current implementation: window-level split (random rows)
    X_train_win, X_test_win, y_train_win, y_test_win, pid_train_win, pid_test_win = train_test_split(
        X, y, patient_ids, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    unique_train_patients = set(pid_train_win)
    unique_test_patients = set(pid_test_win)
    overlapping_patients = unique_train_patients & unique_test_patients

    print(f"Total windows: {len(X):,}")
    print(f"Total unique patients: {len(np.unique(patient_ids)):,}")
    print(f"\nWindow-level split (current implementation):")
    print(f"  Train windows: {len(X_train_win):,}")
    print(f"  Test windows: {len(X_test_win):,}")
    print(f"  Unique patients in train: {len(unique_train_patients):,}")
    print(f"  Unique patients in test: {len(unique_test_patients):,}")
    print(f"  Patients appearing in BOTH train and test: {len(overlapping_patients):,}")
    print(f"  Percentage of test windows with patient also in train: {len(overlapping_patients) / len(unique_test_patients) * 100:.2f}%")

    leakage_report = {
        "split_type": "WINDOW (random rows)",
        "overlapping_patients": int(len(overlapping_patients)),
        "unique_test_patients": int(len(unique_test_patients)),
        "percentage_test_windows_leaked": float(len(overlapping_patients) / len(unique_test_patients) * 100),
    }

    # Patient-level split
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=patient_ids))
    X_train_pat, X_test_pat = X[train_idx], X[test_idx]
    y_train_pat, y_test_pat = y[train_idx], y[test_idx]
    pid_train_pat, pid_test_pat = patient_ids[train_idx], patient_ids[test_idx]

    unique_train_patients_pat = set(pid_train_pat)
    unique_test_patients_pat = set(pid_test_pat)
    overlapping_patients_pat = unique_train_patients_pat & unique_test_patients_pat

    print(f"\nPatient-level split (proposed):")
    print(f"  Train windows: {len(X_train_pat):,}")
    print(f"  Test windows: {len(X_test_pat):,}")
    print(f"  Unique patients in train: {len(unique_train_patients_pat):,}")
    print(f"  Unique patients in test: {len(unique_test_patients_pat):,}")
    print(f"  Patients appearing in BOTH train and test: {len(overlapping_patients_pat):,}")
    print(f"  Percentage of test windows with patient also in train: {len(overlapping_patients_pat) / len(unique_test_patients_pat) * 100:.2f}%")

    leakage_report["patient_split"] = {
        "train_windows": int(len(X_train_pat)),
        "test_windows": int(len(X_test_pat)),
        "unique_train_patients": int(len(unique_train_patients_pat)),
        "unique_test_patients": int(len(unique_test_patients_pat)),
        "overlapping_patients": int(len(overlapping_patients_pat)),
    }

    return leakage_report


def prepare_features(X: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, list[str], tuple]:
    """Prepare features by excluding ICULOS and creating monotone constraints."""
    model_feature_indices = [
        index for index, name in enumerate(feature_names) if name not in EXCLUDED_FEATURES
    ]
    model_feature_names = [feature_names[index] for index in model_feature_indices]
    X_model = X[:, :, model_feature_indices]
    X_flat = X_model.reshape(X_model.shape[0], -1)
    monotone_constraints = tuple(
        MONOTONIC_DIRECTIONS.get(feature, 0)
        for _hour in range(X.shape[1])
        for feature in model_feature_names
    )
    return X_flat, model_feature_names, monotone_constraints


def evaluate_model(
    model, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
) -> dict:
    """Evaluate a model and return metrics."""
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


def evaluate_model_train_test(
    model, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
) -> dict:
    """Evaluate a model on both train and test sets for overfitting check."""
    model.fit(X_train, y_train)

    # Train metrics
    train_predictions = model.predict(X_train)
    if hasattr(model, "predict_proba"):
        train_probabilities = model.predict_proba(X_train)[:, 1]
    else:
        train_probabilities = model.decision_function(X_train)

    # Test metrics
    test_predictions = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        test_probabilities = model.predict_proba(X_test)[:, 1]
    else:
        test_probabilities = model.decision_function(X_test)

    return {
        "train": {
            "ROC-AUC": float(roc_auc_score(y_train, train_probabilities)),
            "PR-AUC": float(average_precision_score(y_train, train_probabilities)),
            "F1 (positive)": float(f1_score(y_train, train_predictions, pos_label=1, zero_division=0)),
        },
        "test": {
            "ROC-AUC": float(roc_auc_score(y_test, test_probabilities)),
            "PR-AUC": float(average_precision_score(y_test, test_probabilities)),
            "F1 (positive)": float(f1_score(y_test, test_predictions, pos_label=1, zero_division=0)),
        },
    }


def patient_level_evaluation(
    X: np.ndarray, y: np.ndarray, patient_ids: np.ndarray, feature_names: list[str]
) -> dict:
    """Evaluate all models with patient-level splits."""
    print("\n" + "=" * 80)
    print("2. PATIENT-LEVEL RE-EVALUATION")
    print("=" * 80)

    X_flat, model_feature_names, monotone_constraints = prepare_features(X, feature_names)

    # Window-level split (current)
    from sklearn.model_selection import train_test_split

    X_train_win, X_test_win, y_train_win, y_test_win = train_test_split(
        X_flat, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Patient-level split
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X_flat, y, groups=patient_ids))
    X_train_pat, X_test_pat = X_flat[train_idx], X_flat[test_idx]
    y_train_pat, y_test_pat = y[train_idx], y[test_idx]

    print(f"Window-level split: {len(X_train_win):,} train / {len(X_test_win):,} test")
    print(f"Patient-level split: {len(X_train_pat):,} train / {len(X_test_pat):,} test")

    results = {}

    # XGBoost
    print("\n--- XGBoost ---")
    scale_pos_weight = float((y_train_pat == 0).sum() / (y_train_pat == 1).sum())
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        monotone_constraints=monotone_constraints,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
    )
    xgb_cal = CalibratedClassifierCV(estimator=xgb, method="sigmoid", cv=5, n_jobs=1)

    xgb_win = evaluate_model(xgb_cal, X_train_win, X_test_win, y_train_win, y_test_win)
    xgb_pat = evaluate_model(xgb_cal, X_train_pat, X_test_pat, y_train_pat, y_test_pat)

    print(f"Window-level: ROC-AUC={xgb_win['ROC-AUC']:.4f}, PR-AUC={xgb_win['PR-AUC']:.4f}, F1={xgb_win['F1 (positive)']:.4f}")
    print(f"Patient-level: ROC-AUC={xgb_pat['ROC-AUC']:.4f}, PR-AUC={xgb_pat['PR-AUC']:.4f}, F1={xgb_pat['F1 (positive)']:.4f}")

    results["XGBoost"] = {"window_level": xgb_win, "patient_level": xgb_pat}

    # Random Forest
    print("\n--- Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=1, class_weight="balanced"
    )
    rf_win = evaluate_model(rf, X_train_win, X_test_win, y_train_win, y_test_win)
    rf_pat = evaluate_model(rf, X_train_pat, X_test_pat, y_train_pat, y_test_pat)

    print(f"Window-level: ROC-AUC={rf_win['ROC-AUC']:.4f}, PR-AUC={rf_win['PR-AUC']:.4f}, F1={rf_win['F1 (positive)']:.4f}")
    print(f"Patient-level: ROC-AUC={rf_pat['ROC-AUC']:.4f}, PR-AUC={rf_pat['PR-AUC']:.4f}, F1={rf_pat['F1 (positive)']:.4f}")

    results["RandomForest"] = {"window_level": rf_win, "patient_level": rf_pat}

    # Gradient Boosting
    print("\n--- Gradient Boosting ---")
    gb = GradientBoostingClassifier(
        n_estimators=50, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=RANDOM_STATE
    )
    gb_win = evaluate_model(gb, X_train_win, X_test_win, y_train_win, y_test_win)
    gb_pat = evaluate_model(gb, X_train_pat, X_test_pat, y_train_pat, y_test_pat)

    print(f"Window-level: ROC-AUC={gb_win['ROC-AUC']:.4f}, PR-AUC={gb_win['PR-AUC']:.4f}, F1={gb_win['F1 (positive)']:.4f}")
    print(f"Patient-level: ROC-AUC={gb_pat['ROC-AUC']:.4f}, PR-AUC={gb_pat['PR-AUC']:.4f}, F1={gb_pat['F1 (positive)']:.4f}")

    results["GradientBoosting"] = {"window_level": gb_win, "patient_level": gb_pat}

    # AdaBoost
    print("\n--- AdaBoost ---")
    tree = DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE)
    ada = AdaBoostClassifier(estimator=tree, n_estimators=100, random_state=RANDOM_STATE)
    ada_win = evaluate_model(ada, X_train_win, X_test_win, y_train_win, y_test_win)
    ada_pat = evaluate_model(ada, X_train_pat, X_test_pat, y_train_pat, y_test_pat)

    print(f"Window-level: ROC-AUC={ada_win['ROC-AUC']:.4f}, PR-AUC={ada_win['PR-AUC']:.4f}, F1={ada_win['F1 (positive)']:.4f}")
    print(f"Patient-level: ROC-AUC={ada_pat['ROC-AUC']:.4f}, PR-AUC={ada_pat['PR-AUC']:.4f}, F1={ada_pat['F1 (positive)']:.4f}")

    results["AdaBoost"] = {"window_level": ada_win, "patient_level": ada_pat}

    return results


def overfitting_diagnostics(
    X: np.ndarray, y: np.ndarray, patient_ids: np.ndarray, feature_names: list[str]
) -> dict:
    """Check for overfitting/underfitting using patient-level split."""
    print("\n" + "=" * 80)
    print("3. OVERFITTING / UNDERFITTING DIAGNOSTICS (patient-level split)")
    print("=" * 80)

    X_flat, model_feature_names, monotone_constraints = prepare_features(X, feature_names)

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X_flat, y, groups=patient_ids))
    X_train, X_test = X_flat[train_idx], X_flat[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    results = {}

    # XGBoost
    print("\n--- XGBoost ---")
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        monotone_constraints=monotone_constraints,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
    )
    xgb_cal = CalibratedClassifierCV(estimator=xgb, method="sigmoid", cv=5, n_jobs=1)
    xgb_metrics = evaluate_model_train_test(xgb_cal, X_train, X_test, y_train, y_test)

    train_auc = xgb_metrics["train"]["ROC-AUC"]
    test_auc = xgb_metrics["test"]["ROC-AUC"]
    train_pr = xgb_metrics["train"]["PR-AUC"]
    test_pr = xgb_metrics["test"]["PR-AUC"]
    train_f1 = xgb_metrics["train"]["F1 (positive)"]
    test_f1 = xgb_metrics["test"]["F1 (positive)"]

    auc_gap = train_auc - test_auc
    pr_gap = train_pr - test_pr

    print(f"Train ROC-AUC: {train_auc:.4f}, Test ROC-AUC: {test_auc:.4f}, Gap: {auc_gap:.4f}")
    print(f"Train PR-AUC: {train_pr:.4f}, Test PR-AUC: {test_pr:.4f}, Gap: {pr_gap:.4f}")
    print(f"Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}, Gap: {train_f1 - test_f1:.4f}")

    verdict = "OK"
    if auc_gap > 0.05 or pr_gap > 0.05:
        verdict = "OVERFITTING"
    elif test_auc < 0.6 and test_pr < 0.3:
        verdict = "UNDERFITTING"

    print(f"Verdict: {verdict}")
    results["XGBoost"] = {"metrics": xgb_metrics, "verdict": verdict}

    # Random Forest
    print("\n--- Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=1, class_weight="balanced"
    )
    rf_metrics = evaluate_model_train_test(rf, X_train, X_test, y_train, y_test)

    train_auc = rf_metrics["train"]["ROC-AUC"]
    test_auc = rf_metrics["test"]["ROC-AUC"]
    train_pr = rf_metrics["train"]["PR-AUC"]
    test_pr = rf_metrics["test"]["PR-AUC"]
    auc_gap = train_auc - test_auc
    pr_gap = train_pr - test_pr

    print(f"Train ROC-AUC: {train_auc:.4f}, Test ROC-AUC: {test_auc:.4f}, Gap: {auc_gap:.4f}")
    print(f"Train PR-AUC: {train_pr:.4f}, Test PR-AUC: {test_pr:.4f}, Gap: {pr_gap:.4f}")

    verdict = "OK"
    if auc_gap > 0.05 or pr_gap > 0.05:
        verdict = "OVERFITTING"
    elif test_auc < 0.6 and test_pr < 0.3:
        verdict = "UNDERFITTING"

    print(f"Verdict: {verdict}")
    results["RandomForest"] = {"metrics": rf_metrics, "verdict": verdict}

    # Gradient Boosting
    print("\n--- Gradient Boosting ---")
    gb = GradientBoostingClassifier(
        n_estimators=50, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=RANDOM_STATE
    )
    gb_metrics = evaluate_model_train_test(gb, X_train, X_test, y_train, y_test)

    train_auc = gb_metrics["train"]["ROC-AUC"]
    test_auc = gb_metrics["test"]["ROC-AUC"]
    train_pr = gb_metrics["train"]["PR-AUC"]
    test_pr = gb_metrics["test"]["PR-AUC"]
    auc_gap = train_auc - test_auc
    pr_gap = train_pr - test_pr

    print(f"Train ROC-AUC: {train_auc:.4f}, Test ROC-AUC: {test_auc:.4f}, Gap: {auc_gap:.4f}")
    print(f"Train PR-AUC: {train_pr:.4f}, Test PR-AUC: {test_pr:.4f}, Gap: {pr_gap:.4f}")

    verdict = "OK"
    if auc_gap > 0.05 or pr_gap > 0.05:
        verdict = "OVERFITTING"
    elif test_auc < 0.6 and test_pr < 0.3:
        verdict = "UNDERFITTING"

    print(f"Verdict: {verdict}")
    results["GradientBoosting"] = {"metrics": gb_metrics, "verdict": verdict}

    # AdaBoost
    print("\n--- AdaBoost ---")
    tree = DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE)
    ada = AdaBoostClassifier(estimator=tree, n_estimators=100, random_state=RANDOM_STATE)
    ada_metrics = evaluate_model_train_test(ada, X_train, X_test, y_train, y_test)

    train_auc = ada_metrics["train"]["ROC-AUC"]
    test_auc = ada_metrics["test"]["ROC-AUC"]
    train_pr = ada_metrics["train"]["PR-AUC"]
    test_pr = ada_metrics["test"]["PR-AUC"]
    auc_gap = train_auc - test_auc
    pr_gap = train_pr - test_pr

    print(f"Train ROC-AUC: {train_auc:.4f}, Test ROC-AUC: {test_auc:.4f}, Gap: {auc_gap:.4f}")
    print(f"Train PR-AUC: {train_pr:.4f}, Test PR-AUC: {test_pr:.4f}, Gap: {pr_gap:.4f}")

    verdict = "OK"
    if auc_gap > 0.05 or pr_gap > 0.05:
        verdict = "OVERFITTING"
    elif test_auc < 0.6 and test_pr < 0.3:
        verdict = "UNDERFITTING"

    print(f"Verdict: {verdict}")
    results["AdaBoost"] = {"metrics": ada_metrics, "verdict": verdict}

    return results


def generate_learning_curve(
    X: np.ndarray, y: np.ndarray, patient_ids: np.ndarray, feature_names: list[str]
) -> None:
    """Generate learning curve for XGBoost with GroupKFold."""
    print("\n" + "=" * 80)
    print("GENERATING LEARNING CURVE FOR XGBoost")
    print("=" * 80)

    X_flat, model_feature_names, monotone_constraints = prepare_features(X, feature_names)

    scale_pos_weight = float((y == 0).sum() / (y == 1).sum())
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        monotone_constraints=monotone_constraints,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
    )

    # Use GroupKFold for patient-level CV
    group_kfold = GroupKFold(n_splits=5)

    train_sizes = np.linspace(0.1, 1.0, 10)
    try:
        result = learning_curve(
            xgb,
            X_flat,
            y,
            groups=patient_ids,
            cv=group_kfold,
            train_sizes=train_sizes,
            scoring="average_precision",
            n_jobs=1,
            random_state=RANDOM_STATE,
            return_times=True,
        )
        # Handle both sklearn versions (some return 3, some return 4 values)
        if len(result) == 4:
            train_scores, val_scores, fit_times, _ = result
        else:
            train_scores, val_scores, fit_times = result
    except TypeError:
        # Older sklearn version doesn't support return_times
        train_scores, val_scores, fit_times = learning_curve(
            xgb,
            X_flat,
            y,
            groups=patient_ids,
            cv=group_kfold,
            train_sizes=train_sizes,
            scoring="average_precision",
            n_jobs=1,
            random_state=RANDOM_STATE,
        )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes * len(X_flat), train_mean, "o-", color="blue", label="Training PR-AUC")
    ax.fill_between(
        train_sizes * len(X_flat), train_mean - train_std, train_mean + train_std, alpha=0.2, color="blue"
    )
    ax.plot(train_sizes * len(X_flat), val_mean, "o-", color="red", label="Validation PR-AUC")
    ax.fill_between(
        train_sizes * len(X_flat), val_mean - val_std, val_mean + val_std, alpha=0.2, color="red"
    )
    ax.set_xlabel("Training set size", fontsize=12)
    ax.set_ylabel("PR-AUC (Average Precision)", fontsize=12)
    ax.set_title("XGBoost Learning Curve (Patient-Level GroupKFold)", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOT_DIR / "learning_curve_xgb.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved learning curve to {plot_path}")


def generate_validation_curve(
    X: np.ndarray, y: np.ndarray, patient_ids: np.ndarray, feature_names: list[str]
) -> None:
    """Generate validation curve for XGBoost max_depth and n_estimators."""
    print("\n" + "=" * 80)
    print("GENERATING VALIDATION CURVE FOR XGBoost")
    print("=" * 80)

    X_flat, model_feature_names, monotone_constraints = prepare_features(X, feature_names)

    scale_pos_weight = float((y == 0).sum() / (y == 1).sum())
    group_kfold = GroupKFold(n_splits=5)

    # Validation curve for max_depth
    param_range = [2, 3, 4, 5, 6, 8]
    train_scores, val_scores = validation_curve(
        XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            monotone_constraints=monotone_constraints,
            objective="binary:logistic",
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            n_jobs=1,
            tree_method="hist",
        ),
        X_flat,
        y,
        param_name="max_depth",
        param_range=param_range,
        cv=group_kfold,
        scoring="average_precision",
        groups=patient_ids,
        n_jobs=1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(param_range, train_mean, "o-", color="blue", label="Training PR-AUC")
    ax1.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2, color="blue")
    ax1.plot(param_range, val_mean, "o-", color="red", label="Validation PR-AUC")
    ax1.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.2, color="red")
    ax1.set_xlabel("max_depth", fontsize=12)
    ax1.set_ylabel("PR-AUC (Average Precision)", fontsize=12)
    ax1.set_title("XGBoost Validation Curve: max_depth", fontsize=14, fontweight="bold")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Validation curve for n_estimators
    param_range = [50, 100, 200, 300, 400, 500]
    train_scores, val_scores = validation_curve(
        XGBClassifier(
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            monotone_constraints=monotone_constraints,
            objective="binary:logistic",
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            n_jobs=1,
            tree_method="hist",
        ),
        X_flat,
        y,
        param_name="n_estimators",
        param_range=param_range,
        cv=group_kfold,
        scoring="average_precision",
        groups=patient_ids,
        n_jobs=1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    ax2.plot(param_range, train_mean, "o-", color="blue", label="Training PR-AUC")
    ax2.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2, color="blue")
    ax2.plot(param_range, val_mean, "o-", color="red", label="Validation PR-AUC")
    ax2.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.2, color="red")
    ax2.set_xlabel("n_estimators", fontsize=12)
    ax2.set_ylabel("PR-AUC (Average Precision)", fontsize=12)
    ax2.set_title("XGBoost Validation Curve: n_estimators", fontsize=14, fontweight="bold")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOT_DIR / "validation_curve_xgb.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved validation curve to {plot_path}")


def baseline_comparison(
    X: np.ndarray, y: np.ndarray, patient_ids: np.ndarray, feature_names: list[str]
) -> dict:
    """Compare with dummy and logistic regression baselines."""
    print("\n" + "=" * 80)
    print("4. BASELINE SANITY")
    print("=" * 80)

    X_flat, model_feature_names, monotone_constraints = prepare_features(X, feature_names)

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X_flat, y, groups=patient_ids))
    X_train, X_test = X_flat[train_idx], X_flat[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    results = {}

    # Dummy baseline (majority class)
    print("\n--- Dummy Classifier (majority class) ---")
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy_metrics = evaluate_model(dummy, X_train, X_test, y_train, y_test)
    print(f"ROC-AUC: {dummy_metrics['ROC-AUC']:.4f}, PR-AUC: {dummy_metrics['PR-AUC']:.4f}, F1: {dummy_metrics['F1 (positive)']:.4f}")
    results["Dummy"] = dummy_metrics

    # Logistic Regression
    print("\n--- Logistic Regression ---")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=1)
    lr_metrics = evaluate_model(lr, X_train, X_test, y_train, y_test)
    print(f"ROC-AUC: {lr_metrics['ROC-AUC']:.4f}, PR-AUC: {lr_metrics['PR-AUC']:.4f}, F1: {lr_metrics['F1 (positive)']:.4f}")
    results["LogisticRegression"] = lr_metrics

    # Permutation test
    print("\n--- Permutation Test (shuffle y 20 times) ---")
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        monotone_constraints=monotone_constraints,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
    )

    perm_aucs = []
    for i in range(20):
        y_train_shuffled = np.random.permutation(y_train)
        xgb.fit(X_train, y_train_shuffled)
        y_pred_proba = xgb.predict_proba(X_test)[:, 1]
        perm_auc = roc_auc_score(y_test, y_pred_proba)
        perm_aucs.append(perm_auc)
        if (i + 1) % 5 == 0:
            print(f"  Iteration {i+1}/20: AUC = {perm_auc:.4f}")

    mean_perm_auc = np.mean(perm_aucs)
    std_perm_auc = np.std(perm_aucs)
    print(f"Mean permutation AUC: {mean_perm_auc:.4f} ± {std_perm_auc:.4f}")

    if mean_perm_auc > 0.55:
        print("WARNING: Permutation AUC is significantly above 0.5 - possible data leakage!")
    else:
        print("OK: Permutation AUC is near 0.5, as expected.")

    results["PermutationTest"] = {"mean_auc": float(mean_perm_auc), "std_auc": float(std_perm_auc)}

    return results


def check_preprocessing_leakage() -> dict:
    """Check if normalization/imputation is fit on training data only."""
    print("\n" + "=" * 80)
    print("5. LEAKAGE IN FEATURES (PREPROCESSING)")
    print("=" * 80)

    # Read the preprocessing script
    with open(PROJECT_ROOT / "src" / "01_preprocessing.py", "r") as f:
        preprocessing_code = f.read()

    print("Checking 01_preprocessing.py...")

    leakage_found = False
    issues = []

    # Check if normalize is called on all data
    if "normalize(X)" in preprocessing_code and "X, feature_means, feature_stds = normalize(X)" in preprocessing_code:
        print("ISSUE FOUND: Normalization is fit on the complete dataset (all windows)")
        print("  This means train/test contamination - statistics leak from test to train")
        issues.append("Normalization fit on all data instead of train only")
        leakage_found = True

    # Check if median imputation is on all data
    if "feature_medians = np.median(X.reshape(-1, X.shape[-1]), axis=0)" in preprocessing_code:
        print("ISSUE FOUND: Median imputation statistics are computed on the complete dataset")
        print("  This means train/test contamination - medians leak from test to train")
        issues.append("Median imputation fit on all data instead of train only")
        leakage_found = True

    if not leakage_found:
        print("OK: No obvious preprocessing leakage detected")
    else:
        print("\nSUMMARY: Preprocessing leakage detected!")
        print("  Normalization and median imputation should be fit on TRAINING data only")
        print("  Then applied to test data using the fitted statistics")

    return {"leakage_found": leakage_found, "issues": issues}


def shap_feature_analysis(
    X: np.ndarray, y: np.ndarray, patient_ids: np.ndarray, feature_names: list[str]
) -> dict:
    """Run SHAP analysis to check for feature leakage."""
    print("\n" + "=" * 80)
    print("6. LEAKAGE IN FEATURES (SHAP ANALYSIS)")
    print("=" * 80)

    try:
        import shap
    except ImportError:
        print("SHAP not installed - skipping feature importance analysis")
        return {"shap_available": False}

    X_flat, model_feature_names, monotone_constraints = prepare_features(X, feature_names)

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X_flat, y, groups=patient_ids))
    X_train, X_test = X_flat[train_idx], X_flat[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train XGBoost on patient-level split
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        monotone_constraints=monotone_constraints,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
    )
    xgb.fit(X_train, y_train)

    # Use a subset of test data for SHAP (faster)
    shap_sample_size = min(100, len(X_test))
    X_test_sample = X_test[:shap_sample_size]

    # Create feature names for flattened features
    flattened_feature_names = [
        f"{name}_t{t}" for t in range(X.shape[1]) for name in model_feature_names
    ]

    print(f"Computing SHAP values on {shap_sample_size} test samples...")
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test_sample)

    # Get mean absolute SHAP values for each feature
    mean_shap = np.abs(shap_values).mean(axis=0)

    # Get top 10 features
    top_indices = np.argsort(mean_shap)[-10:][::-1]
    top_features = [(flattened_feature_names[i], mean_shap[i]) for i in top_indices]

    print("\nTop 10 SHAP features (patient-level model):")
    for i, (feat, val) in enumerate(top_features, 1):
        print(f"  {i}. {feat}: {val:.4f}")

    # Check for suspicious features
    suspicious = []
    for feat, val in top_features:
        if "ICULOS" in feat:
            suspicious.append((feat, val, "ICU length-of-stay - likely leakage"))
        elif "HospAdmTime" in feat:
            suspicious.append((feat, val, "Hospital admission time - potential leakage"))
        elif "Unit" in feat:
            suspicious.append((feat, val, "Unit flag - potential leakage"))

    if suspicious:
        print("\nWARNING: Potentially leaky features detected:")
        for feat, val, reason in suspicious:
            print(f"  {feat}: {val:.4f} - {reason}")
    else:
        print("\nOK: No obviously leaky features in top 10")

    return {
        "shap_available": True,
        "top_features": [(feat, float(val)) for feat, val in top_features],
        "suspicious_features": [(feat, float(val), reason) for feat, val, reason in suspicious],
    }


def main() -> None:
    print("=" * 80)
    print("SEPSIS PROTOTYPE AUDIT: DATA LEAKAGE, OVERFITTING, UNDERFITTING")
    print("=" * 80)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found at {DATA_PATH}")

    X, y, patient_ids, feature_names = load_data()
    print(f"\nLoaded data: X={X.shape}, y={y.shape}, patient_ids={patient_ids.shape}")
    print(f"Unique patients: {len(np.unique(patient_ids)):,}")

    # 1. Split leakage check
    leakage_report = check_split_leakage(X, y, patient_ids)

    # 2. Patient-level re-evaluation
    patient_results = patient_level_evaluation(X, y, patient_ids, feature_names)

    # 3. Overfitting diagnostics
    overfitting_results = overfitting_diagnostics(X, y, patient_ids, feature_names)

    # 4. Baseline comparison
    baseline_results = baseline_comparison(X, y, patient_ids, feature_names)

    # 5. Preprocessing leakage check
    preprocessing_report = check_preprocessing_leakage()

    # 6. SHAP analysis
    shap_results = shap_feature_analysis(X, y, patient_ids, feature_names)

    # 7. Generate learning and validation curves
    generate_learning_curve(X, y, patient_ids, feature_names)
    generate_validation_curve(X, y, patient_ids, feature_names)

    # 8. Compile final verdict table
    print("\n" + "=" * 80)
    print("7. FINAL VERDICT TABLE")
    print("=" * 80)

    verdict_table = []
    for model_name in ["XGBoost", "RandomForest", "GradientBoosting", "AdaBoost"]:
        window_auc = patient_results[model_name]["window_level"]["ROC-AUC"]
        patient_auc = patient_results[model_name]["patient_level"]["ROC-AUC"]
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

    # Check if patient-level numbers are much lower
    for row in verdict_table:
        if row["Window-split AUC"] - row["Patient-split AUC"] > 0.10:
            print(f"\nWARNING: {row['Model']} patient-level AUC ({row['Patient-split AUC']:.4f}) is much lower than window-level ({row['Window-split AUC']:.4f})")
            print(f"  This indicates significant data leakage in the window-level split.")
            print(f"  Report the patient-level numbers: {row['Patient-split AUC']:.4f}")

    # Compile full report
    full_report = {
        "split_leakage": leakage_report,
        "patient_level_evaluation": patient_results,
        "overfitting_diagnostics": overfitting_results,
        "baseline_comparison": baseline_results,
        "preprocessing_leakage": preprocessing_report,
        "shap_analysis": shap_results,
        "final_verdict_table": verdict_table,
    }

    AUDIT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_REPORT_PATH.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(f"\nFull audit report saved to {AUDIT_REPORT_PATH}")

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
