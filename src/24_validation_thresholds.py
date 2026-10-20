"""Compute validation-set-based risk thresholds with operating characteristics.

Creates train/validation/test split with no patient overlap.
Selects thresholds based on sensitivity, specificity, precision on validation set.
Marks thresholds as provisional/experimental (not clinically validated).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_V2_PATH = PROJECT_ROOT / "data" / "processed" / "features_v2.npz"
SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "split.json"
THRESHOLDS_PATH = PROJECT_ROOT / "data" / "processed" / "validation_thresholds.json"

XGB_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "xgboost_v2.pkl"
RF_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "random_forest_v2.pkl"
GB_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "gradient_boosting_v2.pkl"
ADABOOST_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "adaboost_v2.pkl"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.2  # 20% of train for validation


def create_train_val_test_split(X_train, y_train, patient_ids_train, X_test, y_test, patient_ids_test):
    """Split train into train/validation with no patient overlap."""
    from sklearn.model_selection import GroupShuffleSplit

    # Get unique train patients
    unique_train_patients = np.unique(patient_ids_train)
    print(f"Original train patients: {len(unique_train_patients)}")

    # Create patient-level mapping
    patient_to_indices = {}
    for idx, pid in enumerate(patient_ids_train):
        if pid not in patient_to_indices:
            patient_to_indices[pid] = []
        patient_to_indices[pid].append(idx)

    # Split patients into train/validation
    gss = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE)
    # Use the indices as groups - this ensures we split by patient
    train_idx, val_idx = next(gss.split(X_train, y_train, groups=patient_ids_train))

    X_train_new = X_train[train_idx]
    y_train_new = y_train[train_idx]
    patient_ids_train_new = patient_ids_train[train_idx]

    X_val = X_train[val_idx]
    y_val = y_train[val_idx]
    patient_ids_val = patient_ids_train[val_idx]

    # Verify no patient overlap
    train_patients = set(np.unique(patient_ids_train_new))
    val_patients = set(np.unique(patient_ids_val))
    test_patients = set(np.unique(patient_ids_test))

    train_val_overlap = train_patients & val_patients
    train_test_overlap = train_patients & test_patients
    val_test_overlap = val_patients & test_patients

    if train_val_overlap:
        raise ValueError(f"Train/val patient overlap: {len(train_val_overlap)} patients")
    if train_test_overlap:
        raise ValueError(f"Train/test patient overlap: {len(train_test_overlap)} patients")
    if val_test_overlap:
        raise ValueError(f"Val/test patient overlap: {len(val_test_overlap)} patients")

    print(f"Train patients: {len(train_patients)}")
    print(f"Validation patients: {len(val_patients)}")
    print(f"Test patients: {len(test_patients)}")
    print("No patient overlap between splits [OK]")

    return X_train_new, y_train_new, patient_ids_train_new, X_val, y_val, patient_ids_val, X_test, y_test, patient_ids_test


def compute_operating_characteristics(y_true, scores, thresholds):
    """Compute sensitivity, specificity, precision for each threshold."""
    metrics = []
    for thresh in thresholds:
        y_pred = (scores >= thresh).astype(int)
        if y_pred.sum() == 0:
            precision = 0.0
        else:
            precision = precision_score(y_true, y_pred, zero_division=0)
        sensitivity = recall_score(y_true, y_pred, zero_division=0)
        specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        metrics.append({
            "threshold": float(thresh),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision),
            "f1": float(f1),
        })

    return metrics


def select_threshold_by_metrics(metrics, target_sensitivity=0.8, target_specificity=0.7):
    """Select threshold that best balances sensitivity and specificity."""
    # Find threshold closest to target sensitivity while maintaining reasonable specificity
    best_thresh = None
    best_score = -1

    for m in metrics:
        # Score: prioritize sensitivity, but penalize low specificity
        if m["sensitivity"] >= target_sensitivity and m["specificity"] >= target_specificity:
            score = m["sensitivity"] + m["specificity"]
            if score > best_score:
                best_score = score
                best_thresh = m

    # If no threshold meets targets, pick one with best F1
    if best_thresh is None:
        best_thresh = max(metrics, key=lambda x: x["f1"])

    return best_thresh


def main():
    print("=" * 80)
    print("VALIDATION-SET THRESHOLD SELECTION")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    with np.load(DATA_V2_PATH, allow_pickle=True) as data:
        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]
        patient_ids_train = data["patient_ids_train"]
        patient_ids_test = data["patient_ids_test"]
        feature_names = data["feature_names"].tolist()

    print(f"Original train: {len(X_train)} windows, {y_train.sum()} positive")
    print(f"Test: {len(X_test)} windows, {y_test.sum()} positive")

    # Filter out ablated features (Unit1, Unit2, HospAdmTime) as models were trained without them
    ablated_features = ["Unit1", "Unit2", "HospAdmTime"]
    ablated_indices = [i for i, name in enumerate(feature_names) if any(af in name for af in ablated_features)]
    if ablated_indices:
        X_train = np.delete(X_train, ablated_indices, axis=1)
        X_test = np.delete(X_test, ablated_indices, axis=1)
        feature_names = [name for i, name in enumerate(feature_names) if i not in ablated_indices]
        print(f"Filtered {len(ablated_indices)} ablated features, remaining: {len(feature_names)}")

    # Create train/validation/test split
    print("\nCreating train/validation/test split...")
    X_train_new, y_train_new, patient_ids_train_new, X_val, y_val, patient_ids_val, X_test, y_test, patient_ids_test = \
        create_train_val_test_split(X_train, y_train, patient_ids_train, X_test, y_test, patient_ids_test)

    print(f"New train: {len(X_train_new)} windows, {y_train_new.sum()} positive")
    print(f"Validation: {len(X_val)} windows, {y_val.sum()} positive")
    print(f"Test: {len(X_test)} windows, {y_test.sum()} positive")

    # Load models
    print("\nLoading models...")
    models = {
        "xgb": joblib.load(XGB_MODEL_PATH),
        "rf": joblib.load(RF_MODEL_PATH),
        "gb": joblib.load(GB_MODEL_PATH),
        "adaboost": joblib.load(ADABOOST_MODEL_PATH),
    }

    # Compute thresholds for each model
    thresholds = {}
    threshold_analysis = {}

    for name, model in models.items():
        print(f"\n{'=' * 60}")
        print(f"Model: {name.upper()}")
        print(f"{'=' * 60}")

        # Get validation scores
        val_scores = model.predict_proba(X_val)[:, 1]
        print(f"Validation score range: [{val_scores.min():.4f}, {val_scores.max():.4f}]")
        print(f"Validation mean: {val_scores.mean():.4f}, std: {val_scores.std():.4f}")

        # Test score distribution for reference only
        test_scores = model.predict_proba(X_test)[:, 1]
        print(f"Test score range: [{test_scores.min():.4f}, {test_scores.max():.4f}]")

        # Compute operating characteristics across percentile thresholds
        percentile_thresholds = np.linspace(10, 90, 81)  # 10th to 90th percentile
        score_thresholds = np.percentile(val_scores, percentile_thresholds)

        metrics = compute_operating_characteristics(y_val, val_scores, score_thresholds)

        # Select red threshold (high risk) - target high sensitivity
        red_thresh_info = select_threshold_by_metrics(metrics, target_sensitivity=0.85, target_specificity=0.5)
        red_threshold = red_thresh_info["threshold"]

        # Select yellow threshold (intermediate risk) - target moderate sensitivity
        yellow_thresh_info = select_threshold_by_metrics(metrics, target_sensitivity=0.70, target_specificity=0.6)
        yellow_threshold = yellow_thresh_info["threshold"]

        # Ensure red > yellow
        if red_threshold <= yellow_threshold:
            yellow_threshold = red_threshold * 0.8  # Set yellow below red

        thresholds[name] = {
            "red": float(red_threshold),
            "yellow": float(yellow_threshold),
        }

        threshold_analysis[name] = {
            "red_threshold": float(red_threshold),
            "yellow_threshold": float(yellow_threshold),
            "red_sensitivity": float(red_thresh_info["sensitivity"]),
            "red_specificity": float(red_thresh_info["specificity"]),
            "red_precision": float(red_thresh_info["precision"]),
            "red_f1": float(red_thresh_info["f1"]),
            "yellow_sensitivity": float(yellow_thresh_info["sensitivity"]),
            "yellow_specificity": float(yellow_thresh_info["specificity"]),
            "yellow_precision": float(yellow_thresh_info["precision"]),
            "yellow_f1": float(yellow_thresh_info["f1"]),
            "validation_positive_rate": float(y_val.mean()),
            "validation_prevalence": float(y_val.sum() / len(y_val)),
        }

        print(f"\nSelected thresholds:")
        print(f"  Red (>= {red_threshold:.4f}): sensitivity={red_thresh_info['sensitivity']:.3f}, specificity={red_thresh_info['specificity']:.3f}, precision={red_thresh_info['precision']:.3f}")
        print(f"  Yellow (>= {yellow_threshold:.4f}): sensitivity={yellow_thresh_info['sensitivity']:.3f}, specificity={yellow_thresh_info['specificity']:.3f}, precision={yellow_thresh_info['precision']:.3f}")

    # Save thresholds
    THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    thresholds_output = {
        "thresholds": thresholds,
        "analysis": threshold_analysis,
        "metadata": {
            "selection_method": "validation_set_operating_characteristics",
            "validation_size": VALIDATION_SIZE,
            "random_state": RANDOM_STATE,
            "train_patients": int(len(np.unique(patient_ids_train_new))),
            "validation_patients": int(len(np.unique(patient_ids_val))),
            "test_patients": int(len(np.unique(patient_ids_test))),
            "note": "PROVISIONAL/EXPERIMENTAL - Not clinically validated. Research prototype only.",
            "disclaimer": "These thresholds are selected on a validation set using operating characteristics (sensitivity, specificity, precision). They are NOT clinically validated and should NOT be used for clinical decision making.",
        }
    }

    with open(THRESHOLDS_PATH, "w") as f:
        json.dump(thresholds_output, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Saved validation-based thresholds to {THRESHOLDS_PATH}")
    print(f"{'=' * 80}")
    print("\nDISCLAIMER: These thresholds are PROVISIONAL/EXPERIMENTAL.")
    print("They are selected on a validation set using operating characteristics.")
    print("They are NOT clinically validated and should NOT be used for clinical decision making.")
    print("This is a research prototype for demonstration purposes only.")


if __name__ == "__main__":
    main()
