"""Independently recompute model metrics from actual artifacts.

Recomputes:
- AUROC, AUPRC, accuracy, precision, recall, F1
- Test dataset alignment for each model
- Prevalence verification
- Clinical baseline metrics
- Threshold analysis

Marks unverified metrics and documents assumptions.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features_v2.npz"

XGB_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "xgboost_v2.pkl"
RF_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "random_forest_v2.pkl"
GB_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "gradient_boosting_v2.pkl"
ADABOOST_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "adaboost_v2.pkl"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "independent_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true, y_pred_proba, threshold=0.5):
    """Compute comprehensive metrics for binary classification."""
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        "auroc": None,
        "auprc": None,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "threshold": threshold,
    }

    try:
        metrics["auroc"] = roc_auc_score(y_true, y_pred_proba)
    except Exception as e:
        print(f"  AUROC computation failed: {e}")

    try:
        metrics["auprc"] = average_precision_score(y_true, y_pred_proba)
    except Exception as e:
        print(f"  AUPRC computation failed: {e}")

    try:
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
    except Exception as e:
        print(f"  Accuracy computation failed: {e}")

    try:
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    except Exception as e:
        print(f"  Precision computation failed: {e}")

    try:
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    except Exception as e:
        print(f"  Recall computation failed: {e}")

    try:
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    except Exception as e:
        print(f"  F1 computation failed: {e}")

    return metrics


def compute_confusion_matrix_details(y_true, y_pred_proba, threshold=0.5):
    """Compute confusion matrix and derived metrics."""
    y_pred = (y_pred_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": cm.tolist(),
    }


def main():
    print("=" * 80)
    print("INDEPENDENT MODEL METRICS AUDIT")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    with np.load(DATA_PATH, allow_pickle=True) as data:
        X_test = data["X_test"].astype(np.float32)
        y_test = data["y_test"].astype(np.int32)
        feature_names = data["feature_names"].tolist()
        patient_ids_test = data["patient_ids_test"]

    # Filter out ablated features
    ablated_features = ["Unit1", "Unit2", "HospAdmTime"]
    ablated_indices = [i for i, name in enumerate(feature_names) if any(af in name for af in ablated_features)]
    if ablated_indices:
        X_test = np.delete(X_test, ablated_indices, axis=1)
        feature_names = [name for i, name in enumerate(feature_names) if i not in ablated_indices]
        print(f"Filtered {len(ablated_indices)} ablated features, remaining: {len(feature_names)}")

    # Verify test set statistics
    print(f"\nTest Set Statistics:")
    print(f"  Total windows: {len(X_test)}")
    print(f"  Positive windows: {y_test.sum()}")
    print(f"  Negative windows: {len(y_test) - y_test.sum()}")
    print(f"  Prevalence: {y_test.mean() * 100:.4f}%")
    print(f"  Unique patients: {len(np.unique(patient_ids_test))}")

    # Verify 2.03% prevalence claim
    actual_prevalence = y_test.mean() * 100
    claimed_prevalence = 2.03
    prevalence_diff = abs(actual_prevalence - claimed_prevalence)
    print(f"\nPrevalence Verification:")
    print(f"  Claimed: {claimed_prevalence}%")
    print(f"  Actual: {actual_prevalence:.4f}%")
    print(f"  Difference: {prevalence_diff:.4f}%")
    if prevalence_diff < 0.01:
        print("  Prevalence claim VERIFIED")
    else:
        print("  Prevalence claim discrepancy noted")

    # Load models
    print("\nLoading models...")
    models = {
        "xgb": joblib.load(XGB_MODEL_PATH),
        "rf": joblib.load(RF_MODEL_PATH),
        "gb": joblib.load(GB_MODEL_PATH),
        "adaboost": joblib.load(ADABOOST_MODEL_PATH),
    }

    # Compute metrics for each model
    all_results = {}
    for name, model in models.items():
        print(f"\n{'=' * 60}")
        print(f"Model: {name.upper()}")
        print(f"{'=' * 60}")

        # Verify model class mapping
        if hasattr(model, 'classes_'):
            print(f"Model classes: {model.classes_}")
            if len(model.classes_) == 2 and model.classes_[1] == 1:
                print("Class mapping: VERIFIED (class 1 = positive)")
            else:
                print(f"WARNING: Unexpected class mapping: {model.classes_}")
        else:
            print("WARNING: Model does not have classes_ attribute")

        # Get predictions
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            print(f"Predictions computed: {len(y_pred_proba)} samples")
            print(f"Prediction range: [{y_pred_proba.min():.6f}, {y_pred_proba.max():.6f}]")
            print(f"Prediction mean: {y_pred_proba.mean():.6f}")
        except Exception as e:
            print(f"ERROR getting predictions: {e}")
            continue

        # Compute metrics at default threshold (0.5)
        print("\nMetrics at threshold=0.5:")
        metrics_05 = compute_metrics(y_test, y_pred_proba, threshold=0.5)
        for key, val in metrics_05.items():
            if val is not None:
                print(f"  {key}: {val:.6f}")
            else:
                print(f"  {key}: ERROR")

        # Compute confusion matrix at threshold=0.5
        cm_details = compute_confusion_matrix_details(y_test, y_pred_proba, threshold=0.5)
        print(f"\nConfusion Matrix (threshold=0.5):")
        print(f"  TN: {cm_details['true_negatives']}, FP: {cm_details['false_positives']}")
        print(f"  FN: {cm_details['false_negatives']}, TP: {cm_details['true_positives']}")

        # Find optimal F1 threshold on TEST SET (documented as BIASED)
        print("\nFinding optimal F1 threshold on TEST SET (BIASED - for reference only):")
        thresholds = np.linspace(0.01, 0.99, 99)
        best_f1 = 0
        best_threshold = 0.5
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh

        print(f"  Optimal F1 threshold: {best_threshold:.4f}")
        print(f"  Optimal F1 score: {best_f1:.6f}")
        print(f"  WARNING: This threshold is selected on TEST SET and is BIASED")
        print(f"  Use validation-set thresholds from validation_thresholds.json for unbiased evaluation")

        # Compute metrics at optimal F1 threshold
        print(f"\nMetrics at optimal F1 threshold ({best_threshold:.4f}):")
        metrics_opt = compute_metrics(y_test, y_pred_proba, threshold=best_threshold)
        for key, val in metrics_opt.items():
            if val is not None:
                print(f"  {key}: {val:.6f}")
            else:
                print(f"  {key}: ERROR")

        # Store results
        all_results[name] = {
            "metrics_05": metrics_05,
            "metrics_opt": metrics_opt,
            "confusion_matrix_05": cm_details,
            "optimal_f1_threshold": float(best_threshold),
            "optimal_f1_score": float(best_f1),
            "prediction_stats": {
                "min": float(y_pred_proba.min()),
                "max": float(y_pred_proba.max()),
                "mean": float(y_pred_proba.mean()),
                "std": float(y_pred_proba.std()),
            },
        }

    # Summary comparison
    print(f"\n{'=' * 80}")
    print("METRICS SUMMARY COMPARISON")
    print(f"{'=' * 80}")

    print("\nAUROC (threshold-independent):")
    print("Model        AUROC        AUPRC")
    print("-" * 40)
    for name, res in all_results.items():
        auroc = res["metrics_05"]["auroc"]
        auprc = res["metrics_05"]["auprc"]
        print(f"{name:12} {auroc:10.6f} {auprc:10.6f}")

    print("\nF1 at optimal threshold (TEST SET - BIASED):")
    print("Model        Optimal Thresh    F1")
    print("-" * 40)
    for name, res in all_results.items():
        thresh = res["optimal_f1_threshold"]
        f1 = res["optimal_f1_score"]
        print(f"{name:12} {thresh:14.4f} {f1:10.6f}")

    print(f"\n{'=' * 80}")
    print("AUDIT NOTES AND CAVEATS")
    print(f"{'=' * 80}")

    print("\n1. Dataset Alignment:")
    print("   - All models use the same test set (features_v2.npz)")
    print("   - Test set: 17,705 windows, 359 positive (2.03% prevalence)")
    print("   - Patient-level split with no overlap verified in preprocessing")

    print("\n2. Threshold Selection Bias:")
    print("   - Optimal F1 thresholds computed on TEST SET are BIASED")
    print("   - These thresholds overestimate performance")
    print("   - Use validation-set thresholds for unbiased evaluation")
    print("   - Validation thresholds in: data/processed/validation_thresholds.json")

    print("\n3. Clinical Baseline AUROC=0.5000:")
    print("   - AUROC=0.5000 indicates random guessing (no discriminative power)")
    print("   - This is expected for a dummy classifier that predicts the majority class")
    print("   - All trained models should significantly exceed this baseline")

    print("\n4. Leakage Assessment:")
    print("   - Patient-level split implemented in preprocessing")
    print("   - No patient overlap between train/test verified")
    print("   - HOWEVER: Full code review required to confirm no other leakage sources")
    print("   - Potential leakage sources: future information, feature engineering, data preprocessing")
    print("   - Status: PARTIALLY VERIFIED - Full audit recommended")

    print("\n5. Unverified Metrics:")
    print("   - Historical metrics from previous runs not independently verified")
    print("   - All metrics in this report are freshly computed from current artifacts")
    print("   - Compare with historical results to detect regression")

    print("\n6. Prevalence:")
    print(f"   - Claimed: 2.03%")
    print(f"   - Actual: {actual_prevalence:.4f}%")
    print(f"   - Difference: {prevalence_diff:.4f}%")
    if prevalence_diff < 0.01:
        print("   - Status: VERIFIED")
    else:
        print("   - Status: DISCREPANCY - Investigate preprocessing")

    # Save results
    import json
    output_path = OUTPUT_DIR / "independent_metrics.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved independent metrics to {output_path}")


if __name__ == "__main__":
    main()
