"""Compute clinical-score baselines (SIRS, qSOFA-lite) on test set."""

import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_V2_PATH = PROJECT_ROOT / "data" / "processed" / "data_v2.npz"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_v2.npz"
XGB_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "xgb_final_v2.pkl"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "clinical_baselines.json"


def main():
    start_time = time.time()

    # Load features (for matching split)
    data_features = np.load(FEATURES_PATH, allow_pickle=True)
    X_test_features = data_features["X_test"]
    y_test = data_features["y_test"]
    feature_names = data_features["feature_names"].tolist()

    # Load XGBoost model
    with open(XGB_MODEL_PATH, "rb") as f:
        xgb_artifact = pickle.load(f)
    xgb_model = xgb_artifact["model"]

    # Remove ablated features from XGBoost features
    ablated = [i for i, n in enumerate(feature_names) if any(af in n for af in ["Unit1", "Unit2", "HospAdmTime"])]
    X_test_features = np.delete(X_test_features, ablated, axis=1)

    # Get XGBoost predictions
    xgb_pred = xgb_model.predict_proba(X_test_features)[:, 1]

    # Get raw values from summary features (last value)
    # The summary features contain denormalized values from the aggregation step
    # We need to get the original normalization stats to denormalize
    data_v2 = np.load(DATA_V2_PATH, allow_pickle=True)
    feature_means = data_v2["feature_means"].astype(np.float32)
    feature_stds = data_v2["feature_stds"].astype(np.float32)
    raw_feature_names = data_v2["feature_names"].astype(str).tolist()

    # Load the full features to get raw summary values
    data_features_full = np.load(FEATURES_PATH, allow_pickle=True)
    X_test_full = data_features_full["X_test"]
    feature_names_full = data_features_full["feature_names"].tolist()

    # Find indices for clinical variables in full features
    idx_hr_last = feature_names_full.index("HR_last")
    idx_temp_last = feature_names_full.index("Temp_last")
    idx_resp_last = feature_names_full.index("Resp_last")
    idx_wbc_last = feature_names_full.index("WBC_last")
    idx_sbp_last = feature_names_full.index("SBP_last")

    # Get raw last values (already denormalized in aggregation)
    hr_raw = X_test_full[:, idx_hr_last]
    temp_raw = X_test_full[:, idx_temp_last]
    resp_raw = X_test_full[:, idx_resp_last]
    wbc_raw = X_test_full[:, idx_wbc_last]
    sbp_raw = X_test_full[:, idx_sbp_last]

    # Denormalize using original stats from data_v2
    # Need to map summary feature names back to original feature names
    hr_idx_orig = raw_feature_names.index("HR")
    temp_idx_orig = raw_feature_names.index("Temp")
    resp_idx_orig = raw_feature_names.index("Resp")
    wbc_idx_orig = raw_feature_names.index("WBC")
    sbp_idx_orig = raw_feature_names.index("SBP")

    hr_raw = hr_raw * feature_stds[hr_idx_orig] + feature_means[hr_idx_orig]
    temp_raw = temp_raw * feature_stds[temp_idx_orig] + feature_means[temp_idx_orig]
    resp_raw = resp_raw * feature_stds[resp_idx_orig] + feature_means[resp_idx_orig]
    wbc_raw = wbc_raw * feature_stds[wbc_idx_orig] + feature_means[wbc_idx_orig]
    sbp_raw = sbp_raw * feature_stds[sbp_idx_orig] + feature_means[sbp_idx_orig]

    # Check WBC missingness
    wbc_missing = np.isnan(wbc_raw) | (wbc_raw == 0)
    wbc_missing_rate = np.mean(wbc_missing)
    print(f"WBC missing rate: {wbc_missing_rate:.1%}")

    # Compute SIRS criteria
    sirs_temp = (temp_raw > 38) | (temp_raw < 36)
    sirs_hr = hr_raw > 90
    sirs_resp = resp_raw > 20
    sirs_wbc = (wbc_raw > 12) | (wbc_raw < 4)
    sirs_wbc[wbc_missing] = False  # Treat missing as not met

    sirs_count = sirs_temp.astype(int) + sirs_hr.astype(int) + sirs_resp.astype(int) + sirs_wbc.astype(int)

    # Compute qSOFA-lite criteria
    qsofa_resp = resp_raw >= 22
    qsofa_sbp = sbp_raw <= 100

    qsofa_count = qsofa_resp.astype(int) + qsofa_sbp.astype(int)

    # Helper function for metrics
    def compute_metrics(y_true, y_pred_proba, threshold=None):
        if threshold is not None:
            y_pred = (y_pred_proba >= threshold).astype(int)
        else:
            y_pred = y_pred_proba.astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

        return {
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision),
            "f1": float(f1),
        }

    # Compute ROC-AUC for clinical scores
    sirs_auc = roc_auc_score(y_test, sirs_count)
    qsofa_auc = roc_auc_score(y_test, qsofa_count)

    print(f"\nSIRS ROC-AUC: {sirs_auc:.4f}")
    print(f"qSOFA-lite ROC-AUC: {qsofa_auc:.4f}")

    # Compute metrics for clinical rules
    results = {}

    # SIRS >= 2
    sirs_ge2_metrics = compute_metrics(y_test, sirs_count >= 2)
    print(f"\nSIRS >= 2: {sirs_ge2_metrics}")

    # SIRS >= 1
    sirs_ge1_metrics = compute_metrics(y_test, sirs_count >= 1)
    print(f"SIRS >= 1: {sirs_ge1_metrics}")

    # qSOFA-lite >= 2
    qsofa_ge2_metrics = compute_metrics(y_test, qsofa_count >= 2)
    print(f"qSOFA-lite >= 2: {qsofa_ge2_metrics}")

    # qSOFA-lite >= 1
    qsofa_ge1_metrics = compute_metrics(y_test, qsofa_count >= 1)
    print(f"qSOFA-lite >= 1: {qsofa_ge1_metrics}")

    # For each clinical rule, find XGBoost threshold at same sensitivity
    def find_threshold_at_sensitivity(y_true, y_pred_proba, target_sensitivity):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        idx = np.argmin(np.abs(tpr - target_sensitivity))
        return thresholds[idx]

    results["sirs_ge2"] = {
        "roc_auc": float(sirs_auc),
        "clinical_metrics": sirs_ge2_metrics,
    }

    xgb_thresh_sirs2 = find_threshold_at_sensitivity(y_test, xgb_pred, sirs_ge2_metrics["sensitivity"])
    xgb_at_sirs2 = compute_metrics(y_test, xgb_pred, xgb_thresh_sirs2)
    base_rate = np.mean(y_test)
    lift_sirs2 = xgb_at_sirs2["precision"] / base_rate if base_rate > 0 else 0

    results["sirs_ge2"]["xgb_at_same_sensitivity"] = {
        "threshold": float(xgb_thresh_sirs2),
        "metrics": xgb_at_sirs2,
        "lift_over_base_rate": float(lift_sirs2),
    }

    results["sirs_ge1"] = {
        "roc_auc": float(sirs_auc),
        "clinical_metrics": sirs_ge1_metrics,
    }

    xgb_thresh_sirs1 = find_threshold_at_sensitivity(y_test, xgb_pred, sirs_ge1_metrics["sensitivity"])
    xgb_at_sirs1 = compute_metrics(y_test, xgb_pred, xgb_thresh_sirs1)
    lift_sirs1 = xgb_at_sirs1["precision"] / base_rate if base_rate > 0 else 0

    results["sirs_ge1"]["xgb_at_same_sensitivity"] = {
        "threshold": float(xgb_thresh_sirs1),
        "metrics": xgb_at_sirs1,
        "lift_over_base_rate": float(lift_sirs1),
    }

    results["qsofa_ge2"] = {
        "roc_auc": float(qsofa_auc),
        "clinical_metrics": qsofa_ge2_metrics,
    }

    xgb_thresh_qsofa2 = find_threshold_at_sensitivity(y_test, xgb_pred, qsofa_ge2_metrics["sensitivity"])
    xgb_at_qsofa2 = compute_metrics(y_test, xgb_pred, xgb_thresh_qsofa2)
    lift_qsofa2 = xgb_at_qsofa2["precision"] / base_rate if base_rate > 0 else 0

    results["qsofa_ge2"]["xgb_at_same_sensitivity"] = {
        "threshold": float(xgb_thresh_qsofa2),
        "metrics": xgb_at_qsofa2,
        "lift_over_base_rate": float(lift_qsofa2),
    }

    results["qsofa_ge1"] = {
        "roc_auc": float(qsofa_auc),
        "clinical_metrics": qsofa_ge1_metrics,
    }

    xgb_thresh_qsofa1 = find_threshold_at_sensitivity(y_test, xgb_pred, qsofa_ge1_metrics["sensitivity"])
    xgb_at_qsofa1 = compute_metrics(y_test, xgb_pred, xgb_thresh_qsofa1)
    lift_qsofa1 = xgb_at_qsofa1["precision"] / base_rate if base_rate > 0 else 0

    results["qsofa_ge1"]["xgb_at_same_sensitivity"] = {
        "threshold": float(xgb_thresh_qsofa1),
        "metrics": xgb_at_qsofa1,
        "lift_over_base_rate": float(lift_qsofa1),
    }

    results["base_rate"] = float(base_rate)

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {OUTPUT_PATH}")

    # Print table
    print("\n=== Clinical Baselines Table ===")
    print(f"{'Rule':<15} {'ROC-AUC':<10} {'Sens':<8} {'Spec':<8} {'Prec':<8} {'F1':<8}")
    print("-" * 65)
    print(f"{'SIRS >= 2':<15} {sirs_auc:<10.4f} {sirs_ge2_metrics['sensitivity']:<8.4f} {sirs_ge2_metrics['specificity']:<8.4f} {sirs_ge2_metrics['precision']:<8.4f} {sirs_ge2_metrics['f1']:<8.4f}")
    print(f"{'SIRS >= 1':<15} {sirs_auc:<10.4f} {sirs_ge1_metrics['sensitivity']:<8.4f} {sirs_ge1_metrics['specificity']:<8.4f} {sirs_ge1_metrics['precision']:<8.4f} {sirs_ge1_metrics['f1']:<8.4f}")
    print(f"{'qSOFA >= 2':<15} {qsofa_auc:<10.4f} {qsofa_ge2_metrics['sensitivity']:<8.4f} {qsofa_ge2_metrics['specificity']:<8.4f} {qsofa_ge2_metrics['precision']:<8.4f} {qsofa_ge2_metrics['f1']:<8.4f}")
    print(f"{'qSOFA >= 1':<15} {qsofa_auc:<10.4f} {qsofa_ge1_metrics['sensitivity']:<8.4f} {qsofa_ge1_metrics['specificity']:<8.4f} {qsofa_ge1_metrics['precision']:<8.4f} {qsofa_ge1_metrics['f1']:<8.4f}")

    print("\n=== XGBoost at Same Sensitivity ===")
    print(f"{'Rule':<15} {'XGB Spec':<10} {'XGB Prec':<10} {'XGB F1':<10} {'Lift':<10}")
    print("-" * 65)
    print(f"{'SIRS >= 2':<15} {xgb_at_sirs2['specificity']:<10.4f} {xgb_at_sirs2['precision']:<10.4f} {xgb_at_sirs2['f1']:<10.4f} {lift_sirs2:<10.2f}x")
    print(f"{'SIRS >= 1':<15} {xgb_at_sirs1['specificity']:<10.4f} {xgb_at_sirs1['precision']:<10.4f} {xgb_at_sirs1['f1']:<10.4f} {lift_sirs1:<10.2f}x")
    print(f"{'qSOFA >= 2':<15} {xgb_at_qsofa2['specificity']:<10.4f} {xgb_at_qsofa2['precision']:<10.4f} {xgb_at_qsofa2['f1']:<10.4f} {lift_qsofa2:<10.2f}x")
    print(f"{'qSOFA >= 1':<15} {xgb_at_qsofa1['specificity']:<10.4f} {xgb_at_qsofa1['precision']:<10.4f} {xgb_at_qsofa1['f1']:<10.4f} {lift_qsofa1:<10.2f}x")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")

    # Update progress log
    with open(PROJECT_ROOT / "outputs" / "progress_log.md", "a") as f:
        f.write(f"| PART 1: Clinical baselines | SIRS AUC={sirs_auc:.4f}, qSOFA AUC={qsofa_auc:.4f} | {elapsed:.1f} |\n")


if __name__ == "__main__":
    main()
