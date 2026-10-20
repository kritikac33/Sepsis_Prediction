"""Compute operating points for XGBoost and LR."""

import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_v2.npz"
XGB_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "xgb_final_v2.pkl"
LR_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "lr_final_v2.pkl"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "operating_points.json"


def compute_metrics_at_threshold(y_true, y_pred_proba, threshold):
    y_pred = (y_pred_proba >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Alerts per true case caught
    alerts_per_case = (tp + fp) / tp if tp > 0 else 0

    # Lift over base rate
    base_rate = np.mean(y_true)
    lift = precision / base_rate if base_rate > 0 else 0

    return {
        "threshold": float(threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "lift_over_base_rate": float(lift),
        "alerts_per_true_case": float(alerts_per_case),
    }


def find_threshold_at_percentile(y_pred_proba, percentile):
    return np.percentile(y_pred_proba, 100 - percentile)


def find_threshold_at_specificity(y_true, y_pred_proba, target_specificity):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    specificity = 1 - fpr
    idx = np.argmin(np.abs(specificity - target_specificity))
    return thresholds[idx]


def main():
    start_time = time.time()

    # Load data
    data = np.load(FEATURES_PATH, allow_pickle=True)
    X_test = data["X_test"]
    y_test = data["y_test"]
    feature_names = data["feature_names"].tolist()

    # Remove ablated features
    ablated = [i for i, n in enumerate(feature_names) if any(af in n for af in ["Unit1", "Unit2", "HospAdmTime"])]
    X_test = np.delete(X_test, ablated, axis=1)

    # Load models
    with open(XGB_MODEL_PATH, "rb") as f:
        xgb_artifact = pickle.load(f)
    with open(LR_MODEL_PATH, "rb") as f:
        lr_artifact = pickle.load(f)

    xgb_model = xgb_artifact["model"]
    lr_model = lr_artifact["model"]

    # Get predictions
    xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
    lr_pred = lr_model.predict_proba(X_test)[:, 1]

    # Compute metrics
    results = {
        "xgb": {},
        "lr": {},
    }

    # XGBoost
    print("=== XGBoost Operating Points ===")
    for percentile in [1, 2, 5, 10, 20]:
        threshold = find_threshold_at_percentile(xgb_pred, percentile)
        metrics = compute_metrics_at_threshold(y_test, xgb_pred, threshold)
        results["xgb"][f"top_{percentile}%"] = metrics
        print(f"Top {percentile}%: threshold={threshold:.4f}, sens={metrics['sensitivity']:.4f}, spec={metrics['specificity']:.4f}, prec={metrics['precision']:.4f}, lift={metrics['lift_over_base_rate']:.2f}x, alerts/case={metrics['alerts_per_true_case']:.1f}")

    # Sensitivity at high specificity
    for spec in [0.90, 0.95]:
        threshold = find_threshold_at_specificity(y_test, xgb_pred, spec)
        metrics = compute_metrics_at_threshold(y_test, xgb_pred, threshold)
        results["xgb"][f"spec_{int(spec*100)}%"] = metrics
        print(f"Spec {int(spec*100)}%: threshold={threshold:.4f}, sens={metrics['sensitivity']:.4f}, prec={metrics['precision']:.4f}")

    # Logistic Regression
    print("\n=== LR Operating Points ===")
    for percentile in [1, 2, 5, 10, 20]:
        threshold = find_threshold_at_percentile(lr_pred, percentile)
        metrics = compute_metrics_at_threshold(y_test, lr_pred, threshold)
        results["lr"][f"top_{percentile}%"] = metrics
        print(f"Top {percentile}%: threshold={threshold:.4f}, sens={metrics['sensitivity']:.4f}, spec={metrics['specificity']:.4f}, prec={metrics['precision']:.4f}, lift={metrics['lift_over_base_rate']:.2f}x, alerts/case={metrics['alerts_per_true_case']:.1f}")

    for spec in [0.90, 0.95]:
        threshold = find_threshold_at_specificity(y_test, lr_pred, spec)
        metrics = compute_metrics_at_threshold(y_test, lr_pred, threshold)
        results["lr"][f"spec_{int(spec*100)}%"] = metrics
        print(f"Spec {int(spec*100)}%: threshold={threshold:.4f}, sens={metrics['sensitivity']:.4f}, prec={metrics['precision']:.4f}")

    # Overall metrics
    results["xgb"]["overall"] = {
        "roc_auc": float(roc_auc_score(y_test, xgb_pred)),
    }
    results["lr"]["overall"] = {
        "roc_auc": float(roc_auc_score(y_test, lr_pred)),
    }

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {OUTPUT_PATH}")

    # Print table
    print("\n=== Operating Points Table ===")
    print(f"{'Model':<6} {'Percentile':<12} {'Threshold':<10} {'Sens':<8} {'Spec':<8} {'Prec':<8} {'Lift':<8} {'Alerts/Case':<12}")
    print("-" * 85)
    for model_name, model_pred in [("XGB", xgb_pred), ("LR", lr_pred)]:
        for percentile in [1, 2, 5, 10, 20]:
            threshold = find_threshold_at_percentile(model_pred, percentile)
            metrics = compute_metrics_at_threshold(y_test, model_pred, threshold)
            print(f"{model_name:<6} {f'Top {percentile}%':<12} {threshold:<10.4f} {metrics['sensitivity']:<8.4f} {metrics['specificity']:<8.4f} {metrics['precision']:<8.4f} {metrics['lift_over_base_rate']:<8.2f} {metrics['alerts_per_true_case']:<12.1f}")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")

    # Update progress log
    with open(PROJECT_ROOT / "outputs" / "progress_log.md", "a") as f:
        f.write(f"| PART 2: Operating points | XGB AUC={results['xgb']['overall']['roc_auc']:.4f}, LR AUC={results['lr']['overall']['roc_auc']:.4f} | {elapsed:.1f} |\n")


if __name__ == "__main__":
    main()
