"""
Comprehensive ML Audit Script for Sepsis Prediction Project
============================================================

This script performs a thorough audit of all trained models including:
1. Loading all model artifacts from outputs/models/
2. Loading test data from data/processed/features_v2.npz
3. Calculating comprehensive metrics for each model
4. Checking for data leakage in preprocessing and train/test split
5. Verifying historical reported scores
6. Evaluating clinical baselines
7. Saving results to AUDIT-FINAL-14OCT.xlsx with 10 worksheets

Author: Audit Script
Date: October 14, 2025
"""

from __future__ import annotations

import json
import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features_v2.npz"
SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "split.json"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
OUTPUT_EXCEL = PROJECT_ROOT / "AUDIT-FINAL-14OCT.xlsx"

# Historical reported scores for verification
HISTORICAL_SCORES = {
    "XGBoost": 0.8544,
    "LogisticRegression": 0.7943,
    "Ensemble": 0.8485,
}

# Clinical baseline thresholds
CLINICAL_BASELINES = {
    "lactate_4": 4.0,  # mmol/L
    "lactate_8": 8.0,  # mmol/L
    "map_65": 65,  # mmHg
    "map_50": 50,  # mmHg
    "hr_130": 130,  # bpm
    "hr_160": 160,  # bpm
    "o2sat_90": 90,  # %
    "o2sat_82": 82,  # %
}

# Ablated features to remove
ABLATED_FEATURES = ["Unit1", "Unit2", "HospAdmTime"]


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load test data from features_v2.npz."""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    with np.load(DATA_PATH, allow_pickle=True) as data:
        X_test = data["X_test"]
        y_test = data["y_test"]
        patient_ids_test = data["patient_ids_test"]
        feature_names = data["feature_names"].tolist()
    
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"patient_ids_test shape: {patient_ids_test.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Test set prevalence: {y_test.mean():.4f}")
    print(f"Number of test patients: {len(np.unique(patient_ids_test))}")
    
    return X_test, y_test, patient_ids_test, feature_names


def load_split_info() -> Dict[str, List[str]]:
    """Load train/test split information."""
    print("\n" + "=" * 80)
    print("LOADING SPLIT INFORMATION")
    print("=" * 80)
    
    with open(SPLIT_PATH, "r") as f:
        split_info = json.load(f)
    
    train_patients = set(split_info["train_patients"])
    test_patients = set(split_info["test_patients"])
    
    print(f"Number of train patients: {len(train_patients)}")
    print(f"Number of test patients: {len(test_patients)}")
    
    overlap = train_patients & test_patients
    if overlap:
        print(f"WARNING: {len(overlap)} patients appear in BOTH train and test sets!")
        print(f"Overlap patients: {sorted(list(overlap))[:10]}...")
    else:
        print("[OK] No patient overlap between train and test sets")
    
    return split_info


def remove_ablated_features(
    X: np.ndarray, feature_names: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """Remove ablated features from data."""
    ablated_indices = [
        i for i, name in enumerate(feature_names) if any(af in name for af in ABLATED_FEATURES)
    ]
    X_clean = np.delete(X, ablated_indices, axis=1)
    feature_names_clean = [name for i, name in enumerate(feature_names) if i not in ablated_indices]
    
    print(f"\nRemoved {len(ablated_indices)} ablated features")
    print(f"Remaining features: {len(feature_names_clean)}")
    
    return X_clean, feature_names_clean


def calculate_comprehensive_metrics(
    y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate comprehensive metrics for binary classification."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # F-scores
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = (5 * precision * recall) / (4 * precision + recall + 1e-10) if (precision + recall) > 0 else 0
    
    # ROC and PR
    try:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        roc_auc = np.nan
    
    try:
        pr_auc = average_precision_score(y_true, y_pred_proba)
    except ValueError:
        pr_auc = np.nan
    
    # Balanced accuracy
    balanced_accuracy = (recall + specificity) / 2
    
    # Error rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Correlation metrics
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except ValueError:
        mcc = np.nan
    
    try:
        kappa = cohen_kappa_score(y_true, y_pred)
    except ValueError:
        kappa = np.nan
    
    # Probabilistic metrics
    try:
        logloss = log_loss(y_true, y_pred_proba)
    except ValueError:
        logloss = np.nan
    
    try:
        brier = brier_score_loss(y_true, y_pred_proba)
    except ValueError:
        brier = np.nan
    
    return {
        "AUROC": roc_auc,
        "PR-AUC": pr_auc,
        "Accuracy": accuracy,
        "Balanced_Accuracy": balanced_accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": f1,
        "F2": f2,
        "NPV": npv,
        "FPR": fpr,
        "FNR": fnr,
        "MCC": mcc,
        "Cohen_Kappa": kappa,
        "Log_Loss": logloss,
        "Brier_Score": brier,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def load_model(model_path: Path) -> Any:
    """Load a model from a pickle or joblib file."""
    try:
        # Try joblib first
        model = joblib.load(model_path)
        return model
    except:
        try:
            # Try pickle
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
            return None


def get_model_predictions(model: Any, X: np.ndarray) -> np.ndarray:
    """Get probability predictions from a model."""
    try:
        # Check if it's a dictionary artifact
        if isinstance(model, dict) and "model" in model:
            model = model["model"]
        
        # Try predict_proba
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        # Try decision_function
        elif hasattr(model, "decision_function"):
            proba = model.decision_function(X)
            # Normalize to [0, 1]
            proba = (proba - proba.min()) / (proba.max() - proba.min() + 1e-10)
            return proba
        else:
            print("Model has no predict_proba or decision_function method")
            return np.zeros(len(X))
    except Exception as e:
        print(f"Error getting predictions: {e}")
        return np.zeros(len(X))


def find_optimal_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Find optimal threshold using F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return best_threshold


def audit_all_models(
    X_test: np.ndarray, y_test: np.ndarray, feature_names: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Audit all models in the models directory."""
    print("\n" + "=" * 80)
    print("AUDITING ALL MODELS")
    print("=" * 80)
    
    results = {}
    
    # Find all model files
    model_files = list(MODELS_DIR.glob("*.pkl"))
    print(f"Found {len(model_files)} model files")
    
    # Process each model
    for model_path in sorted(model_files):
        model_name = model_path.stem
        print(f"\n--- Loading {model_name} ---")
        
        model = load_model(model_path)
        if model is None:
            print(f"Skipping {model_name} - could not load")
            continue
        
        # Get predictions
        try:
            y_pred_proba = get_model_predictions(model, X_test)
            
            # Find optimal threshold
            optimal_threshold = find_optimal_threshold(y_test, y_pred_proba)
            
            # Calculate metrics with default threshold
            metrics_default = calculate_comprehensive_metrics(y_test, y_pred_proba, threshold=0.5)
            
            # Calculate metrics with optimal threshold
            metrics_optimal = calculate_comprehensive_metrics(y_test, y_pred_proba, threshold=optimal_threshold)
            
            results[model_name] = {
                "model_path": str(model_path),
                "predictions_available": True,
                "optimal_threshold": optimal_threshold,
                "metrics_default": metrics_default,
                "metrics_optimal": metrics_optimal,
            }
            
            print(f"  AUROC (default): {metrics_default['AUROC']:.4f}")
            print(f"  AUROC (optimal): {metrics_optimal['AUROC']:.4f}")
            print(f"  Optimal threshold: {optimal_threshold:.4f}")
            
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            results[model_name] = {
                "model_path": str(model_path),
                "predictions_available": False,
                "error": str(e),
            }
    
    return results


def check_data_leakage() -> Dict[str, Any]:
    """Check for data leakage in preprocessing and splits."""
    print("\n" + "=" * 80)
    print("CHECKING FOR DATA LEAKAGE")
    print("=" * 80)
    
    leakage_report = {}
    
    # Check split leakage
    split_info = load_split_info()
    train_patients = set(split_info["train_patients"])
    test_patients = set(split_info["test_patients"])
    overlap = train_patients & test_patients
    
    leakage_report["split_leakage"] = {
        "has_leakage": len(overlap) > 0,
        "num_overlapping_patients": len(overlap),
        "overlap_percentage": len(overlap) / len(test_patients) * 100 if test_patients else 0,
    }
    
    # Check preprocessing code for leakage
    preprocessing_path = PROJECT_ROOT / "src" / "01_preprocessing_v2.py"
    if preprocessing_path.exists():
        with open(preprocessing_path, "r") as f:
            preprocessing_code = f.read()
        
        # Check for global normalization
        has_global_norm = "normalize(X)" in preprocessing_code and "fit on complete" in preprocessing_code.lower()
        
        # Check for global imputation
        has_global_impute = "median(X.reshape" in preprocessing_code or "mean(X.reshape" in preprocessing_code
        
        leakage_report["preprocessing_leakage"] = {
            "has_global_normalization": has_global_norm,
            "has_global_imputation": has_global_impute,
        }
    else:
        leakage_report["preprocessing_leakage"] = {
            "has_global_normalization": "unknown",
            "has_global_imputation": "unknown",
        }
    
    # Overall verdict
    has_leakage = (
        leakage_report["split_leakage"]["has_leakage"]
        or leakage_report["preprocessing_leakage"]["has_global_normalization"]
        or leakage_report["preprocessing_leakage"]["has_global_imputation"]
    )
    
    leakage_report["overall_verdict"] = "LEAKAGE DETECTED" if has_leakage else "NO LEAKAGE DETECTED"
    
    print(f"\nSplit leakage: {leakage_report['split_leakage']['has_leakage']}")
    print(f"Preprocessing leakage: {leakage_report['preprocessing_leakage']}")
    print(f"Overall verdict: {leakage_report['overall_verdict']}")
    
    return leakage_report


def verify_historical_scores(
    model_results: Dict[str, Dict[str, Any]], ensemble_results: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Verify historical reported scores against current evaluation."""
    print("\n" + "=" * 80)
    print("VERIFYING HISTORICAL SCORES")
    print("=" * 80)
    
    verification_results = {}
    
    # Map model names to historical scores
    model_mapping = {
        "xgb_baseline": "XGBoost",
        "xgb_final_v2": "XGBoost",
        "xgb_v2": "XGBoost",
        "xgboost_v2": "XGBoost",
        "lr_final_v2": "LogisticRegression",
        "lr_v2": "LogisticRegression",
        "logistic_regression_v2": "LogisticRegression",
    }
    
    for model_name, model_results_data in model_results.items():
        if not model_results_data.get("predictions_available"):
            continue
        
        # Find corresponding historical score
        hist_name = None
        for key, val in model_mapping.items():
            if key in model_name.lower():
                hist_name = val
                break
        
        if hist_name and hist_name in HISTORICAL_SCORES:
            historical_score = HISTORICAL_SCORES[hist_name]
            current_score = model_results_data["metrics_default"]["AUROC"]
            
            difference = abs(current_score - historical_score)
            within_tolerance = difference < 0.05  # 5% tolerance
            
            verification_results[model_name] = {
                "historical_name": hist_name,
                "historical_score": historical_score,
                "current_score": current_score,
                "difference": difference,
                "within_tolerance": within_tolerance,
                "verdict": "MATCH" if within_tolerance else "MISMATCH",
            }
            
            print(f"\n{model_name}:")
            print(f"  Historical: {historical_score:.4f}")
            print(f"  Current: {current_score:.4f}")
            print(f"  Difference: {difference:.4f}")
            print(f"  Verdict: {verification_results[model_name]['verdict']}")
    
    # Verify ensemble score
    if ensemble_results.get("predictions_available"):
        historical_ensemble = HISTORICAL_SCORES.get("Ensemble")
        if historical_ensemble:
            current_ensemble = ensemble_results["metrics_default"]["AUROC"]
            difference = abs(current_ensemble - historical_ensemble)
            within_tolerance = difference < 0.05
            
            verification_results["Ensemble_XGB_LR"] = {
                "historical_name": "Ensemble",
                "historical_score": historical_ensemble,
                "current_score": current_ensemble,
                "difference": difference,
                "within_tolerance": within_tolerance,
                "verdict": "MATCH" if within_tolerance else "MISMATCH",
            }
            
            print(f"\nEnsemble_XGB_LR:")
            print(f"  Historical: {historical_ensemble:.4f}")
            print(f"  Current: {current_ensemble:.4f}")
            print(f"  Difference: {difference:.4f}")
            print(f"  Verdict: {verification_results['Ensemble_XGB_LR']['verdict']}")
    
    return verification_results


def evaluate_clinical_baselines(
    X_test: np.ndarray, y_test: np.ndarray, feature_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """Evaluate clinical baseline rules."""
    print("\n" + "=" * 80)
    print("EVALUATING CLINICAL BASELINES")
    print("=" * 80)
    
    baseline_results = {}
    
    # Map feature names to indices (handle hour-suffixed features)
    feature_to_idx = {}
    for i, name in enumerate(feature_names):
        # Store both full name and base name (without hour suffix)
        feature_to_idx[name] = i
        if "_" in name:
            base_name = name.rsplit("_", 1)[0]
            feature_to_idx[base_name] = i
    
    # Define baseline rules
    def lactate_rule(X, threshold):
        """Predict sepsis if lactate > threshold."""
        lactate_idx = feature_to_idx.get("Lactate")
        if lactate_idx is None:
            print("  Warning: Lactate feature not found")
            return np.zeros(len(X))
        return (X[:, lactate_idx] > threshold).astype(int)
    
    def map_rule(X, threshold):
        """Predict sepsis if MAP < threshold."""
        map_idx = feature_to_idx.get("MAP")
        if map_idx is None:
            print("  Warning: MAP feature not found")
            return np.zeros(len(X))
        return (X[:, map_idx] < threshold).astype(int)
    
    def hr_rule(X, threshold):
        """Predict sepsis if HR > threshold."""
        hr_idx = feature_to_idx.get("HR")
        if hr_idx is None:
            print("  Warning: HR feature not found")
            return np.zeros(len(X))
        return (X[:, hr_idx] > threshold).astype(int)
    
    def o2sat_rule(X, threshold):
        """Predict sepsis if O2Sat < threshold."""
        o2sat_idx = feature_to_idx.get("O2Sat")
        if o2sat_idx is None:
            print("  Warning: O2Sat feature not found")
            return np.zeros(len(X))
        return (X[:, o2sat_idx] < threshold).astype(int)
    
    # Evaluate each baseline
    baselines = [
        ("Lactate > 4", lambda X: lactate_rule(X, 4.0)),
        ("Lactate > 8", lambda X: lactate_rule(X, 8.0)),
        ("MAP < 65", lambda X: map_rule(X, 65)),
        ("MAP < 50", lambda X: map_rule(X, 50)),
        ("HR > 130", lambda X: hr_rule(X, 130)),
        ("HR > 160", lambda X: hr_rule(X, 160)),
        ("O2Sat < 90", lambda X: o2sat_rule(X, 90)),
        ("O2Sat < 82", lambda X: o2sat_rule(X, 82)),
    ]
    
    for name, rule_func in baselines:
        try:
            y_pred = rule_func(X_test)
            if y_pred.sum() == 0:
                print(f"{name}: No positive predictions (feature may not be in test set)")
                baseline_results[name] = {"AUROC": 0.5, "PR-AUC": y_test.mean(), "error": "No positive predictions"}
            else:
                metrics = calculate_comprehensive_metrics(y_test, y_pred.astype(float), threshold=0.5)
                baseline_results[name] = metrics
                print(f"{name}: AUROC={metrics['AUROC']:.4f}, PR-AUC={metrics['PR-AUC']:.4f}")
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            baseline_results[name] = {"error": str(e)}
    
    return baseline_results


def evaluate_ensemble(
    model_results: Dict[str, Dict[str, Any]], X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, float]:
    """Evaluate ensemble of XGBoost and Logistic Regression."""
    print("\n" + "=" * 80)
    print("EVALUATING ENSEMBLE")
    print("=" * 80)
    
    # Find XGBoost and LR models with predictions
    xgb_pred = None
    lr_pred = None
    
    for model_name, data in model_results.items():
        if not data.get("predictions_available"):
            continue
        
        if "xgb" in model_name.lower() and "final" in model_name.lower():
            # Get predictions by loading the model again
            model_path = Path(data["model_path"])
            model = load_model(model_path)
            if model is not None:
                try:
                    if isinstance(model, dict) and "model" in model:
                        model = model["model"]
                    xgb_pred = model.predict_proba(X_test)[:, 1]
                    print(f"Using {model_name} for ensemble")
                except:
                    pass
        
        if "lr" in model_name.lower() and "final" in model_name.lower():
            model_path = Path(data["model_path"])
            model = load_model(model_path)
            if model is not None:
                try:
                    if isinstance(model, dict) and "model" in model:
                        model = model["model"]
                    lr_pred = model.predict_proba(X_test)[:, 1]
                    print(f"Using {model_name} for ensemble")
                except:
                    pass
    
    if xgb_pred is None or lr_pred is None:
        print("Could not load both XGBoost and LR models for ensemble")
        return {"error": "Could not load models"}
    
    # Rank-normalize predictions
    xgb_rank = rankdata(xgb_pred) / len(xgb_pred)
    lr_rank = rankdata(lr_pred) / len(lr_pred)
    
    # Ensemble by averaging ranks
    ensemble_pred = (xgb_rank + lr_rank) / 2
    
    # Calculate metrics
    optimal_threshold = find_optimal_threshold(y_test, ensemble_pred)
    metrics_default = calculate_comprehensive_metrics(y_test, ensemble_pred, threshold=0.5)
    metrics_optimal = calculate_comprehensive_metrics(y_test, ensemble_pred, threshold=optimal_threshold)
    
    print(f"Ensemble AUROC (default): {metrics_default['AUROC']:.4f}")
    print(f"Ensemble AUROC (optimal): {metrics_optimal['AUROC']:.4f}")
    print(f"Ensemble vs Historical XGBoost (0.8544): {metrics_default['AUROC']:.4f}")
    
    return {
        "predictions_available": True,
        "optimal_threshold": optimal_threshold,
        "metrics_default": metrics_default,
        "metrics_optimal": metrics_optimal,
    }


def create_excel_report(
    model_results: Dict[str, Dict[str, Any]],
    leakage_report: Dict[str, Any],
    verification_results: Dict[str, Dict[str, Any]],
    clinical_baselines: Dict[str, Dict[str, float]],
    ensemble_results: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Create comprehensive Excel report with 10 worksheets."""
    print("\n" + "=" * 80)
    print("CREATING EXCEL REPORT")
    print("=" * 80)
    
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        
        # Worksheet 1: Summary
        print("Creating Worksheet 1: Summary...")
        summary_data = []
        for model_name, data in model_results.items():
            if data.get("predictions_available"):
                metrics = data["metrics_default"]
                summary_data.append({
                    "Model": model_name,
                    "AUROC": metrics["AUROC"],
                    "PR-AUC": metrics["PR-AUC"],
                    "Accuracy": metrics["Accuracy"],
                    "F1": metrics["F1"],
                    "Optimal_Threshold": data["optimal_threshold"],
                })
        # Add ensemble if available
        if ensemble_results.get("predictions_available"):
            metrics = ensemble_results["metrics_default"]
            summary_data.append({
                "Model": "Ensemble_XGB_LR",
                "AUROC": metrics["AUROC"],
                "PR-AUC": metrics["PR-AUC"],
                "Accuracy": metrics["Accuracy"],
                "F1": metrics["F1"],
                "Optimal_Threshold": ensemble_results["optimal_threshold"],
            })
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        
        # Worksheet 2: Comprehensive Metrics (Default Threshold)
        print("Creating Worksheet 2: Comprehensive Metrics (Default)...")
        metrics_data = []
        for model_name, data in model_results.items():
            if data.get("predictions_available"):
                metrics = data["metrics_default"]
                row = {"Model": model_name}
                row.update(metrics)
                metrics_data.append(row)
        df_metrics = pd.DataFrame(metrics_data)
        df_metrics.to_excel(writer, sheet_name="Metrics_Default", index=False)
        
        # Worksheet 3: Comprehensive Metrics (Optimal Threshold)
        print("Creating Worksheet 3: Comprehensive Metrics (Optimal)...")
        metrics_opt_data = []
        for model_name, data in model_results.items():
            if data.get("predictions_available"):
                metrics = data["metrics_optimal"]
                row = {"Model": model_name}
                row.update(metrics)
                metrics_opt_data.append(row)
        df_metrics_opt = pd.DataFrame(metrics_opt_data)
        df_metrics_opt.to_excel(writer, sheet_name="Metrics_Optimal", index=False)
        
        # Worksheet 4: Data Leakage Report
        print("Creating Worksheet 4: Data Leakage Report...")
        leakage_data = []
        leakage_data.append({
            "Check": "Split Leakage",
            "Result": leakage_report["split_leakage"]["has_leakage"],
            "Details": f"{leakage_report['split_leakage']['num_overlapping_patients']} overlapping patients",
        })
        leakage_data.append({
            "Check": "Global Normalization",
            "Result": leakage_report["preprocessing_leakage"]["has_global_normalization"],
            "Details": "N/A",
        })
        leakage_data.append({
            "Check": "Global Imputation",
            "Result": leakage_report["preprocessing_leakage"]["has_global_imputation"],
            "Details": "N/A",
        })
        leakage_data.append({
            "Check": "Overall Verdict",
            "Result": leakage_report["overall_verdict"],
            "Details": "N/A",
        })
        df_leakage = pd.DataFrame(leakage_data)
        df_leakage.to_excel(writer, sheet_name="Data_Leakage", index=False)
        
        # Worksheet 5: Historical Score Verification
        print("Creating Worksheet 5: Historical Score Verification...")
        verification_data = []
        for model_name, data in verification_results.items():
            verification_data.append({
                "Model": model_name,
                "Historical_Name": data["historical_name"],
                "Historical_Score": data["historical_score"],
                "Current_Score": data["current_score"],
                "Difference": data["difference"],
                "Verdict": data["verdict"],
            })
        df_verification = pd.DataFrame(verification_data)
        df_verification.to_excel(writer, sheet_name="Historical_Verification", index=False)
        
        # Worksheet 6: Clinical Baselines
        print("Creating Worksheet 6: Clinical Baselines...")
        baseline_data = []
        for name, metrics in clinical_baselines.items():
            if "error" not in metrics:
                row = {"Baseline": name}
                row.update(metrics)
                baseline_data.append(row)
        df_baselines = pd.DataFrame(baseline_data)
        df_baselines.to_excel(writer, sheet_name="Clinical_Baselines", index=False)
        
        # Worksheet 7: Model Comparison
        print("Creating Worksheet 7: Model Comparison...")
        comparison_data = []
        for model_name, data in model_results.items():
            if data.get("predictions_available"):
                metrics_def = data["metrics_default"]
                metrics_opt = data["metrics_optimal"]
                comparison_data.append({
                    "Model": model_name,
                    "AUROC_Default": metrics_def["AUROC"],
                    "AUROC_Optimal": metrics_opt["AUROC"],
                    "AUROC_Improvement": metrics_opt["AUROC"] - metrics_def["AUROC"],
                    "F1_Default": metrics_def["F1"],
                    "F1_Optimal": metrics_opt["F1"],
                    "F1_Improvement": metrics_opt["F1"] - metrics_def["F1"],
                })
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison.to_excel(writer, sheet_name="Model_Comparison", index=False)
        
        # Worksheet 8: Data Statistics
        print("Creating Worksheet 8: Data Statistics...")
        stats_data = [
            {"Metric": "Test Samples", "Value": len(y_test)},
            {"Metric": "Test Positives", "Value": int(y_test.sum())},
            {"Metric": "Test Negatives", "Value": int((1 - y_test).sum())},
            {"Metric": "Prevalence", "Value": y_test.mean()},
            {"Metric": "Features", "Value": X_test.shape[1]},
            {"Metric": "Ablated Features", "Value": len(ABLATED_FEATURES)},
        ]
        df_stats = pd.DataFrame(stats_data)
        df_stats.to_excel(writer, sheet_name="Data_Statistics", index=False)
        
        # Worksheet 9: Confusion Matrices
        print("Creating Worksheet 9: Confusion Matrices...")
        cm_data = []
        for model_name, data in model_results.items():
            if data.get("predictions_available"):
                metrics = data["metrics_default"]
                cm_data.append({
                    "Model": model_name,
                    "TP": metrics["TP"],
                    "TN": metrics["TN"],
                    "FP": metrics["FP"],
                    "FN": metrics["FN"],
                })
        df_cm = pd.DataFrame(cm_data)
        df_cm.to_excel(writer, sheet_name="Confusion_Matrices", index=False)
        
        # Worksheet 10: Audit Metadata
        print("Creating Worksheet 10: Audit Metadata...")
        metadata_data = [
            {"Key": "Audit Date", "Value": time.strftime("%Y-%m-%d %H:%M:%S")},
            {"Key": "Project Path", "Value": str(PROJECT_ROOT)},
            {"Key": "Data Path", "Value": str(DATA_PATH)},
            {"Key": "Models Directory", "Value": str(MODELS_DIR)},
            {"Key": "Output File", "Value": str(OUTPUT_EXCEL)},
            {"Key": "Number of Models Audited", "Value": len(model_results)},
            {"Key": "Number of Models with Predictions", "Value": sum(1 for d in model_results.values() if d.get("predictions_available"))},
            {"Key": "Data Leakage Detected", "Value": leakage_report["overall_verdict"]},
            {"Key": "Ensemble Evaluated", "Value": ensemble_results.get("predictions_available", False)},
            {"Key": "Ensemble AUROC", "Value": ensemble_results.get("metrics_default", {}).get("AUROC", "N/A")},
        ]
        df_metadata = pd.DataFrame(metadata_data)
        df_metadata.to_excel(writer, sheet_name="Audit_Metadata", index=False)
    
    print(f"\nExcel report saved to: {OUTPUT_EXCEL}")


def main():
    """Main audit function."""
    start_time = time.time()
    
    print("=" * 80)
    print("COMPREHENSIVE ML AUDIT FOR SEPSIS PREDICTION PROJECT")
    print("=" * 80)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    X_test, y_test, patient_ids_test, feature_names = load_data()
    
    # Remove ablated features
    X_test_clean, feature_names_clean = remove_ablated_features(X_test, feature_names)
    
    # Check for data leakage
    leakage_report = check_data_leakage()
    
    # Audit all models
    model_results = audit_all_models(X_test_clean, y_test, feature_names_clean)
    
    # Evaluate clinical baselines
    clinical_baselines = evaluate_clinical_baselines(X_test_clean, y_test, feature_names_clean)
    
    # Evaluate ensemble
    ensemble_results = evaluate_ensemble(model_results, X_test_clean, y_test)
    
    # Verify historical scores (after ensemble is evaluated)
    verification_results = verify_historical_scores(model_results, ensemble_results)
    
    # Create Excel report
    create_excel_report(
        model_results,
        leakage_report,
        verification_results,
        clinical_baselines,
        ensemble_results,
        X_test_clean,
        y_test,
    )
    
    # Final summary
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Report saved to: {OUTPUT_EXCEL}")
    print(f"\nIMPORTANT NOTES:")
    print(f"- If data leakage was detected, results should be marked as NOT VALIDLY EVALUATED")
    print(f"- Historical score verification shows any discrepancies")
    print(f"- Clinical baselines provide context for model performance")
    print(f"- All metrics are calculated on the held-out test set")
    
    # Add validity warning if leakage detected
    if leakage_report["overall_verdict"] == "LEAKAGE DETECTED":
        print("\n" + "!" * 80)
        print("WARNING: DATA LEAKAGE DETECTED - RESULTS SHOULD BE MARKED AS NOT VALIDLY EVALUATED")
        print("!" * 80)


if __name__ == "__main__":
    main()
