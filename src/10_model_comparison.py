"""Compare all four trained baseline models (RF, XGBoost, GB, AdaBoost)."""

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
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "data.npz"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
PLOT_DIR = PROJECT_ROOT / "outputs" / "plots"
COMPARISON_JSON = PROJECT_ROOT / "outputs" / "model_comparison.json"
COMPARISON_CSV = PROJECT_ROOT / "outputs" / "model_comparison.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
EXCLUDED_FEATURES = {"ICULOS"}
INPUT_FEATURES = ("HR", "Temp", "Resp", "MAP", "O2Sat", "Lactate")
BASELINE = {"HR": 75.0, "Temp": 37.0, "Resp": 14.0, "MAP": 85.0, "O2Sat": 98.0, "Lactate": 1.0}

MODEL_CONFIGS = {
    "RandomForest": {
        "path": MODEL_DIR / "rf_baseline.pkl",
        "monotonic": False,
        "uses_iculos": False,
        "cv_full_dataset": False,  # CV on subsample for speed
    },
    "XGBoost": {
        "path": MODEL_DIR / "xgb_baseline.pkl",
        "monotonic": True,
        "uses_iculos": False,
        "cv_full_dataset": True,
    },
    "GradientBoosting": {
        "path": MODEL_DIR / "gb_baseline.pkl",
        "monotonic": False,
        "uses_iculos": False,
        "cv_full_dataset": False,  # CV on subsample
    },
    "AdaBoost": {
        "path": MODEL_DIR / "adaboost_baseline.pkl",
        "monotonic": False,
        "uses_iculos": False,
        "cv_full_dataset": False,  # CV on subsample
    },
}


def load_data_and_split() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str], int]:
    """Load preprocessed data and create the same train/test split used by all models."""
    with np.load(DATA_PATH, allow_pickle=False) as data:
        X = data["X"]
        y = data["y"]
        feature_names = data["feature_names"].astype(str).tolist()

    # Return full X and y, along with all feature names
    # Each model will select its own features
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_names, X.shape[1]


def get_positive_probability(model: object, X: np.ndarray) -> np.ndarray:
    """Extract positive class probability from model predictions."""
    positive_index = int(np.where(model.classes_ == 1)[0][0])
    return model.predict_proba(X)[:, positive_index]


def evaluate_model_on_test(
    model: object, X_test: np.ndarray, y_test: np.ndarray, selected_names: list[str], raw_names: list[str]
) -> dict[str, float]:
    """Evaluate a model on the held-out test set."""
    # Select the features the model was trained on
    selected_indices = [raw_names.index(name) for name in selected_names]
    X_test_selected = X_test[:, :, selected_indices].reshape(len(X_test), -1)

    probabilities = get_positive_probability(model, X_test_selected)
    predictions = model.predict(X_test_selected)

    return {
        "ROC-AUC": float(roc_auc_score(y_test, probabilities)),
        "PR-AUC": float(average_precision_score(y_test, probabilities)),
        "F1 (positive)": float(f1_score(y_test, predictions, pos_label=1, zero_division=0)),
    }


def run_sanity_check(
    model: object,
    medians: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    selected_names: list[str],
    raw_names: list[str],
    sequence_length: int,
) -> dict[str, float]:
    """Run the same sanity-check protocol used by src/05_sanity_checks.py."""
    indices = {name: selected_names.index(name) for name in INPUT_FEATURES if name in selected_names}

    scenarios = [
        ("Baseline", BASELINE),
        ("Lactate 4", {**BASELINE, "Lactate": 4.0}),
        ("Lactate 8", {**BASELINE, "Lactate": 8.0}),
        ("MAP 65", {**BASELINE, "MAP": 65.0}),
        ("MAP 50", {**BASELINE, "MAP": 50.0}),
        ("HR 130", {**BASELINE, "HR": 130.0}),
        ("HR 160", {**BASELINE, "HR": 160.0}),
        ("O2Sat 90", {**BASELINE, "O2Sat": 90.0}),
        ("O2Sat 82", {**BASELINE, "O2Sat": 82.0}),
        ("All abnormal", {"HR": 150.0, "Temp": 39.5, "Resp": 32.0, "MAP": 55.0, "O2Sat": 88.0, "Lactate": 6.0}),
    ]

    scores = {}
    for name, vitals in scenarios:
        raw_sequence = np.tile(medians, (sequence_length, 1))
        for feature_name, value in vitals.items():
            if feature_name in indices:
                raw_sequence[:, indices[feature_name]] = value
        model_input = ((raw_sequence - means) / stds).reshape(1, -1)
        prob = get_positive_probability(model, model_input)[0]
        scores[name] = float(prob)

    return scores


def count_sanity_check_passes(scores: dict[str, float], selected_names: list[str]) -> int:
    """Count how many of the 5 direction checks pass."""
    checks = []
    if "Lactate" in selected_names:
        checks.append(("Lactate", [scores["Baseline"], scores["Lactate 4"], scores["Lactate 8"]]))
    if "MAP" in selected_names:
        checks.append(("MAP", [scores["Baseline"], scores["MAP 65"], scores["MAP 50"]]))
    if "HR" in selected_names:
        checks.append(("HR", [scores["Baseline"], scores["HR 130"], scores["HR 160"]]))
    if "O2Sat" in selected_names:
        checks.append(("O2Sat", [scores["Baseline"], scores["O2Sat 90"], scores["O2Sat 82"]]))

    passes = 0
    for name, values in checks:
        if np.all(np.diff(values) >= 0):
            passes += 1

    # Check if "All abnormal" is highest
    other_maximum = max(value for key, value in scores.items() if key != "All abnormal")
    if scores["All abnormal"] > other_maximum:
        passes += 1

    return passes


def main() -> None:
    print("Loading data and creating train/test split...")
    X_train, X_test, y_train, y_test, raw_names, sequence_length = load_data_and_split()

    with np.load(DATA_PATH, allow_pickle=False) as data:
        raw_medians = data["feature_medians"]
        raw_means = data["feature_means"]
        raw_stds = np.where(data["feature_stds"] == 0, 1.0, data["feature_stds"])

    print(f"Test set size: {len(X_test):,} samples")
    print(f"Total features: {len(raw_names)} × {sequence_length} hours = {len(raw_names) * sequence_length} total\n")

    # Load existing metrics for XGBoost and the new models
    xgb_metrics = json.loads((MODEL_DIR / "xgb_baseline_metrics.json").read_text(encoding="utf-8"))
    additional_metrics = json.loads((MODEL_DIR / "additional_baseline_metrics.json").read_text(encoding="utf-8"))

    comparison_data = []

    for model_name, config in MODEL_CONFIGS.items():
        print(f"Loading {model_name}...")
        model = joblib.load(config["path"])

        # Get the model's selected feature names
        selected_names = model.selected_feature_names_.astype(str).tolist()
        selected_indices = [raw_names.index(name) for name in selected_names]
        medians = raw_medians[selected_indices]
        means = raw_means[selected_indices]
        stds = raw_stds[selected_indices]

        print(f"  Features: {len(selected_names)} × {sequence_length} = {len(selected_names) * sequence_length} flattened")

        # Evaluate on test set (fair comparison across all models)
        test_metrics = evaluate_model_on_test(model, X_test, y_test, selected_names, raw_names)

        # Get CV metrics from saved files if available
        if model_name == "XGBoost":
            cv_metrics = {
                "roc_auc_mean": xgb_metrics["cross_validation"]["roc_auc_mean"],
                "roc_auc_std": xgb_metrics["cross_validation"]["roc_auc_std"],
                "f1_mean": xgb_metrics["cross_validation"]["f1_mean"],
                "f1_std": xgb_metrics["cross_validation"]["f1_std"],
            }
        elif model_name in additional_metrics:
            cv_metrics = additional_metrics[model_name]["cross_validation"]
        else:
            # RandomForest - no saved CV metrics, evaluate on subsample for speed
            print(f"  Computing 3-fold CV for {model_name} on subsample (~2000 samples)...")
            X_train_selected = X_train[:, :, selected_indices].reshape(len(X_train), -1)
            X_test_selected = X_test[:, :, selected_indices].reshape(len(X_test), -1)
            X_flat_full = np.vstack([X_train_selected, X_test_selected])
            y_full = np.concatenate([y_train, y_test])

            # Subsample for CV
            cv_sample_size = 2000
            if len(X_flat_full) > cv_sample_size:
                X_cv, _, y_cv, _ = train_test_split(
                    X_flat_full, y_full, train_size=cv_sample_size, random_state=RANDOM_STATE, stratify=y_full
                )
                print(f"    CV using subsample: {len(X_cv):,} samples")
            else:
                X_cv, y_cv = X_flat_full, y_full

            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
            aucs = []
            f1s = []
            for train_idx, val_idx in cv.split(X_cv, y_cv):
                cv_model = type(model)(
                    **{k: v for k, v in model.get_params().items() if k != "selected_feature_names_" and k != "sequence_length_"}
                )
                cv_model.random_state = RANDOM_STATE
                cv_model.fit(X_cv[train_idx], y_cv[train_idx])
                probs = get_positive_probability(cv_model, X_cv[val_idx])
                preds = cv_model.predict(X_cv[val_idx])
                aucs.append(float(roc_auc_score(y_cv[val_idx], probs)))
                f1s.append(float(f1_score(y_cv[val_idx], preds, pos_label=1, zero_division=0)))

            cv_metrics = {
                "roc_auc_mean": float(np.mean(aucs)),
                "roc_auc_std": float(np.std(aucs)),
                "f1_mean": float(np.mean(f1s)),
                "f1_std": float(np.std(f1s)),
            }

        # Run sanity checks
        sanity_scores = run_sanity_check(model, medians, means, stds, selected_names, raw_names, sequence_length)
        sanity_passes = count_sanity_check_passes(sanity_scores, selected_names)

        # Count total possible checks (direction checks + "all abnormal" check)
        total_checks = sum(1 for feat in ["Lactate", "MAP", "HR", "O2Sat"] if feat in selected_names) + 1

        comparison_data.append({
            "Model": model_name,
            "ROC-AUC": test_metrics["ROC-AUC"],
            "PR-AUC": test_metrics["PR-AUC"],
            "F1 (positive)": test_metrics["F1 (positive)"],
            "CV-AUC (mean±std)": f"{cv_metrics['roc_auc_mean']:.4f} ± {cv_metrics['roc_auc_std']:.4f}",
            "CV-F1 (mean±std)": f"{cv_metrics['f1_mean']:.4f} ± {cv_metrics['f1_std']:.4f}",
            "Monotonic constraints?": "Yes" if config["monotonic"] else "No",
            "Uses ICULOS?": "Yes" if config["uses_iculos"] else "No",
            "CV data source": "Full dataset" if config["cv_full_dataset"] else "Stratified subsample (~2000) (for speed)",
            "Sanity check passes": f"{sanity_passes}/{total_checks}",
        })

        print(f"  Test ROC-AUC: {test_metrics['ROC-AUC']:.4f}")
        print(f"  Test PR-AUC: {test_metrics['PR-AUC']:.4f}")
        print(f"  Test F1: {test_metrics['F1 (positive)']:.4f}")
        print(f"  Sanity checks: {sanity_passes}/5 passed\n")

    # Create comparison table
    df = pd.DataFrame(comparison_data)

    print("\n" + "=" * 120)
    print("MODEL COMPARISON TABLE")
    print("=" * 120)
    print(df.to_string(index=False))
    print("=" * 120)
    print("\nNOTE: CV metrics for RandomForest, GradientBoosting, and AdaBoost were computed on")
    print("      a stratified subsample (~2000 samples) for computational feasibility, while")
    print("      XGBoost used the full dataset. TEST SET metrics are the fairer comparison across")
    print("      models since all four were evaluated on the same held-out test set.")
    print("=" * 120 + "\n")

    # Save comparison table
    COMPARISON_JSON.parent.mkdir(parents=True, exist_ok=True)
    COMPARISON_JSON.write_text(json.dumps(comparison_data, indent=2), encoding="utf-8")
    print(f"Saved comparison JSON to {COMPARISON_JSON}")

    df.to_csv(COMPARISON_CSV, index=False)
    print(f"Saved comparison CSV to {COMPARISON_CSV}\n")

    # Create PR-AUC bar chart
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#E45756", "#0D7381", "#4C78A8", "#F58518"]
    bars = ax.bar(df["Model"], df["PR-AUC"], color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)

    ax.set_ylabel("PR-AUC (Average Precision)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_title("Precision-Recall AUC Comparison Across Models", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(df["PR-AUC"]) * 1.2)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on bars
    for bar, value in zip(bars, df["PR-AUC"]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    fig.tight_layout()
    plot_path = PLOT_DIR / "model_comparison_pr_auc.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PR-AUC comparison plot to {plot_path}\n")

    # Print sanity-check summary table
    print("SANITY-CHECK PASS/FAIL SUMMARY")
    print("=" * 80)
    sanity_summary = df[["Model", "Sanity check passes"]]
    print(sanity_summary.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
