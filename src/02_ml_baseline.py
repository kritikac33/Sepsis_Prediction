"""Train and evaluate a calibrated, monotonic XGBoost sepsis baseline."""

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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from xgboost import XGBClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.20
EXCLUDED_FEATURES = {"ICULOS"}  # Avoid ICU length-of-stay leakage in the prototype model.
MONOTONIC_DIRECTIONS = {
    "HR": 1,
    "Temp": 1,
    "Resp": 1,
    "MAP": -1,
    "O2Sat": -1,
    "Lactate": 1,
}
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "data.npz"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "xgb_baseline.pkl"
ROC_PATH = PROJECT_ROOT / "outputs" / "plots" / "xgb_roc.png"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "outputs" / "plots" / "xgb_confusion_matrix.png"
PR_CURVE_PATH = PROJECT_ROOT / "outputs" / "plots" / "xgb_precision_recall.png"
METRICS_PATH = PROJECT_ROOT / "outputs" / "models" / "xgb_baseline_metrics.json"


def make_model(monotone_constraints: tuple[int, ...], scale_pos_weight: float) -> XGBClassifier:
    """Create a compact XGBoost classifier with clinically directed constraints."""
    return XGBClassifier(
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


def make_calibrated_model(monotone_constraints: tuple[int, ...], scale_pos_weight: float) -> CalibratedClassifierCV:
    """Apply Platt scaling without changing the constrained tree model's ranking."""
    return CalibratedClassifierCV(
        estimator=make_model(monotone_constraints, scale_pos_weight),
        method="sigmoid",
        cv=5,
        n_jobs=1,
    )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {DATA_PATH}. Run src/01_preprocessing.py first."
        )

    with np.load(DATA_PATH, allow_pickle=False) as data:
        X = data["X"]
        y = data["y"]
        feature_names = data["feature_names"].astype(str).tolist()

    if X.ndim != 3:
        raise ValueError(f"Expected 3D sequences (samples, hours, features), got X={X.shape}")
    if len(X) != len(y):
        raise ValueError(f"X and y have different sample counts: {len(X)} and {len(y)}")
    if len(np.unique(y)) != 2:
        raise ValueError("Both sepsis and non-sepsis samples are required for a stratified split.")

    model_feature_indices = [
        index for index, name in enumerate(feature_names) if name not in EXCLUDED_FEATURES
    ]
    model_feature_names = [feature_names[index] for index in model_feature_indices]
    if not model_feature_indices:
        raise ValueError("No model features remain after exclusion.")

    # Exclude ICULOS before flattening so every timestep is leakage-free.
    X_model = X[:, :, model_feature_indices]
    X_flat = X_model.reshape(X_model.shape[0], -1)
    monotone_constraints = tuple(
        MONOTONIC_DIRECTIONS.get(feature, 0)
        for _hour in range(X.shape[1])
        for feature in model_feature_names
    )
    print(f"Loaded sequences: X={X.shape}, y={y.shape}")
    print(f"Excluded features: {', '.join(sorted(EXCLUDED_FEATURES))}")
    print(f"Model features: {len(model_feature_names)} of {len(feature_names)}")
    print(f"Monotone clinical features: {', '.join(MONOTONIC_DIRECTIONS)}")
    print(f"Classical ML input after flattening: {X_flat.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_flat,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Train/test split: {len(X_train):,}/{len(X_test):,}")

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    model = make_calibrated_model(monotone_constraints, scale_pos_weight)
    print("Training monotonic XGBoost baseline with 5-fold Platt calibration...")
    model.fit(X_train, y_train)
    # Store the feature contract so downstream SHAP/API consumers use the same columns.
    model.selected_feature_names_ = np.asarray(model_feature_names)
    model.sequence_length_ = X.shape[1]

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "PR-AUC (average precision)": average_precision_score(y_test, probabilities),
    }

    print("\nTest metrics")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    matrix_table = pd.DataFrame(
        matrix,
        index=pd.Index(["Actual: No Sepsis", "Actual: Sepsis"], name=""),
        columns=pd.Index(["Predicted: No Sepsis", "Predicted: Sepsis"], name=""),
    )
    print("\nConfusion matrix (raw counts)")
    print(matrix_table.to_string())

    per_class = pd.DataFrame(
        {
            "Class": ["No Sepsis (0)", "Sepsis (1)"],
            "Precision": precision_score(y_test, predictions, labels=[0, 1], average=None, zero_division=0),
            "Recall": recall_score(y_test, predictions, labels=[0, 1], average=None, zero_division=0),
            "F1": f1_score(y_test, predictions, labels=[0, 1], average=None, zero_division=0),
            "Support": np.bincount(y_test, minlength=2),
        }
    )
    print("\nPer-class test metrics")
    print(per_class.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("\nRunning 5-fold stratified cross-validation...")
    cross_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scale_pos_weight = float((y == 0).sum() / (y == 1).sum())
    cv_model = make_calibrated_model(monotone_constraints, cv_scale_pos_weight)
    cv_scores = cross_validate(
        cv_model,
        X_flat,
        y,
        cv=cross_validator,
        scoring={"auc": "roc_auc", "f1": "f1"},
        n_jobs=1,
    )
    print(
        "5-fold CV ROC-AUC: "
        f"{cv_scores['test_auc'].mean():.4f} ± {cv_scores['test_auc'].std():.4f}"
    )
    print(
        "5-fold CV F1: "
        f"{cv_scores['test_f1'].mean():.4f} ± {cv_scores['test_f1'].std():.4f}"
    )

    # Keep a compact, reproducible evaluation record for reports and model cards.
    evaluation_record = {
        "dataset": {"samples": int(len(y)), "positive_samples": int(y.sum())},
        "test": {
            **{name: float(value) for name, value in metrics.items()},
            "confusion_matrix": matrix.tolist(),
            "per_class": per_class.to_dict(orient="records"),
        },
        "cross_validation": {
            "folds": 5,
            "roc_auc_mean": float(cv_scores["test_auc"].mean()),
            "roc_auc_std": float(cv_scores["test_auc"].std()),
            "f1_mean": float(cv_scores["test_f1"].mean()),
            "f1_std": float(cv_scores["test_f1"].std()),
        },
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(evaluation_record, indent=2), encoding="utf-8")
    print(f"Saved evaluation metrics to {METRICS_PATH}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot(fpr, tpr, color="#E45756", linewidth=2, label=f"XGBoost (AUC = {metrics['ROC-AUC']:.3f})")
    axis.plot([0, 1], [0, 1], color="gray", linestyle="--", label="No-skill classifier")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="True positive rate", title="Monotonic XGBoost ROC curve")
    axis.legend(loc="lower right")
    figure.tight_layout()
    ROC_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(ROC_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved ROC curve to {ROC_PATH}")

    figure, axis = plt.subplots(figsize=(7, 6))
    heatmap = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(heatmap, ax=axis, label="Count")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["No Sepsis", "Sepsis"],
        yticklabels=["No Sepsis", "Sepsis"],
        xlabel="Predicted label",
        ylabel="Actual label",
        title="Monotonic XGBoost confusion matrix",
    )
    for row, column in np.ndindex(matrix.shape):
        axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color="white" if matrix[row, column] > matrix.max() / 2 else "black")
    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved confusion matrix heatmap to {CONFUSION_MATRIX_PATH}")

    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot(recall, precision, color="#0D7381", linewidth=2, label=f"XGBoost (AP = {metrics['PR-AUC (average precision)']:.3f})")
    axis.axhline(y_test.mean(), color="gray", linestyle="--", label=f"No-skill prevalence ({y_test.mean():.3f})")
    axis.set(xlim=(0, 1), ylim=(0, 1.05), xlabel="Recall", ylabel="Precision", title="Monotonic XGBoost precision-recall curve")
    axis.legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(PR_CURVE_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved precision-recall curve to {PR_CURVE_PATH}")


if __name__ == "__main__":
    main()
