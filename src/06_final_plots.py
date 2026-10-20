"""Generate presentation-ready evaluation and explainability figures."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib"))

import matplotlib.pyplot as plt
import seaborn as sns
import shap


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "data.npz"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "xgb_baseline.pkl"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "plots" / "final"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def set_plot_style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.color": "#DCE7EA",
            "grid.linewidth": 0.8,
            "axes.edgecolor": "#AABCC1",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#243B45",
            "xtick.color": "#405860",
            "ytick.color": "#405860",
        },
    )


def get_tree_estimators(model: object) -> list[object]:
    calibrated = getattr(model, "calibrated_classifiers_", None)
    if calibrated is not None:
        return [classifier.estimator for classifier in calibrated]
    return [model]


def shap_positive_values(values: object) -> np.ndarray:
    if isinstance(values, list):
        return np.asarray(values[1])
    array = np.asarray(values)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[-1] == 2:
        return array[:, :, 1]
    if array.ndim == 3 and array.shape[1] == 2:
        return array[:, 1, :]
    raise ValueError(f"Unexpected SHAP output shape: {array.shape}")


def main() -> None:
    if not DATA_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError("Run preprocessing and src/02_ml_baseline.py first.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    model = joblib.load(MODEL_PATH)
    with np.load(DATA_PATH, allow_pickle=False) as data:
        X = data["X"]
        y = data["y"]
        raw_names = data["feature_names"].astype(str).tolist()
    selected_names = model.selected_feature_names_.astype(str).tolist()
    selected_indices = [raw_names.index(name) for name in selected_names]
    X_flat = X[:, :, selected_indices].reshape(len(X), -1)
    _, X_test, _, y_test = train_test_split(
        X_flat, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)

    # ROC curve.
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    roc_auc = auc(fpr, tpr)
    figure, axis = plt.subplots(figsize=(8.5, 7))
    axis.plot(fpr, tpr, color="#0D7381", linewidth=3.2, label=f"Calibrated XGBoost (AUC = {roc_auc:.3f})")
    axis.plot([0, 1], [0, 1], color="#8B9BA0", linestyle="--", linewidth=1.6, label="No-skill reference")
    axis.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate", title="ROC curve — ICULOS-free model")
    axis.legend(loc="lower right", frameon=True, facecolor="white")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "01_roc_curve.png", dpi=260, bbox_inches="tight")
    plt.close(figure)

    # Precision-recall curve.
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    pr_auc = auc(recall[::-1], precision[::-1])
    prevalence = y_test.mean()
    figure, axis = plt.subplots(figsize=(8.5, 7))
    axis.plot(recall, precision, color="#D65A5A", linewidth=3.2, label=f"Calibrated XGBoost (PR-AUC = {pr_auc:.3f})")
    axis.axhline(prevalence, color="#8B9BA0", linestyle="--", linewidth=1.6, label=f"No-skill prevalence ({prevalence:.1%})")
    axis.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision", title="Precision–Recall curve — ICULOS-free model")
    axis.legend(loc="upper right", frameon=True, facecolor="white")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "02_precision_recall_curve.png", dpi=260, bbox_inches="tight")
    plt.close(figure)

    # Confusion matrix.
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(8, 6.8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap=sns.light_palette("#0D7381", as_cmap=True),
        cbar_kws={"label": "Window count"},
        linewidths=2,
        linecolor="white",
        square=True,
        xticklabels=["No Sepsis", "Sepsis"],
        yticklabels=["No Sepsis", "Sepsis"],
        annot_kws={"size": 20, "weight": "bold"},
        ax=axis,
    )
    axis.set(xlabel="Predicted class", ylabel="Actual class", title="Confusion matrix — ICULOS-free model")
    axis.tick_params(axis="x", rotation=0)
    axis.tick_params(axis="y", rotation=0)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "03_confusion_matrix.png", dpi=260, bbox_inches="tight")
    plt.close(figure)

    # SHAP summary: average values from the five calibrated tree estimators.
    shap_values = np.mean(
        [shap_positive_values(shap.TreeExplainer(estimator).shap_values(X_test)) for estimator in get_tree_estimators(model)],
        axis=0,
    )
    sequence_length = X.shape[1]
    flat_names = [
        f"{feature} (t-{sequence_length - hour - 1})"
        for hour in range(sequence_length)
        for feature in selected_names
    ]
    importances = np.abs(shap_values).mean(axis=0)
    order = np.argsort(importances)[-15:]
    figure, axis = plt.subplots(figsize=(11, 8.5))
    colors = sns.color_palette("crest", n_colors=15)
    axis.barh(np.arange(15), importances[order], color=colors)
    axis.set_yticks(np.arange(15), [flat_names[index] for index in order])
    axis.set(xlabel="Mean absolute SHAP value", title="Top 15 feature influences — ICULOS-free model")
    axis.grid(axis="y", visible=False)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "04_shap_summary_top15.png", dpi=260, bbox_inches="tight")
    plt.close(figure)

    # Leakage-fix comparison. The old model did not report PR-AUC, so mark it as unavailable.
    metric_names = ["ROC-AUC", "F1 (Sepsis)", "PR-AUC"]
    old_values = [0.9998, 0.9434, np.nan]
    new_values = [roc_auc, 0.8514, pr_auc]
    positions = np.arange(len(metric_names))
    width = 0.34
    figure, axis = plt.subplots(figsize=(10, 7))
    old_bars = axis.bar(positions - width / 2, np.nan_to_num(old_values, nan=0), width, color="#A9B9BE", label="Previous model (with ICULOS)")
    new_bars = axis.bar(positions + width / 2, new_values, width, color="#0D7381", label="Leakage-reduced model (without ICULOS)")
    for index, bar in enumerate(old_bars):
        if np.isnan(old_values[index]):
            axis.text(bar.get_x() + bar.get_width() / 2, 0.025, "Not reported", ha="center", va="bottom", rotation=90, color="#5C7078", fontsize=11, fontweight="bold")
        else:
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{old_values[index]:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    for bar, value in zip(new_bars, new_values):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    axis.set(xticks=positions, xticklabels=metric_names, ylim=(0, 1.13), ylabel="Score", title="Performance before and after removing ICULOS")
    axis.legend(loc="upper left", frameon=True, facecolor="white")
    axis.grid(axis="x", visible=False)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "05_leakage_fix_comparison.png", dpi=260, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved 5 presentation-ready plots to {OUTPUT_DIR}")
    print(f"ROC-AUC={roc_auc:.4f}; PR-AUC={pr_auc:.4f}; confusion matrix={matrix.tolist()}")


if __name__ == "__main__":
    main()
