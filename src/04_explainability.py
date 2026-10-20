"""Produce a focused SHAP feature-importance summary for the RF baseline."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib"))

import matplotlib.pyplot as plt
import shap


RANDOM_STATE = 42  # Matches src/02_ml_baseline.py.
TEST_SIZE = 0.20
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "data.npz"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "xgb_baseline.pkl"
PLOT_PATH = PROJECT_ROOT / "outputs" / "plots" / "shap_summary.png"


def positive_class_shap_values(values: object) -> np.ndarray:
    """Return the positive-class SHAP matrix across SHAP's supported output formats."""
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


def get_tree_estimators(model: object) -> list[object]:
    """Access the constrained tree estimators inside an optional calibration wrapper."""
    calibrated = getattr(model, "calibrated_classifiers_", None)
    if calibrated is not None:
        return [classifier.estimator for classifier in calibrated]
    return [model]


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found at {DATA_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Random Forest model not found at {MODEL_PATH}. Run src/02_ml_baseline.py first.")

    model = joblib.load(MODEL_PATH)
    with np.load(DATA_PATH, allow_pickle=False) as data:
        X = data["X"]
        y = data["y"]
        feature_names = data["feature_names"].astype(str).tolist()

    if X.ndim != 3:
        raise ValueError(f"Expected sequence data with shape (samples, hours, features), got {X.shape}")
    if not hasattr(model, "selected_feature_names_"):
        raise ValueError("Model has no saved feature schema. Re-run src/02_ml_baseline.py.")
    model_feature_names = model.selected_feature_names_.astype(str).tolist()
    model_feature_indices = [feature_names.index(name) for name in model_feature_names]
    X_model = X[:, :, model_feature_indices]
    X_flat = X_model.reshape(X_model.shape[0], -1)
    _, X_test, _, _ = train_test_split(
        X_flat, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    sequence_length = X_model.shape[1]
    flattened_names = [
        f"{feature} (t-{sequence_length - hour - 1})"
        for hour in range(sequence_length)
        for feature in model_feature_names
    ]
    if len(flattened_names) != X_test.shape[1]:
        raise ValueError("Saved feature names do not match the flattened model input.")

    print(f"Explaining {len(X_test):,} held-out windows with SHAP TreeExplainer...")
    shap_values = np.mean(
        [
            positive_class_shap_values(shap.TreeExplainer(estimator).shap_values(X_test))
            for estimator in get_tree_estimators(model)
        ],
        axis=0,
    )

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_signed_shap = shap_values.mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-10:][::-1]
    print("\nTop 10 features (ranked by mean absolute SHAP value):")
    for rank, index in enumerate(top_indices, start=1):
        print(
            f"{rank}. {flattened_names[index]}: "
            f"mean |SHAP|={mean_abs_shap[index]:.6f}, mean SHAP={mean_signed_shap[index]:.6f}"
        )

    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=flattened_names,
        plot_type="bar",
        max_display=15,
        show=False,
    )
    plt.title("Monotonic XGBoost: top 15 SHAP feature importances")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
