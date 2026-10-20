"""Train a Random Forest baseline model compatible with the API.

This follows the same feature contract and train/test split as src/02_ml_baseline.py
to ensure compatibility with the API and other models.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "data.npz"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "rf_baseline.pkl"
METRICS_PATH = PROJECT_ROOT / "outputs" / "models" / "rf_baseline_metrics.json"
RANDOM_STATE = 42
TEST_SIZE = 0.20
EXCLUDED_FEATURES = {"ICULOS"}
INPUT_FEATURES = ("HR", "Temp", "Resp", "MAP", "O2Sat", "Lactate")


def make_random_forest() -> RandomForestClassifier:
    """Create a Random Forest classifier with parameters similar to XGBoost."""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
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

    # Use the same feature selection as XGBoost (exclude ICULOS)
    model_feature_indices = [
        index for index, name in enumerate(feature_names) if name not in EXCLUDED_FEATURES
    ]
    model_feature_names = [feature_names[index] for index in model_feature_indices]
    if not model_feature_indices:
        raise ValueError("No model features remain after exclusion.")

    # Exclude ICULOS before flattening
    X_model = X[:, :, model_feature_indices]
    X_flat = X_model.reshape(X_model.shape[0], -1)

    print(f"Loaded sequences: X={X.shape}, y={y.shape}")
    print(f"Excluded features: {', '.join(sorted(EXCLUDED_FEATURES))}")
    print(f"Model features: {len(model_feature_names)} of {len(feature_names)}")
    print(f"Classical ML input after flattening: {X_flat.shape}")

    # Use the same train/test split as XGBoost
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Train/test split: {len(X_train):,}/{len(X_test):,}")

    model = make_random_forest()
    print("Training Random Forest baseline...")
    model.fit(X_train, y_train)

    # Store the feature contract for API compatibility
    model.selected_feature_names_ = np.asarray(model_feature_names)
    model.sequence_length_ = X.shape[1]

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy": float((predictions == y_test).mean()),
        "ROC-AUC": float(roc_auc_score(y_test, probabilities)),
        "PR-AUC (average precision)": float(average_precision_score(y_test, probabilities)),
        "F1 (positive)": float(f1_score(y_test, predictions, pos_label=1, zero_division=0)),
    }

    print("\nTest metrics")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # Cross-validation
    print("\nRunning 3-fold stratified cross-validation...")
    cross_validator = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    cv_model = type(model)(**{k: v for k, v in model.get_params().items() if k != "selected_feature_names_" and k != "sequence_length_"})
    cv_scores = cross_validate(
        cv_model,
        X_flat,
        y,
        cv=cross_validator,
        scoring={"auc": "roc_auc", "f1": "f1"},
        n_jobs=1,
    )
    print(
        "3-fold CV ROC-AUC: "
        f"{cv_scores['test_auc'].mean():.4f} ± {cv_scores['test_auc'].std():.4f}"
    )
    print(
        "3-fold CV F1: "
        f"{cv_scores['test_f1'].mean():.4f} ± {cv_scores['test_f1'].std():.4f}"
    )

    # Save metrics
    evaluation_record = {
        "dataset": {"samples": int(len(y)), "positive_samples": int(y.sum())},
        "test": metrics,
        "cross_validation": {
            "folds": 3,
            "roc_auc_mean": float(cv_scores["test_auc"].mean()),
            "roc_auc_std": float(cv_scores["test_auc"].std()),
            "f1_mean": float(cv_scores["test_f1"].mean()),
            "f1_std": float(cv_scores["test_f1"].std()),
        },
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(evaluation_record, indent=2), encoding="utf-8")
    print(f"Saved evaluation metrics to {METRICS_PATH}")

    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
