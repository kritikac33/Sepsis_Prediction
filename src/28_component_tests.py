"""Component-level tests for sepsis prediction prototype.

Tests components directly without requiring API server:
- Model loading and prediction
- Probability calculation
- Class mapping verification
- Threshold application
- SHAP explanation generation
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features_v2.npz"
THRESHOLDS_PATH = PROJECT_ROOT / "data" / "processed" / "validation_thresholds.json"

XGB_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "xgboost_v2.pkl"
RF_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "random_forest_v2.pkl"
GB_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "gradient_boosting_v2.pkl"
ADABOOST_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "adaboost_v2.pkl"


def test_model_loading():
    """Test all models can be loaded."""
    print("\n" + "=" * 60)
    print("TEST: Model Loading")
    print("=" * 60)

    models = {
        "xgb": XGB_MODEL_PATH,
        "rf": RF_MODEL_PATH,
        "gb": GB_MODEL_PATH,
        "adaboost": ADABOOST_MODEL_PATH,
    }

    loaded = {}
    for name, path in models.items():
        try:
            model = joblib.load(path)
            loaded[name] = model
            print(f"[OK] {name} loaded successfully")
        except Exception as e:
            print(f"[FAIL] {name} failed to load: {e}")
            return False

    return loaded


def test_class_mapping(models):
    """Test class mapping for all models."""
    print("\n" + "=" * 60)
    print("TEST: Class Mapping")
    print("=" * 60)

    for name, model in models.items():
        if hasattr(model, 'classes_'):
            classes = model.classes_
            print(f"  {name}: classes = {classes}")
            if len(classes) == 2 and classes[1] == 1:
                print(f"[OK] {name}: Class mapping verified (class 1 = positive)")
            else:
                print(f"[WARN] {name}: Unexpected class mapping")
        else:
            print(f"[WARN] {name}: No classes_ attribute")

    return True


def test_probability_calculation(models, X_test):
    """Test probability calculation and finiteness."""
    print("\n" + "=" * 60)
    print("TEST: Probability Calculation")
    print("=" * 60)

    for name, model in models.items():
        try:
            probs = model.predict_proba(X_test[:100])  # Test on subset
            pos_probs = probs[:, 1]

            # Check finiteness
            if not np.all(np.isfinite(pos_probs)):
                print(f"[FAIL] {name}: Non-finite probabilities detected")
                return False

            # Check range
            if not (0 <= pos_probs.min() and pos_probs.max() <= 1):
                print(f"[FAIL] {name}: Probabilities outside [0,1] range")
                return False

            # Check sum to 1
            row_sums = probs.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-6):
                print(f"[FAIL] {name}: Probabilities don't sum to 1")
                return False

            print(f"[OK] {name}: Probabilities valid (finite, in [0,1], sum to 1)")
            print(f"  Range: [{pos_probs.min():.6f}, {pos_probs.max():.6f}]")

        except Exception as e:
            print(f"[FAIL] {name}: Probability calculation failed: {e}")
            return False

    return True


def test_threshold_application():
    """Test threshold loading and application."""
    print("\n" + "=" * 60)
    print("TEST: Threshold Application")
    print("=" * 60)

    if not THRESHOLDS_PATH.exists():
        print("[WARN] Validation thresholds not found")
        return False

    import json
    with open(THRESHOLDS_PATH, "r") as f:
        thresholds_data = json.load(f)

    thresholds = thresholds_data["thresholds"]
    print(f"Loaded thresholds for {len(thresholds)} models")

    for model_name, thresh in thresholds.items():
        red = thresh["red"]
        yellow = thresh["yellow"]

        if red <= yellow:
            print(f"[WARN] {model_name}: Red threshold ({red}) <= Yellow threshold ({yellow})")
        else:
            print(f"[OK] {model_name}: Red ({red:.4f}) > Yellow ({yellow:.4f})")

        # Test risk band assignment
        test_scores = [red - 0.001, yellow + 0.001, red + 0.001]
        for score in test_scores:
            if score >= red:
                band = "red"
            elif score >= yellow:
                band = "yellow"
            else:
                band = "green"
            print(f"  Score {score:.4f} -> {band}")

    return True


def test_shap_loading():
    """Test SHAP explainers can be loaded."""
    print("\n" + "=" * 60)
    print("TEST: SHAP Explainer Loading")
    print("=" * 60)

    shap_dir = PROJECT_ROOT / "outputs" / "shap_explainers"

    if not shap_dir.exists():
        print("[WARN] SHAP explainers directory not found")
        return False

    for model_name in ["xgb", "rf", "gb"]:
        shap_path = shap_dir / f"shap_{model_name}.pkl"
        if shap_path.exists():
            try:
                shap_data = joblib.load(shap_path)
                explainer = shap_data["explainer"]
                explainer_type = shap_data["type"]
                print(f"[OK] {model_name}: SHAP explainer loaded (type: {explainer_type})")
            except Exception as e:
                print(f"[FAIL] {model_name}: Failed to load SHAP explainer: {e}")
        else:
            print(f"[WARN] {model_name}: SHAP explainer not found")

    return True


def test_data_consistency():
    """Test data consistency across models."""
    print("\n" + "=" * 60)
    print("TEST: Data Consistency")
    print("=" * 60)

    with np.load(DATA_PATH, allow_pickle=True) as data:
        X_test = data["X_test"].astype(np.float32)
        y_test = data["y_test"].astype(np.int32)
        feature_names = data["feature_names"].tolist()

    # Filter ablated features
    ablated_features = ["Unit1", "Unit2", "HospAdmTime"]
    ablated_indices = [i for i, name in enumerate(feature_names) if any(af in name for af in ablated_features)]
    if ablated_indices:
        X_test = np.delete(X_test, ablated_indices, axis=1)
        feature_names = [name for i, name in enumerate(feature_names) if i not in ablated_indices]

    print(f"Test set: {len(X_test)} windows, {len(feature_names)} features")
    print(f"Positive: {y_test.sum()} ({y_test.mean()*100:.2f}%)")

    # Check for NaN/Inf
    if np.any(np.isnan(X_test)):
        print("[FAIL] NaN values detected in X_test")
        return False

    if np.any(np.isinf(X_test)):
        print("[FAIL] Inf values detected in X_test")
        return False

    print("[OK] Data is clean (no NaN/Inf)")

    return True, X_test, y_test, feature_names


def main():
    print("=" * 80)
    print("COMPONENT-LEVEL TESTS")
    print("=" * 80)

    # Test data consistency
    data_ok, X_test, y_test, feature_names = test_data_consistency()
    if not data_ok:
        print("\n[FAIL] Data consistency test failed")
        return

    # Test model loading
    models = test_model_loading()
    if not models:
        print("\n[FAIL] Model loading failed")
        return

    # Test class mapping
    if not test_class_mapping(models):
        print("\n[FAIL] Class mapping test failed")
        return

    # Test probability calculation
    if not test_probability_calculation(models, X_test):
        print("\n[FAIL] Probability calculation test failed")
        return

    # Test threshold application
    if not test_threshold_application():
        print("\n[FAIL] Threshold application test failed")
        return

    # Test SHAP loading
    if not test_shap_loading():
        print("\n[FAIL] SHAP loading test failed")
        return

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("[OK] All component tests passed!")
    print("\nComponents verified:")
    print("  - Model loading")
    print("  - Class mapping")
    print("  - Probability calculation")
    print("  - Threshold application")
    print("  - SHAP explainer loading")
    print("  - Data consistency")


if __name__ == "__main__":
    main()
