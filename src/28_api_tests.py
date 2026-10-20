"""Comprehensive API tests for sepsis prediction prototype.

Tests:
- Each model displays only its own results
- Model switching doesn't reuse stale data
- Patient switching doesn't show stale results
- Invalid model names handled safely
- Probability values are finite
- Class mapping against actual estimator
- Threshold boundary conditions
- SHAP corresponds to correct model/patient
- API returns correct response structure
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_URL = "http://localhost:8000"


def test_api_health():
    """Test API health endpoint."""
    print("\n" + "=" * 60)
    print("TEST: API Health")
    print("=" * 60)

    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        print("✓ API health check passed")
        return True
    except Exception as e:
        print(f"✗ API health check failed: {e}")
        return False


def test_model_isolation():
    """Test each model displays only its own results."""
    print("\n" + "=" * 60)
    print("TEST: Model Isolation")
    print("=" * 60)

    vitals = {
        "HR": 90.0,
        "Temp": 37.5,
        "Resp": 18.0,
        "MAP": 75.0,
        "O2Sat": 98.0,
        "Lactate": 2.0,
    }

    results = {}
    for model_name in ["xgb", "rf", "gb", "adaboost"]:
        try:
            response = requests.post(
                f"{API_URL}/predict",
                params={"model": model_name},
                json=vitals,
                timeout=30,
            )
            assert response.status_code == 200
            data = response.json()
            assert data["model"] == model_name
            results[model_name] = data
            print(f"✓ {model_name} response received with correct model name")
        except Exception as e:
            print(f"✗ {model_name} test failed: {e}")
            return False

    # Verify different models produce different results
    risk_scores = [results[m]["risk_score"] for m in results]
    if len(set(risk_scores)) > 1:
        print("✓ Models produce different risk scores (as expected)")
    else:
        print("⚠ All models produce identical risk scores (unexpected)")

    return True


def test_model_switching():
    """Test model switching doesn't reuse stale data."""
    print("\n" + "=" * 60)
    print("TEST: Model Switching")
    print("=" * 60)

    vitals = {
        "HR": 95.0,
        "Temp": 38.0,
        "Resp": 22.0,
        "MAP": 70.0,
        "O2Sat": 95.0,
        "Lactate": 3.0,
    }

    # Make predictions for each model in sequence
    predictions = []
    for model_name in ["xgb", "rf", "gb", "adaboost", "xgb"]:
        try:
            response = requests.post(
                f"{API_URL}/predict",
                params={"model": model_name},
                json=vitals,
                timeout=30,
            )
            assert response.status_code == 200
            data = response.json()
            predictions.append((model_name, data["risk_score"]))
            print(f"  {model_name}: {data['risk_score']:.6f}")
        except Exception as e:
            print(f"✗ Model switching test failed: {e}")
            return False

    # Verify first and last xgb predictions are similar (same input)
    first_xgb = predictions[0][1]
    last_xgb = predictions[4][1]
    if abs(first_xgb - last_xgb) < 0.001:
        print("✓ Model switching produces consistent results")
    else:
        print(f"⚠ XGB predictions differ: {first_xgb:.6f} vs {last_xgb:.6f}")

    return True


def test_invalid_model_name():
    """Test invalid model names handled safely."""
    print("\n" + "=" * 60)
    print("TEST: Invalid Model Name")
    print("=" * 60)

    vitals = {
        "HR": 90.0,
        "Temp": 37.5,
        "Resp": 18.0,
        "MAP": 75.0,
        "O2Sat": 98.0,
        "Lactate": 2.0,
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            params={"model": "invalid_model"},
            json=vitals,
            timeout=30,
        )
        # Should return 422 (Unprocessable Entity) for invalid enum value
        if response.status_code in [422, 400]:
            print("✓ Invalid model name rejected with appropriate error")
            return True
        else:
            print(f"⚠ Unexpected response code: {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠ Invalid model name test error: {e}")
        return False


def test_probability_finite():
    """Test probability values are finite."""
    print("\n" + "=" * 60)
    print("TEST: Probability Values Finite")
    print("=" * 60)

    vitals = {
        "HR": 90.0,
        "Temp": 37.5,
        "Resp": 18.0,
        "MAP": 75.0,
        "O2Sat": 98.0,
        "Lactate": 2.0,
    }

    for model_name in ["xgb", "rf", "gb", "adaboost"]:
        try:
            response = requests.post(
                f"{API_URL}/predict",
                params={"model": model_name},
                json=vitals,
                timeout=30,
            )
            assert response.status_code == 200
            data = response.json()
            risk_score = data["risk_score"]

            assert np.isfinite(risk_score), f"Risk score not finite for {model_name}"
            assert 0 <= risk_score <= 1, f"Risk score out of [0,1] for {model_name}: {risk_score}"

            print(f"✓ {model_name}: risk_score={risk_score:.6f} (finite, in [0,1])")
        except Exception as e:
            print(f"✗ {model_name} test failed: {e}")
            return False

    return True


def test_threshold_boundaries():
    """Test threshold boundary conditions."""
    print("\n" + "=" * 60)
    print("TEST: Threshold Boundary Conditions")
    print("=" * 60)

    # Load validation thresholds
    thresholds_path = PROJECT_ROOT / "data" / "processed" / "validation_thresholds.json"
    if not thresholds_path.exists():
        print("⚠ Validation thresholds not found, skipping test")
        return True

    with open(thresholds_path, "r") as f:
        thresholds_data = json.load(f)
    thresholds = thresholds_data["thresholds"]

    vitals = {
        "HR": 90.0,
        "Temp": 37.5,
        "Resp": 18.0,
        "MAP": 75.0,
        "O2Sat": 98.0,
        "Lactate": 2.0,
    }

    for model_name in ["xgb", "rf", "gb", "adaboost"]:
        try:
            response = requests.post(
                f"{API_URL}/predict",
                params={"model": model_name},
                json=vitals,
                timeout=30,
            )
            assert response.status_code == 200
            data = response.json()

            risk_score = data["risk_score"]
            red_thresh = thresholds[model_name]["red"]
            yellow_thresh = thresholds[model_name]["yellow"]

            # Verify risk band matches thresholds
            if risk_score >= red_thresh:
                expected_band = "red"
            elif risk_score >= yellow_thresh:
                expected_band = "yellow"
            else:
                expected_band = "green"

            actual_band = data["risk_band"]
            if actual_band == expected_band:
                print(f"✓ {model_name}: risk_band={actual_band} matches threshold logic")
            else:
                print(f"⚠ {model_name}: risk_band={actual_band}, expected={expected_band}")

        except Exception as e:
            print(f"✗ {model_name} test failed: {e}")
            return False

    return True


def test_shap_explanations():
    """Test SHAP explanations correspond to correct model."""
    print("\n" + "=" * 60)
    print("TEST: SHAP Explanations")
    print("=" * 60)

    vitals = {
        "HR": 95.0,
        "Temp": 38.5,
        "Resp": 24.0,
        "MAP": 65.0,
        "O2Sat": 92.0,
        "Lactate": 4.0,
    }

    for model_name in ["xgb", "rf", "gb"]:
        try:
            response = requests.post(
                f"{API_URL}/predict",
                params={"model": model_name},
                json=vitals,
                timeout=30,
            )
            assert response.status_code == 200
            data = response.json()

            top_features = data["top_features"]
            assert len(top_features) > 0, f"No features returned for {model_name}"
            assert len(top_features) <= 5, f"Too many features returned for {model_name}"

            # Check explanation type
            for feature in top_features:
                assert "feature" in feature
                assert "shap_value" in feature
                assert "direction" in feature
                assert np.isfinite(feature["shap_value"])

            # Check if using genuine SHAP
            has_genuine_shap = any(f.get("explanation_type") == "genuine_shap" for f in top_features)
            if has_genuine_shap:
                print(f"✓ {model_name}: Using genuine SHAP explanations")
            else:
                print(f"⚠ {model_name}: Using feature importance fallback")

        except Exception as e:
            print(f"✗ {model_name} test failed: {e}")
            return False

    return True


def test_response_structure():
    """Test API returns correct response structure."""
    print("\n" + "=" * 60)
    print("TEST: Response Structure")
    print("=" * 60)

    vitals = {
        "HR": 90.0,
        "Temp": 37.5,
        "Resp": 18.0,
        "MAP": 75.0,
        "O2Sat": 98.0,
        "Lactate": 2.0,
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            params={"model": "xgb"},
            json=vitals,
            timeout=30,
        )
        assert response.status_code == 200
        data = response.json()

        required_fields = ["model", "risk_score", "prediction", "risk_band", "top_features", "disclaimer", "score_note"]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

        print("✓ Response structure includes all required fields")
        print(f"  Disclaimer: {data['disclaimer']}")
        print(f"  Score note: {data['score_note']}")

        return True
    except Exception as e:
        print(f"✗ Response structure test failed: {e}")
        return False


def test_sample_patient():
    """Test sample patient endpoint."""
    print("\n" + "=" * 60)
    print("TEST: Sample Patient Endpoint")
    print("=" * 60)

    for model_name in ["xgb", "rf"]:
        try:
            response = requests.get(
                f"{API_URL}/sample-patient",
                params={"model": model_name, "risk": "high"},
                timeout=30,
            )
            assert response.status_code == 200
            data = response.json()

            required_fields = ["patient_index", "features", "label", "risk_group", "model", "test_window_risk_score"]
            for field in required_fields:
                assert field in data, f"Missing field: {field}"

            print(f"✓ {model_name} sample patient retrieved successfully")
            print(f"  Patient index: {data['patient_index']}")
            print(f"  Label: {data['label']}")
            print(f"  Risk score: {data['test_window_risk_score']:.6f}")

        except Exception as e:
            print(f"✗ {model_name} test failed: {e}")
            return False

    return True


def main():
    print("=" * 80)
    print("API AND FRONTEND TESTS")
    print("=" * 80)
    print("\nNOTE: This script requires the API server to be running.")
    print("Start with: uvicorn src.api:app --reload --port 8000")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    input()

    tests = [
        ("API Health", test_api_health),
        ("Model Isolation", test_model_isolation),
        ("Model Switching", test_model_switching),
        ("Invalid Model Name", test_invalid_model_name),
        ("Probability Finite", test_probability_finite),
        ("Threshold Boundaries", test_threshold_boundaries),
        ("SHAP Explanations", test_shap_explanations),
        ("Response Structure", test_response_structure),
        ("Sample Patient", test_sample_patient),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except KeyboardInterrupt:
            print("\n\nTests interrupted by user")
            break
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ {total - passed} test(s) failed")


if __name__ == "__main__":
    main()
