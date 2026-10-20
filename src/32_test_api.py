"""Test window_id prediction for each model via API."""

import requests
import json

API_URL = "http://localhost:8000"

print("STEP 3: Real API Tests")
print("=" * 80)

for model in ['xgb', 'rf', 'gb', 'adaboost']:
    print(f"\n{model.upper()}:")
    
    # Test high risk (should use window_id internally via frontend)
    # For now, test directly with window_id
    high_samples = [292, 293, 294, 295, 296]  # Known TP above red threshold
    
    for window_id in high_samples:
        try:
            response = requests.post(
                f"{API_URL}/predict",
                params={"model": model},
                json={"HR": 90, "Temp": 37, "Resp": 18, "MAP": 75, "O2Sat": 98, "Lactate": 2, "window_id": window_id},
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                print(f"  window {window_id}: score={data['risk_score']:.4f}, band={data['risk_band']}, mode={data.get('prediction_mode', 'N/A')}")
            else:
                print(f"  window {window_id}: ERROR {response.status_code}")
        except Exception as e:
            print(f"  window {window_id}: EXCEPTION {e}")
    
    # Test low risk
    low_samples = [17, 18, 19, 20, 21]
    
    for window_id in low_samples:
        try:
            response = requests.post(
                f"{API_URL}/predict",
                params={"model": model},
                json={"HR": 90, "Temp": 37, "Resp": 18, "MAP": 75, "O2Sat": 98, "Lactate": 2, "window_id": window_id},
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                print(f"  window {window_id}: score={data['risk_score']:.4f}, band={data['risk_band']}, mode={data.get('prediction_mode', 'N/A')}")
            else:
                print(f"  window {window_id}: ERROR {response.status_code}")
        except Exception as e:
            print(f"  window {window_id}: EXCEPTION {e}")
