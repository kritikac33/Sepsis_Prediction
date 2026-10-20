"""Simulate frontend clicks via API for XGBoost and RF."""

import requests
import json

API_URL = "http://localhost:8000"

print("TASK 1: Frontend Verification (simulated via API)")
print("=" * 80)

for model in ['xgb', 'rf']:
    print(f"\n{model.upper()}:")
    
    # Test high risk (5 clicks)
    print("  High Risk (5 clicks):")
    for i in range(5):
        response = requests.get(f"{API_URL}/sample-patient?risk=high&model={model}")
        sample = response.json()
        window_id = sample.get('window_id')
        
        # Predict with window_id
        pred_response = requests.post(
            f"{API_URL}/predict?model={model}",
            json={"HR": 90, "Temp": 37, "Resp": 18, "MAP": 75, "O2Sat": 98, "Lactate": 2, "window_id": window_id}
        )
        pred = pred_response.json()
        print(f"    Click {i+1}: score={pred['risk_score']:.4f}, band={pred['risk_band']}, mode={pred.get('prediction_mode', 'N/A')}")
    
    # Test low risk (5 clicks)
    print("  Low Risk (5 clicks):")
    for i in range(5):
        response = requests.get(f"{API_URL}/sample-patient?risk=low&model={model}")
        sample = response.json()
        window_id = sample.get('window_id')
        
        # Predict with window_id
        pred_response = requests.post(
            f"{API_URL}/predict?model={model}",
            json={"HR": 90, "Temp": 37, "Resp": 18, "MAP": 75, "O2Sat": 98, "Lactate": 2, "window_id": window_id}
        )
        pred = pred_response.json()
        print(f"    Click {i+1}: score={pred['risk_score']:.4f}, band={pred['risk_band']}, mode={pred.get('prediction_mode', 'N/A')}")
