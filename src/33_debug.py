"""Test simple request to debug 422 error."""

import requests

API_URL = "http://localhost:8000"

# Test without window_id first
response = requests.post(
    f"{API_URL}/predict",
    params={"model": "xgb"},
    json={"HR": 90, "Temp": 37, "Resp": 18, "MAP": 75, "O2Sat": 98, "Lactate": 2},
    timeout=30
)
print(f"Without window_id: {response.status_code}")
print(response.text)

# Test with window_id
response = requests.post(
    f"{API_URL}/predict",
    params={"model": "xgb"},
    json={"HR": 90, "Temp": 37, "Resp": 18, "MAP": 75, "O2Sat": 98, "Lactate": 2, "window_id": 292},
    timeout=30
)
print(f"With window_id: {response.status_code}")
print(response.text)
