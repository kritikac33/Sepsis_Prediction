"""Sanity checks through API for XGBoost model."""

import json
import time
import requests

API_URL = "http://localhost:8000"
INPUT_FEATURES = ("HR", "Temp", "Resp", "MAP", "O2Sat", "Lactate")
BASELINE = {"HR": 75.0, "Temp": 37.0, "Resp": 14.0, "MAP": 85.0, "O2Sat": 98.0, "Lactate": 1.0}

start = time.time()
print(f"[{time.time()-start:.1f}s] Starting API sanity checks...")

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
    response = requests.post(f"{API_URL}/predict?model=xgboost", json=vitals)
    if response.status_code == 200:
        result = response.json()
        scores[name] = result["risk_score"]
        print(f"[{time.time()-start:.1f}s] {name}: {result['risk_score']:.6f}")
    else:
        print(f"[{time.time()-start:.1f}s] ERROR for {name}: {response.status_code}")
        scores[name] = None

# Check monotonicity
checks = {
    "Lactate 1->4->8": [scores["Baseline"], scores["Lactate 4"], scores["Lactate 8"]],
    "MAP 85->65->50": [scores["Baseline"], scores["MAP 65"], scores["MAP 50"]],
    "HR 75->130->160": [scores["Baseline"], scores["HR 130"], scores["HR 160"]],
    "O2Sat 98->90->82": [scores["Baseline"], scores["O2Sat 90"], scores["O2Sat 82"]],
}

print(f"\n[{time.time()-start:.1f}s] Sanity check pass/fail table:")
print("=" * 60)
for label, values in checks.items():
    if all(v is not None for v in values):
        is_monotonic = all(values[i] <= values[i+1] for i in range(len(values)-1))
        status = "PASS" if is_monotonic else "FAIL"
        print(f"{label:20s} {[f'{v:.6f}' for v in values]} {status}")
    else:
        print(f"{label:20s} ERROR - missing values")

# Check if "All abnormal" is highest
all_abnormal = scores["All abnormal"]
if all_abnormal is not None:
    other_max = max(v for k, v in scores.items() if k != "All abnormal" and v is not None)
    is_highest = all_abnormal > other_max
    print(f"\nAll abnormal highest: {is_highest} ({all_abnormal:.6f} vs {other_max:.6f})")
else:
    print(f"\nAll abnormal: ERROR")

print(f"[{time.time()-start:.1f}s] DONE")
