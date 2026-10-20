"""Run sanity checks on the API to verify monotonic behavior."""

import requests
import time

BASE_URL = "http://127.0.0.1:8000"

# Normal baseline values
NORMAL_VALUES = {
    "HR": 75,
    "Temp": 37.0,
    "Resp": 16,
    "MAP": 85,
    "O2Sat": 98,
    "Lactate": 1.0,
}


def predict(vitals: dict, model: str = "xgb") -> dict:
    """Call the /predict endpoint."""
    response = requests.post(f"{BASE_URL}/predict", json=vitals, params={"model": model})
    response.raise_for_status()
    return response.json()


def main():
    print("=== API Sanity Checks ===\n")

    start_time = time.time()

    # Wait for API to be ready
    for _ in range(10):
        try:
            requests.get(f"{BASE_URL}/health")
            break
        except:
            time.sleep(0.5)
    else:
        print("API not ready")
        return

    results = []

    # Lactate sweep (higher = higher risk)
    print("Lactate sweep (1, 4, 8):")
    for lactate in [1, 4, 8]:
        vitals = NORMAL_VALUES.copy()
        vitals["Lactate"] = lactate
        resp = predict(vitals, "xgb")
        risk = resp["risk_score"]
        results.append(("Lactate", lactate, risk))
        print(f"  Lactate={lactate}: risk={risk:.4f}")

    # MAP sweep (lower = higher risk)
    print("\nMAP sweep (85, 65, 50):")
    for map_val in [85, 65, 50]:
        vitals = NORMAL_VALUES.copy()
        vitals["MAP"] = map_val
        resp = predict(vitals, "xgb")
        risk = resp["risk_score"]
        results.append(("MAP", map_val, risk))
        print(f"  MAP={map_val}: risk={risk:.4f}")

    # HR sweep (higher = higher risk)
    print("\nHR sweep (75, 130, 160):")
    for hr in [75, 130, 160]:
        vitals = NORMAL_VALUES.copy()
        vitals["HR"] = hr
        resp = predict(vitals, "xgb")
        risk = resp["risk_score"]
        results.append(("HR", hr, risk))
        print(f"  HR={hr}: risk={risk:.4f}")

    # O2Sat sweep (lower = higher risk)
    print("\nO2Sat sweep (98, 90, 82):")
    for o2sat in [98, 90, 82]:
        vitals = NORMAL_VALUES.copy()
        vitals["O2Sat"] = o2sat
        resp = predict(vitals, "xgb")
        risk = resp["risk_score"]
        results.append(("O2Sat", o2sat, risk))
        print(f"  O2Sat={o2sat}: risk={risk:.4f}")

    # All abnormal at once
    print("\nAll abnormal (HR 150, Temp 39.5, Resp 32, MAP 55, O2Sat 88, Lactate 6.0):")
    abnormal_vitals = {
        "HR": 150,
        "Temp": 39.5,
        "Resp": 32,
        "MAP": 55,
        "O2Sat": 88,
        "Lactate": 6.0,
    }
    resp = predict(abnormal_vitals, "xgb")
    risk = resp["risk_score"]
    results.append(("All abnormal", "combined", risk))
    print(f"  Risk: {risk:.4f}")

    # Check monotonicity
    print("\n=== Monotonicity Checks ===")
    failed = []

    # Lactate: should increase
    lactate_risks = [r for name, val, r in results if name == "Lactate"]
    if not (lactate_risks[0] < lactate_risks[1] < lactate_risks[2]):
        failed.append("Lactate not monotonic increasing")
        print(f"FAIL: Lactate {lactate_risks}")
    else:
        print(f"PASS: Lactate {lactate_risks}")

    # MAP: should decrease
    map_risks = [r for name, val, r in results if name == "MAP"]
    if not (map_risks[0] > map_risks[1] > map_risks[2]):
        failed.append("MAP not monotonic decreasing")
        print(f"FAIL: MAP {map_risks}")
    else:
        print(f"PASS: MAP {map_risks}")

    # HR: should increase
    hr_risks = [r for name, val, r in results if name == "HR"]
    if not (hr_risks[0] < hr_risks[1] < hr_risks[2]):
        failed.append("HR not monotonic increasing")
        print(f"FAIL: HR {hr_risks}")
    else:
        print(f"PASS: HR {hr_risks}")

    # O2Sat: should decrease
    o2sat_risks = [r for name, val, r in results if name == "O2Sat"]
    if not (o2sat_risks[0] > o2sat_risks[1] > o2sat_risks[2]):
        failed.append("O2Sat not monotonic decreasing")
        print(f"FAIL: O2Sat {o2sat_risks}")
    else:
        print(f"PASS: O2Sat {o2sat_risks}")

    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
    else:
        print("\nAll monotonicity checks PASSED")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")

    # Update progress log
    with open("outputs/progress_log.md", "a") as f:
        f.write(f"| Step 2: Sanity checks | {'PASS' if not failed else 'FAIL: ' + ', '.join(failed)} | {elapsed:.1f} |\n")


if __name__ == "__main__":
    main()
