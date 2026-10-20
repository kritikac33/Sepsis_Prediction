"""Set demo thresholds and test sample-patient endpoint."""

import requests
import time

BASE_URL = "http://127.0.0.1:8000"


def main():
    print("=== Step 3: Demo Thresholds ===\n")

    start_time = time.time()

    # Wait for API
    for _ in range(10):
        try:
            requests.get(f"{BASE_URL}/health")
            break
        except:
            time.sleep(0.5)

    # Test sample-patient endpoint
    print("Testing /sample-patient?risk=high (XGBoost):")
    high_risks = []
    for i in range(5):
        resp = requests.get(f"{BASE_URL}/sample-patient", params={"risk": "high", "model": "xgb"})
        resp.raise_for_status()
        data = resp.json()
        high_risks.append(data["snapshot_risk_score"])
        print(f"  {i+1}: risk={data['snapshot_risk_score']:.4f}, label={data['label']}, band={data.get('band', 'N/A')}")

    print("\nTesting /sample-patient?risk=low (XGBoost):")
    low_risks = []
    for i in range(5):
        resp = requests.get(f"{BASE_URL}/sample-patient", params={"risk": "low", "model": "xgb"})
        resp.raise_for_status()
        data = resp.json()
        low_risks.append(data["snapshot_risk_score"])
        print(f"  {i+1}: risk={data['snapshot_risk_score']:.4f}, label={data['label']}, band={data.get('band', 'N/A')}")

    # Check colors
    print("\n=== Color Distribution ===")
    red_count = sum(1 for r in high_risks if r >= 0.05)  # Approximate red threshold
    green_count = sum(1 for r in low_risks if r < 0.03)  # Approximate green threshold
    print(f"High risk samples: {red_count}/5 red (expected mostly red)")
    print(f"Low risk samples: {green_count}/5 green (expected mostly green)")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")

    # Update progress log
    with open("outputs/progress_log.md", "a") as f:
        f.write(f"| Step 3: Demo thresholds | High risks={high_risks}, Low risks={low_risks} | {elapsed:.1f} |\n")


if __name__ == "__main__":
    main()
