"""Permutation test and bootstrap CI for final XGBoost model."""

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_v2.npz"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "xgboost_v2.pkl"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "final_evaluation.json"

N_PERMUTATIONS = 50
N_BOOTSTRAP = 200
RANDOM_STATE = 42

start = time.time()
print(f"[{time.time()-start:.1f}s] Loading data and model...")

with np.load(FEATURES_PATH, allow_pickle=True) as data:
    X_test = data["X_test"]
    y_test = data["y_test"]
    feature_names = data["feature_names"].tolist()

# Remove ablated features to match model
ABLATED_FEATURES = ["Unit1", "Unit2", "HospAdmTime"]
ablated_indices = [i for i, name in enumerate(feature_names) if any(af in name for af in ABLATED_FEATURES)]
X_test = np.delete(X_test, ablated_indices, axis=1)

model = joblib.load(MODEL_PATH)
y_pred = model.predict_proba(X_test)[:, 1]

# Baseline metrics
roc_auc = roc_auc_score(y_test, y_pred)
pr_auc = average_precision_score(y_test, y_pred)
prevalence = y_test.mean()

print(f"[{time.time()-start:.1f}s] Test ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
print(f"[{time.time()-start:.1f}s] Test prevalence: {prevalence:.4f} (PR-AUC baseline)")

# Permutation test
print(f"[{time.time()-start:.1f}s] Running permutation test ({N_PERMUTATIONS} iterations)...")
rng = np.random.default_rng(RANDOM_STATE)
perm_aucs = []
for i in range(N_PERMUTATIONS):
    y_perm = rng.permutation(y_test)
    perm_auc = roc_auc_score(y_perm, y_pred)
    perm_aucs.append(perm_auc)
    if (i + 1) % 10 == 0:
        print(f"[{time.time()-start:.1f}s] {i+1}/{N_PERMUTATIONS} permutations...")

perm_mean = np.mean(perm_aucs)
perm_std = np.std(perm_aucs)
print(f"[{time.time()-start:.1f}s] Permutation AUC: {perm_mean:.4f} ± {perm_std:.4f}")

if perm_mean < 0.55:
    print(f"[{time.time()-start:.1f}s] Permutation mean near 0.5 as expected (model has signal)")
else:
    print(f"[{time.time()-start:.1f}s] WARNING: Permutation mean > 0.5 - possible data leakage or imbalance")

# Bootstrap CI
print(f"[{time.time()-start:.1f}s] Running bootstrap CI ({N_BOOTSTRAP} iterations)...")
boot_roc_aucs = []
boot_pr_aucs = []
for i in range(N_BOOTSTRAP):
    idx = rng.choice(len(y_test), size=len(y_test), replace=True)
    boot_roc_auc = roc_auc_score(y_test[idx], y_pred[idx])
    boot_pr_auc = average_precision_score(y_test[idx], y_pred[idx])
    boot_roc_aucs.append(boot_roc_auc)
    boot_pr_aucs.append(boot_pr_auc)
    if (i + 1) % 50 == 0:
        print(f"[{time.time()-start:.1f}s] {i+1}/{N_BOOTSTRAP} bootstraps...")

roc_ci_low, roc_ci_high = np.percentile(boot_roc_aucs, [2.5, 97.5])
pr_ci_low, pr_ci_high = np.percentile(boot_pr_aucs, [2.5, 97.5])

print(f"[{time.time()-start:.1f}s] ROC-AUC 95% CI: [{roc_ci_low:.4f}, {roc_ci_high:.4f}]")
print(f"[{time.time()-start:.1f}s] PR-AUC 95% CI: [{pr_ci_low:.4f}, {pr_ci_high:.4f}]")

results = {
    "test_roc_auc": float(roc_auc),
    "test_pr_auc": float(pr_auc),
    "test_prevalence": float(prevalence),
    "permutation_mean": float(perm_mean),
    "permutation_std": float(perm_std),
    "roc_auc_ci_95": [float(roc_ci_low), float(roc_ci_high)],
    "pr_auc_ci_95": [float(pr_ci_low), float(pr_ci_high)],
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"[{time.time()-start:.1f}s] Saved to {OUTPUT_PATH}")
print(f"[{time.time()-start:.1f}s] DONE")
