"""Evaluate no-ICULOS models with bootstrap CIs."""

import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils import resample

DATA_PATH = Path('data/processed/features_v2.npz')

with np.load(DATA_PATH) as data:
    X_test = data['X_test']
    y_test = data['y_test']
    patient_ids_test = data['patient_ids_test']
    feature_names = data['feature_names'].tolist()

# Remove ablated features
ablated = ['Unit1', 'Unit2', 'HospAdmTime']
ablated_idx = [i for i, n in enumerate(feature_names) if any(a in n for a in ablated)]
X_test = np.delete(X_test, ablated_idx, axis=1)
feature_names = [n for i, n in enumerate(feature_names) if i not in ablated_idx]

# Remove ICULOS
iculos_idx = [i for i, n in enumerate(feature_names) if 'ICULOS' in n]
X_test_noiculos = np.delete(X_test, iculos_idx, axis=1)
feature_names_noiculos = [n for i, n in enumerate(feature_names) if i not in iculos_idx]

# Load models
xgb_noiculos = joblib.load('outputs/models/xgb_noiculos.pkl')
rf_noiculos = joblib.load('outputs/models/rf_noiculos.pkl')

# Load original models for comparison
xgb_orig = joblib.load('outputs/models/xgboost_v2.pkl')
rf_orig = joblib.load('outputs/models/random_forest_v2.pkl')

print("TASK 2: ICULOS Ablation Results")
print("=" * 80)

# Patient-level bootstrap
patient_to_indices = {}
for idx, pid in enumerate(patient_ids_test):
    if pid not in patient_to_indices:
        patient_to_indices[pid] = []
    patient_to_indices[pid].append(idx)

all_aurocs_noic_xgb = []
all_praucs_noic_xgb = []
all_aurocs_noic_rf = []
all_praucs_noic_rf = []

rng = np.random.RandomState(42)

for i in range(200):
    sampled_patients = resample(list(patient_to_indices.keys()), random_state=i)
    sampled_indices = []
    for pid in sampled_patients:
        sampled_indices.extend(patient_to_indices[pid])
    
    X_boot = X_test_noiculos[sampled_indices]
    y_boot = y_test[sampled_indices]
    
    # XGB no ICULOS
    scores = xgb_noiculos.predict_proba(X_boot)[:, 1]
    all_aurocs_noic_xgb.append(roc_auc_score(y_boot, scores))
    all_praucs_noic_xgb.append(average_precision_score(y_boot, scores))
    
    # RF no ICULOS
    scores = rf_noiculos.predict_proba(X_boot)[:, 1]
    all_aurocs_noic_rf.append(roc_auc_score(y_boot, scores))
    all_praucs_noic_rf.append(average_precision_score(y_boot, scores))

# Original CIs (from earlier)
print("\nOriginal (with ICULOS):")
print(f"  XGB: AUROC [0.8082, 0.9038], PR-AUC [0.0823, 0.3292]")
print(f"  RF:  AUROC [0.8448, 0.9260], PR-AUC [0.1186, 0.4025]")

print("\nNo ICULOS:")
xgb_auroc_ci = np.percentile(all_aurocs_noic_xgb, [2.5, 97.5])
xgb_prauc_ci = np.percentile(all_praucs_noic_xgb, [2.5, 97.5])
rf_auroc_ci = np.percentile(all_aurocs_noic_rf, [2.5, 97.5])
rf_prauc_ci = np.percentile(all_praucs_noic_rf, [2.5, 97.5])

print(f"  XGB: AUROC [{xgb_auroc_ci[0]:.4f}, {xgb_auroc_ci[1]:.4f}], PR-AUC [{xgb_prauc_ci[0]:.4f}, {xgb_prauc_ci[1]:.4f}]")
print(f"  RF:  AUROC [{rf_auroc_ci[0]:.4f}, {rf_auroc_ci[1]:.4f}], PR-AUC [{rf_prauc_ci[0]:.4f}, {rf_prauc_ci[1]:.4f}]")

# Calculate change vs CI width
xgb_orig_mean = (0.8082 + 0.9038) / 2
xgb_noic_mean = (xgb_auroc_ci[0] + xgb_auroc_ci[1]) / 2
xgb_ci_width = xgb_auroc_ci[1] - xgb_auroc_ci[0]
xgb_change = xgb_orig_mean - xgb_noic_mean

rf_orig_mean = (0.8448 + 0.9260) / 2
rf_noic_mean = (rf_auroc_ci[0] + rf_auroc_ci[1]) / 2
rf_ci_width = rf_auroc_ci[1] - rf_auroc_ci[0]
rf_change = rf_orig_mean - rf_noic_mean

print("\nChange analysis:")
print(f"  XGB: AUROC change = {xgb_change:.4f}, CI width = {xgb_ci_width:.4f}")
print(f"  RF:  AUROC change = {rf_change:.4f}, CI width = {rf_ci_width:.4f}")

print(f"\nXGB: Change {'IS' if abs(xgb_change) > xgb_ci_width else 'is NOT'} larger than CI width")
print(f"RF:  Change {'IS' if abs(rf_change) > rf_ci_width else 'is NOT'} larger than CI width")

# Top 10 features
print("\nTop 10 features by gain (no ICULOS):")
print("XGB:")
sorted_indices = np.argsort(xgb_noiculos.feature_importances_)[::-1][:10]
for idx in sorted_indices:
    print(f"  {feature_names_noiculos[idx]}: {xgb_noiculos.feature_importances_[idx]:.4f}")

print("RF:")
sorted_indices = np.argsort(rf_noiculos.feature_importances_)[::-1][:10]
for idx in sorted_indices:
    print(f"  {feature_names_noiculos[idx]}: {rf_noiculos.feature_importances_[idx]:.4f}")
