"""Retrain XGBoost and RandomForest without ICULOS features."""

import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

PROJECT_ROOT = Path('.')
DATA_PATH = PROJECT_ROOT / 'data/processed/features_v2.npz'
OUTPUT_DIR = PROJECT_ROOT / 'outputs/models'

RANDOM_STATE = 42
N_FOLDS = 2

with np.load(DATA_PATH) as data:
    X_train = data['X_train']
    y_train = data['y_train']
    patient_ids_train = data['patient_ids_train']
    X_test = data['X_test']
    y_test = data['y_test']
    patient_ids_test = data['patient_ids_test']
    feature_names = data['feature_names'].tolist()

# Remove Unit1, Unit2, HospAdmTime (already ablated)
ablated_features = ["Unit1", "Unit2", "HospAdmTime"]
ablated_indices = [i for i, name in enumerate(feature_names) if any(af in name for af in ablated_features)]
X_train = np.delete(X_train, ablated_indices, axis=1)
X_test = np.delete(X_test, ablated_indices, axis=1)
feature_names = [name for i, name in enumerate(feature_names) if i not in ablated_indices]

# Remove ICULOS features
iculos_indices = [i for i, name in enumerate(feature_names) if 'ICULOS' in name]
print(f"Removing {len(iculos_indices)} ICULOS features")
X_train_noiculos = np.delete(X_train, iculos_indices, axis=1)
X_test_noiculos = np.delete(X_test, iculos_indices, axis=1)
feature_names_noiculos = [name for i, name in enumerate(feature_names) if i not in iculos_indices]

print(f"Features with ICULOS: {len(feature_names)}")
print(f"Features without ICULOS: {len(feature_names_noiculos)}")

# Train XGBoost without ICULOS
print("\nTraining XGBoost without ICULOS...")
xgb_model = xgb.XGBClassifier(
    n_estimators=30,
    max_depth=3,
    learning_rate=0.1,
    objective='binary:logistic',
    eval_metric='auc',
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

gkf = GroupKFold(n_splits=N_FOLDS)
roc_aucs = []
pr_aucs = []

for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train_noiculos, y_train, groups=patient_ids_train)):
    X_fold_train, X_fold_val = X_train_noiculos[train_idx], X_train_noiculos[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
    
    xgb_model.fit(X_fold_train, y_fold_train)
    y_pred = xgb_model.predict_proba(X_fold_val)[:, 1]
    
    roc_aucs.append(roc_auc_score(y_fold_val, y_pred))
    pr_aucs.append(average_precision_score(y_fold_val, y_pred))

print(f"XGBoost CV AUROC: {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")
print(f"XGBoost CV PR-AUC: {np.mean(pr_aucs):.4f} ± {np.std(pr_aucs):.4f}")

# Retrain on full training data
xgb_model.fit(X_train_noiculos, y_train)
joblib.dump(xgb_model, OUTPUT_DIR / 'xgb_noiculos.pkl')
print("Saved xgb_noiculos.pkl")

# Train Random Forest without ICULOS
print("\nTraining Random Forest without ICULOS...")
rf_model = RandomForestClassifier(
    n_estimators=30,
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

roc_aucs_rf = []
pr_aucs_rf = []

for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train_noiculos, y_train, groups=patient_ids_train)):
    X_fold_train, X_fold_val = X_train_noiculos[train_idx], X_train_noiculos[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
    
    rf_model.fit(X_fold_train, y_fold_train)
    y_pred = rf_model.predict_proba(X_fold_val)[:, 1]
    
    roc_aucs_rf.append(roc_auc_score(y_fold_val, y_pred))
    pr_aucs_rf.append(average_precision_score(y_fold_val, y_pred))

print(f"RF CV AUROC: {np.mean(roc_aucs_rf):.4f} ± {np.std(roc_aucs_rf):.4f}")
print(f"RF CV PR-AUC: {np.mean(pr_aucs_rf):.4f} ± {np.std(pr_aucs_rf):.4f}")

# Retrain on full training data
rf_model.fit(X_train_noiculos, y_train)
joblib.dump(rf_model, OUTPUT_DIR / 'rf_noiculos.pkl')
print("Saved rf_noiculos.pkl")
