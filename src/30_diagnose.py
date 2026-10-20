"""Diagnose sample-patient vs predict mismatch."""

import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path('.')
DATA_PATH = PROJECT_ROOT / 'data/processed/features_v2.npz'

# Load data
with np.load(DATA_PATH) as data:
    X_test = data['X_test']
    y_test = data['y_test']
    feature_names = data['feature_names'].tolist()
    X_train = data['X_train']

# Filter ablated features
ablated = ['Unit1', 'Unit2', 'HospAdmTime']
ablated_idx = [i for i, n in enumerate(feature_names) if any(a in n for a in ablated)]
X_test = np.delete(X_test, ablated_idx, axis=1)
X_train = np.delete(X_train, ablated_idx, axis=1)
feature_names = [n for i, n in enumerate(feature_names) if i not in ablated_idx]

# Get stats for normalization
means = X_train.mean(axis=0)
stds = X_train.std(axis=0)

INPUT_FEATURES = ('HR', 'Temp', 'Resp', 'MAP', 'O2Sat', 'Lactate')

def create_snapshot(vitals_dict):
    """Create 6-vital snapshot feature vector."""
    features = np.full(len(feature_names), np.nan)
    
    # Fill medians first
    for i in range(len(feature_names)):
        features[i] = means[i]
    
    # Set ICULOS to low value
    for i, name in enumerate(feature_names):
        if 'ICULOS' in name:
            features[i] = -1.0
    
    # Set demographics to neutral
    for i, name in enumerate(feature_names):
        if name in ['Age_last', 'Age_mean', 'Age_min', 'Age_max', 'Age_slope']:
            features[i] = 0.0
        if name in ['Gender_last', 'Gender_mean', 'Gender_min', 'Gender_max', 'Gender_slope']:
            features[i] = 0.0
    
    # Set vitals from input
    for vital_name, value in vitals_dict.items():
        normalized_value = (value - means[feature_names.index(f'{vital_name}_last')]) / stds[feature_names.index(f'{vital_name}_last')]
        
        for suffix in ['_last', '_mean', '_min', '_max']:
            feature_name = f'{vital_name}{suffix}'
            if feature_name in feature_names:
                idx = feature_names.index(feature_name)
                features[idx] = normalized_value
        
        # Set slope to 0
        slope_name = f'{vital_name}_slope'
        if slope_name in feature_names:
            idx = feature_names.index(slope_name)
            features[idx] = 0.0
    
    return features

# Load models
models = {
    'xgb': joblib.load('outputs/models/xgboost_v2.pkl'),
    'rf': joblib.load('outputs/models/random_forest_v2.pkl'),
    'gb': joblib.load('outputs/models/gradient_boosting_v2.pkl'),
    'adaboost': joblib.load('outputs/models/adaboost_v2.pkl')
}

# Get 5 true-positive test windows
tp_indices = np.flatnonzero(y_test == 1)[:5]

print("STEP 1 DIAGNOSIS")
print("=" * 80)
print("Comparing full 226-feature vector vs 6-vital snapshot for 5 true-positive windows")
print("=" * 80)

for idx in tp_indices:
    full_vector = X_test[idx]
    
    # Extract vitals from full vector
    vitals_dict = {}
    for vital in INPUT_FEATURES:
        last_idx = feature_names.index(f'{vital}_last')
        raw_value = full_vector[last_idx] * stds[last_idx] + means[last_idx]
        vitals_dict[vital] = raw_value
    
    snapshot = create_snapshot(vitals_dict)
    
    print(f"\nWindow {idx} (label={y_test[idx]}):")
    print(f"  Vitals: {vitals_dict}")
    
    for name, model in models.items():
        full_score = model.predict_proba(full_vector.reshape(1, -1))[0, 1]
        snap_score = model.predict_proba(snapshot.reshape(1, -1))[0, 1]
        diff = snap_score - full_score
        print(f"  {name}: full={full_score:.4f}, snapshot={snap_score:.4f}, diff={diff:.4f}")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
