"""Test window_id prediction for each model."""

import joblib
import numpy as np
from pathlib import Path

DATA_PATH = Path('data/processed/features_v2.npz')

with np.load(DATA_PATH) as data:
    X_test = data['X_test']
    y_test = data['y_test']
    feature_names = data['feature_names'].tolist()

ablated = ['Unit1', 'Unit2', 'HospAdmTime']
ablated_idx = [i for i, n in enumerate(feature_names) if any(a in n for a in ablated)]
X_test = np.delete(X_test, ablated_idx, axis=1)
feature_names = [n for i, n in enumerate(feature_names) if i not in ablated_idx]

models = {
    'xgb': joblib.load('outputs/models/xgboost_v2.pkl'),
    'rf': joblib.load('outputs/models/random_forest_v2.pkl'),
    'gb': joblib.load('outputs/models/gradient_boosting_v2.pkl'),
    'adaboost': joblib.load('outputs/models/adaboost_v2.pkl')
}

# Compute thresholds
thresholds = {}
for name, model in models.items():
    scores = model.predict_proba(X_test)[:, 1]
    red_threshold = np.percentile(scores, 50)
    yellow_threshold = np.percentile(scores, 30)
    thresholds[name] = {'red': red_threshold, 'yellow': yellow_threshold}

# Get TP indices above red threshold
print("STEP 3: Verification")
print("=" * 80)

for name, model in models.items():
    scores = model.predict_proba(X_test)[:, 1]
    red_thresh = thresholds[name]['red']
    yellow_thresh = thresholds[name]['yellow']
    
    tp_indices = np.flatnonzero(y_test == 1)
    tp_high = tp_indices[scores[tp_indices] >= red_thresh]
    tn_indices = np.flatnonzero(y_test == 0)
    tn_low = tn_indices[scores[tn_indices] < yellow_thresh]
    
    print(f"\n{name}:")
    print(f"  Thresholds: red={red_thresh:.4f}, yellow={yellow_thresh:.4f}")
    print(f"  TP above red: {len(tp_high)}")
    print(f"  TN below yellow: {len(tn_low)}")
    
    if len(tp_high) > 0:
        print(f"  High sample scores:")
        for i in range(min(5, len(tp_high))):
            idx = tp_high[i]
            score = scores[idx]
            band = 'red' if score >= red_thresh else 'yellow' if score >= yellow_thresh else 'green'
            print(f"    window {idx}: {score:.4f} ({band})")
    
    if len(tn_low) > 0:
        print(f"  Low sample scores:")
        for i in range(min(5, len(tn_low))):
            idx = tn_low[i]
            score = scores[idx]
            band = 'red' if score >= red_thresh else 'yellow' if score >= yellow_thresh else 'green'
            print(f"    window {idx}: {score:.4f} ({band})")
