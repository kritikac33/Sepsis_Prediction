"""Debug scaler and feature mapping."""

import pickle
import numpy as np

# Load data
d = np.load(r'data\processed\features_v2.npz', allow_pickle=True)
X_train = d['X_train']
feature_names = d['feature_names'].tolist()

# Remove ablated features
ablated = [i for i, n in enumerate(feature_names) if any(af in n for af in ['Unit1', 'Unit2', 'HospAdmTime'])]
X_train = np.delete(X_train, ablated, axis=1)
feature_names = [name for i, name in enumerate(feature_names) if i not in ablated]

print('X_train shape:', X_train.shape)
print('Feature medians sample:', np.median(X_train, axis=0)[:5])

# Load model artifact
with open(r'outputs\models\xgb_final_v2.pkl', 'rb') as f:
    art = pickle.load(f)

print('Scaler n_features_in_:', art['scaler'].n_features_in_)
print('Scaler mean sample:', art['scaler'].mean_[:5])
print('Scaler var sample:', art['scaler'].var_[:5])

# Test mapping
from sklearn.preprocessing import StandardScaler
scaler = art['scaler']
medians = np.median(X_train, axis=0)

# Test with a simple value
test_vitals = {"HR": 75, "Temp": 37.0, "Resp": 16, "MAP": 85, "O2Sat": 98, "Lactate": 1.0}
features = medians.copy()

for vital_name, value in test_vitals.items():
    for suffix in ["_last", "_mean", "_min", "_max"]:
        feature_name = f"{vital_name}{suffix}"
        if feature_name in feature_names:
            idx = feature_names.index(feature_name)
            features[idx] = value
            print(f"Set {feature_name} (idx {idx}) to {value}")

features_scaled = scaler.transform(features.reshape(1, -1))
print('Scaled features sample:', features_scaled[0, :5])
print('Raw features sample:', features[:5])
