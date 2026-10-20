"""Check ICULOS and other features in saved models."""

import joblib
from pathlib import Path

models = {
    'xgb': 'xgboost_v2.pkl',
    'rf': 'random_forest_v2.pkl',
    'gb': 'gradient_boosting_v2.pkl',
    'adaboost': 'adaboost_v2.pkl'
}

print("STEP 4: ICULOS and Feature Report")
print("=" * 80)

for name, model_file in models.items():
    model = joblib.load(f'outputs/models/{model_file}')
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        print(f"\n{name}:")
        
        # Load feature names
        import numpy as np
        DATA_PATH = Path('data/processed/features_v2.npz')
        with np.load(DATA_PATH) as data:
            feature_names = data['feature_names'].tolist()
        
        ablated = ['Unit1', 'Unit2', 'HospAdmTime']
        ablated_idx = [i for i, n in enumerate(feature_names) if any(a in n for a in ablated)]
        feature_names_clean = [n for i, n in enumerate(feature_names) if i not in ablated_idx]
        
        # Check for ICULOS, HospAdmTime, Unit1, Unit2
        has_iculos = any('ICULOS' in f for f in feature_names_clean)
        has_hospadmtime = any('HospAdmTime' in f for f in feature_names_clean)
        has_unit1 = any('Unit1' in f for f in feature_names_clean)
        has_unit2 = any('Unit2' in f for f in feature_names_clean)
        
        print(f"  ICULOS: {'YES' if has_iculos else 'NO'}")
        print(f"  HospAdmTime: {'YES' if has_hospadmtime else 'NO'}")
        print(f"  Unit1: {'YES' if has_unit1 else 'NO'}")
        print(f"  Unit2: {'YES' if has_unit2 else 'NO'}")
        
        # Top 10 features by gain
        sorted_indices = np.argsort(importances)[::-1][:10]
        print(f"  Top 10 features by gain:")
        for idx in sorted_indices:
            print(f"    {idx}: {feature_names_clean[idx]} ({importances[idx]:.4f})")
