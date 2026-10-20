"""Debug feature importances."""

import pickle
import numpy as np

m = pickle.load(open(r'outputs\models\xgb_final_v2.pkl', 'rb'))
print('Monotone constraints:', m['model'].get_params()['monotone_constraints'])

fn = ['HR_last','HR_mean','HR_min','HR_max','HR_slope','O2Sat_last','O2Sat_mean','O2Sat_min','O2Sat_max','O2Sat_slope','Temp_last','Temp_mean','Temp_min','Temp_max','Temp_slope','Resp_last','Resp_mean','Resp_min','Resp_max','Resp_slope','MAP_last','MAP_mean','MAP_min','MAP_max','MAP_slope','Lactate_last','Lactate_mean','Lactate_min','Lactate_max','Lactate_slope']

print('\nNon-zero importance features (first 30):')
for i, imp in enumerate(m['model'].feature_importances_):
    if imp > 0 and i < 30:
        print(f'{i}: {fn[i] if i < len(fn) else i} = {imp:.4f}')

print(f'\nLactate_last (idx 110) importance: {m["model"].feature_importances_[110]:.4f}')
print(f'MAP_last (idx 20) importance: {m["model"].feature_importances_[20]:.4f}')
print(f'HR_last (idx 0) importance: {m["model"].feature_importances_[0]:.4f}')
