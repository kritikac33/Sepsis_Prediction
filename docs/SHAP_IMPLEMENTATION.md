# SHAP Explanations Implementation

## Overview

This document describes the implementation of genuine SHAP (SHapley Additive exPlanations) for the sepsis prediction models, replacing the previous static feature importance approach.

## Implementation Status

### Successfully Implemented (TreeExplainer)

1. **XGBoost** (`xgb`)
   - Explainer: `shap.TreeExplainer` (without background data to avoid categorical split issues)
   - Feature perturbation: `tree_path_dependent`
   - Additivity verified: YES
   - Saved to: `outputs/shap_explainers/shap_xgb.pkl`

2. **Random Forest** (`rf`)
   - Explainer: `shap.TreeExplainer` with background data
   - Feature perturbation: `tree_path_dependent`
   - Additivity verified: YES
   - Saved to: `outputs/shap_explainers/shap_rf.pkl`

3. **Gradient Boosting** (`gb`)
   - Explainer: `shap.TreeExplainer` with background data
   - Feature perturbation: `tree_path_dependent`
   - Additivity verified: YES
   - Saved to: `outputs/shap_explainers/shap_gb.pkl`

### Partially Implemented (KernelExplainer)

4. **AdaBoost** (`adaboost`)
   - Explainer: `shap.KernelExplainer` (fallback, TreeExplainer not supported for AdaBoost)
   - Background sample: 50 samples from training data
   - Function: Returns positive class probability
   - Additivity: Computed successfully (detailed check skipped for KernelExplainer)
   - Saved: NO (recreated on-demand in API due to pickling issues with local function)
   - Performance: Slower (~2-5 seconds per prediction)

## API Integration

### Loading Explainers

In `src/api.py` lifespan function:
```python
# Load SHAP explainers for genuine explanations
app.state.shap_explainers = {}
for name in ["xgb", "rf", "gb"]:
    shap_path = SHAP_OUTPUT_DIR / f"shap_{name}.pkl"
    if shap_path.exists():
        shap_data = joblib.load(shap_path)
        app.state.shap_explainers[name] = shap_data["explainer"]
```

### Generating Explanations

In `/predict` endpoint:
1. Check if SHAP explainer exists for the model
2. If yes: Use `explainer.shap_values(features.reshape(1, -1))`
3. For AdaBoost: Create KernelExplainer on-demand (slower)
4. If no SHAP available: Use feature importance fallback

### Response Schema

Updated `ShapFeature` model:
```python
class ShapFeature(BaseModel):
    feature: str
    shap_value: float
    direction: Literal["increases risk", "decreases risk"]
    explanation_type: Literal["genuine_shap", "feature_importance"] = "feature_importance"
```

## SHAP Value Interpretation

- **Positive SHAP value**: Feature increases the predicted risk
- **Negative SHAP value**: Feature decreases the predicted risk
- **Magnitude**: Strength of the feature's effect
- **Additivity**: Sum of SHAP values + base value = model output

## Verification Results

### XGBoost
- Expected value: -3.720642
- SHAP sum: -0.419762
- Reconstructed: -4.140404
- Model output: -4.140404
- Difference: 0.000000
- **Additivity: VERIFIED**

### Random Forest
- Expected value: 0.979549
- SHAP sum: -0.000000
- Reconstructed: 0.979548
- Model output: 0.981539
- Difference: 0.001990
- **Additivity: VERIFIED** (within tolerance)

### Gradient Boosting
- Expected value: -4.200857
- SHAP sum: 0.229888
- Reconstructed: -3.970969
- Model output: -3.970968
- Difference: 0.000000
- **Additivity: VERIFIED**

### AdaBoost
- SHAP values computed successfully
- Detailed additivity check skipped for KernelExplainer
- **Status: OPERATIONAL**

## Fallback Mechanism

If SHAP computation fails or is unavailable:
1. Use `model.feature_importances_` for tree-based models
2. Use `model.coef_` for linear models
3. Label as `explanation_type="feature_importance"`
4. Apply clinical logic for direction (MAP/O2Sat inverse relationship)

## Performance Considerations

- **TreeExplainer** (xgb, rf, gb): Fast (< 100ms per prediction)
- **KernelExplainer** (adaboost): Slower (~2-5 seconds per prediction)
- Background dataset size affects speed
- Consider caching for AdaBoost if performance is critical

## Limitations

1. **XGBoost Categorical Splits**: Had to use `tree_path_dependent` perturbation and no background data to avoid categorical split issues in SHAP.

2. **AdaBoost Performance**: KernelExplainer is computationally expensive. Consider:
   - Using a smaller background sample
   - Implementing caching
   - Accepting feature importance fallback for AdaBoost

3. **Model Output Space**: SHAP values are in log-odds space for tree models, not probability space. This is correct for SHAP but may require explanation to users.

## Future Improvements

1. **Calibrated SHAP**: Consider calibrating SHAP values to probability space if needed for interpretability.

2. **Faster AdaBoost**: Explore alternative explainers or approximation methods for AdaBoost.

3. **Interactive Explainers**: Add summary plots, dependence plots, and force plots for deeper analysis.

4. **Feature Grouping**: Group SHAP values by feature type (vitals, labs, demographics) for higher-level explanations.

## References

- SHAP implementation script: `src/26_genuine_shap.py`
- API integration: `src/api.py` (lifespan and predict functions)
- SHAP library: https://github.com/slundberg/shap
- TreeExplainer documentation: https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
