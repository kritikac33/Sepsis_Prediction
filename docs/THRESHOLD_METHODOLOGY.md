# Risk Threshold Selection Methodology

## Overview

This document describes the methodology used to select risk thresholds for the sepsis prediction prototype. **These thresholds are PROVISIONAL/EXPERIMENTAL and are NOT clinically validated.** They should NOT be used for clinical decision making.

## Selection Method

### Previous (Invalid) Approach
The original implementation used test-set percentiles to select thresholds:
- Red threshold: 50th percentile of test scores
- Yellow threshold: 30th percentile of test scores

**This approach is invalid** because it uses the test set for threshold selection, which introduces optimistic bias and violates the principle that the test set should be held out for final evaluation only.

### Current (Validation-Based) Approach

The current implementation uses a **train/validation/test split with no patient overlap**:

1. **Split Configuration:**
   - Original train set (80% of patients) split into:
     - New train set (64% of patients)
     - Validation set (16% of patients)
   - Test set remains unchanged (20% of patients)
   - Total: 64% train / 16% validation / 20% test
   - No patient appears in more than one split

2. **Threshold Selection on Validation Set:**
   - For each model, compute operating characteristics (sensitivity, specificity, precision, F1) across a range of thresholds on the validation set
   - Select thresholds based on target operating characteristics:
     - **Red threshold (high risk):** Target sensitivity ≥ 0.85, specificity ≥ 0.5
     - **Yellow threshold (intermediate risk):** Target sensitivity ≥ 0.70, specificity ≥ 0.6
   - If no threshold meets targets, select the threshold with best F1 score

3. **Threshold Values (from validation set):**

   | Model | Red Threshold | Yellow Threshold | Red Sensitivity | Red Specificity | Red Precision | Yellow Sensitivity | Yellow Specificity | Yellow Precision |
   |-------|---------------|------------------|-----------------|-----------------|---------------|-------------------|-------------------|------------------|
   | XGBoost | 0.0156 | 0.0125 | 0.865 | 0.510 | 0.048 | 0.798 | 0.642 | 0.060 |
   | Random Forest | 0.0180 | 0.0144 | 0.856 | 0.603 | 0.058 | 0.819 | 0.663 | 0.065 |
   | Gradient Boosting | 0.0514 | 0.0168 | 0.447 | 0.900 | 0.113 | 0.737 | 0.661 | 0.059 |
   | AdaBoost | 0.3131 | 0.2505 | 0.851 | 0.520 | 0.048 | 0.760 | 0.703 | 0.068 |

## Limitations and Caveats

1. **Not Clinically Validated:** These thresholds are selected on a single validation set from a research dataset. They have not been validated in clinical practice or on external datasets.

2. **Low Precision:** Due to the low prevalence of sepsis in the dataset (~2.8%), even high-sensitivity thresholds have low precision (4-11%). This means many false positives for each true positive.

3. **Dataset-Specific:** Thresholds are specific to this dataset and preprocessing pipeline. They may not generalize to other patient populations or data collection methods.

4. **Research Prototype Only:** This is a research prototype for demonstration purposes. The thresholds should be re-evaluated with:
   - Larger, more diverse datasets
   - Clinical expert input
   - Cost-benefit analysis of false positives vs false negatives
   - External validation studies

5. **No Clinical Utility Assessment:** The thresholds have not been assessed for clinical utility, impact on patient outcomes, or integration into clinical workflows.

## Recommendations for Future Work

1. **Clinical Validation:** Conduct prospective clinical studies to validate thresholds in real-world settings.

2. **Cost-Sensitive Analysis:** Perform cost-sensitive analysis to balance the costs of false positives (unnecessary interventions) vs false negatives (missed sepsis cases).

3. **Expert Review:** Have clinical experts review and adjust thresholds based on clinical judgment and risk tolerance.

4. **External Validation:** Validate thresholds on external datasets from different hospitals and patient populations.

5. **Dynamic Thresholding:** Consider patient-specific or context-specific thresholds based on risk factors, comorbidities, or clinical setting.

## References

- Threshold selection script: `src/24_validation_thresholds.py`
- Threshold data: `data/processed/validation_thresholds.json`
- API implementation: `src/api.py` (lifespan function)

## Disclaimer

**WARNING:** These thresholds are PROVISIONAL/EXPERIMENTAL and are NOT clinically validated. They should NOT be used for clinical decision making. This is a research prototype for demonstration purposes only.
