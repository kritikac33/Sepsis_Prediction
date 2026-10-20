# Sepsis Early Detection — Prototype Model Card

## Purpose and dataset

This Week 2 prototype estimates sepsis risk from six-hour ICU time windows. It uses the **PhysioNet/CinC Challenge 2019** training-set-A data, restricted to **2,000 patient PSV files**. Preprocessing produced **15,273 six-hour windows**: **365 sepsis-positive (2.39%)** and **14,908 non-sepsis (97.61%)**. Labels are the `SepsisLabel` value at the final hour of each window. Values are forward-filled, back-filled, then zero-filled within each patient; z-score statistics and medians are saved with the processed dataset.

## Features and leakage control

The data contain 66 engineered input features: 40 clinical/demographic variables plus 26 pre-imputation laboratory-measurement flags. **ICULOS (ICU length of stay) is deliberately excluded** from model fitting. It was strongly correlated with the label and drove the previous near-perfect result, making it a leakage-prone proxy for when sepsis is labelled rather than a clinically useful vital-sign signal.

**Clinical/demographic variables:** HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2, BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Phosphate, Potassium, Bilirubin_total, TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets, Age, Gender, Unit1, Unit2, HospAdmTime.  
**Measurement flags:** BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Phosphate, Potassium, Bilirubin_total, TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, and Platelets — each with a `_was_measured` suffix.

Each retained feature is represented at all six hours (65 retained features × 6 = 390 flattened inputs). The six user-adjustable vitals use clinical monotonic constraints: HR, Temp, Resp, and Lactate increase risk; MAP and O2Sat decrease risk as their values rise.

## Current model

The active artifact is **calibrated monotonic XGBoost**, not the earlier Random Forest baseline. It uses 300 trees, maximum depth 4, learning rate 0.05, subsample 0.8, column subsample 0.8, histogram tree construction, random seed 42, and imbalance weighting through `scale_pos_weight`. Platt/sigmoid calibration uses five folds. An 80/20 stratified window-level split (seed 42) is used for the reported test results.

## Final evaluation

| Held-out test metric | Result |
|---|---:|
| ROC-AUC | 0.9969 |
| PR-AUC (average precision) | 0.9154 |
| Accuracy | 0.9928 |
| No-sepsis precision / recall / F1 | 0.9966 / 0.9960 / 0.9963 |
| Sepsis precision / recall / F1 | 0.8400 / 0.8630 / 0.8514 |

Held-out confusion matrix (actual rows, predicted columns): `[[2970, 12], [10, 63]]` for No Sepsis and Sepsis respectively. Five-fold stratified CV ROC-AUC is **0.9977 ± 0.0004**. Default-threshold CV F1 is **0.0000 ± 0.0000**: after calibration, no fold reaches sklearn's fixed 0.50 probability cutoff. This discrepancy makes explicit threshold selection/validation a required next step; it should not be interpreted as clinical readiness.

## Limitations and next steps

- This is a single-source prototype with 2,000 patients; it has no external or prospective validation. The window-level split may also allow related windows from a patient in both partitions, so patient-level evaluation is needed.
- Severe class imbalance remains. XGBoost uses class weighting (`scale_pos_weight`); SMOTE or other resampling strategies have not yet been evaluated.
- Imputation and the `_was_measured` flags may reflect local measurement practices as well as physiology. Outputs are educational/research only and are not for clinical decisions.

Planned modules: mortality prediction, K-means patient subtyping, causal inference, GAN-based data augmentation, and Hugging Face clinical NLP. Near-term evaluation work will add patient-level splits, threshold selection, and external validation.
