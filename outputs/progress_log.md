# Progress Log - ICULOS Exclusion Retraining

## Date: 2026-10-15

### Steps:
- [ ] Step 1: Retrain all four models without ICULOS
- [ ] Step 2: Compute provisional bands for all four
- [ ] Step 3: Update API with new models
- [ ] Step 4: Update frontend
- [ ] Step 5: Fix extra parameters panel with real values
- [ ] Step 6: Verify with real calls

### Log:

- 2026-10-15: Read the progress log before work; all listed steps were unchecked.
- 2026-10-15: Backed up `frontend/src/App.jsx` to `frontend/src/App.backup.jsx` and `src/api.py` to `src/api_before_fix.py` before the small crash fix.
- 2026-10-15: Fixed `/model-info` JSON serialization of non-finite XGBoost hyperparameters (`missing: NaN`), added null-safe metric lookups, and verified HTTP 200 for xgb/rf/gb/adaboost on the patched API.
- 2026-10-15: Made the lab-panel response shape null-safe, added an app error boundary, preserved Not measured/Age/Gender display rules, and added static leakage/band captions.
- 2026-10-15: `npm run build` passed. Verified high/low `/sample-patient` -> `/predict` calls for xgb/rf/gb/adaboost on the patched local API (temporary port 8001 because an orphaned process held 8000).
- 2026-10-15: Began isolated all-6,000-patient physiology-only XGBoost improvement run. It writes only new `_xgbimprove` artifacts and does not alter the active split, models, metrics, API, or frontend.
- 2026-10-15: Fixed the new run's incomplete candidate-feature bottleneck by using fixed physiologic ranges for the requested abnormal-reading count, then restarted the isolated run. No test data or existing artifacts were used or changed.
- 2026-10-15: Completed all four train-only candidate representations. Train-only 3-fold PR-AUC: 6h basic 0.06416; 6h extra 0.06972; 12h basic 0.06763; 12h extra 0.07535. Random search is continuing on the selected 12h-extra representation; test patients remain untouched.
- 2026-10-15: Completed the 15-configuration train-only search; configuration 6 was selected (PR-AUC 0.08362). The first finalizer was stopped before fitting because 6h and 12h windows did not share the same endpoints for a valid paired bootstrap. Finalization was corrected to compare the current baseline configuration on matched 12h-basic windows against selected 12h-extra windows.
- 2026-10-15: Inventoried and summarized all current `src/` files on request; no project code or artifacts were changed.
