# Project Structure

This is a small, sequential research prototype: raw PSV patient files become normalized six-hour windows, then a trained model, evaluation figures, an API, and a browser dashboard. Paths below are relative to the project root.

## Python pipeline (`src/`)

| File | What it does and reads | Produces / connects to |
|---|---|---|
| `00_download_sample.py` | Optionally reads `training_setA.zip` (or checks the PhysioNet archive); extracts the first 2,000 `.psv` patient files. | Writes `data/raw/*.psv`, the input to preprocessing. |
| `01_preprocessing.py` | Reads every raw pipe-separated PSV file; forward/back-fills values, adds lab `_was_measured` flags, creates 6-hour windows, labels each final hour, and z-score normalizes. | Writes `data/processed/data.npz` with `X`, `y`, feature names, medians, means, and standard deviations. |
| `02_ml_baseline.py` | Reads `data.npz`; removes ICULOS, flattens sequences, trains calibrated monotonic XGBoost, evaluates a held-out split and five-fold CV. | Writes `outputs/models/xgb_baseline.pkl`, `xgb_baseline_metrics.json`, and standard ROC/PR/confusion-matrix plots. This is the API’s active model. |
| `03_lstm_baseline.py` | Reads the same 3-D `data.npz` sequences and trains a small LSTM comparison model. | Writes `outputs/models/lstm_baseline.keras` and `outputs/plots/lstm_history.png`. It requires a TensorFlow-compatible Python installation. |
| `04_explainability.py` | Reads `data.npz` and the trained XGBoost artifact; averages TreeSHAP values across the calibrated tree estimators. | Prints the top features and writes `outputs/plots/shap_summary.png`. |
| `05_sanity_checks.py` | Reads `data.npz` statistics and `xgb_baseline.pkl`; sends hand-crafted normal and abnormal vital patterns through the same median-filled representation as the API. | Prints risk scores and monotonic-direction checks; it does not create a model. |
| `06_final_plots.py` | Reads `data.npz` and the active XGBoost model, recreating the held-out split used for evaluation. | Writes polished ROC, PR, confusion-matrix, SHAP, and leakage-comparison figures under `outputs/plots/final/`. |
| `api.py` | On startup, reads `data.npz` plus `xgb_baseline.pkl`, prepares held-out sampling pools and SHAP explainers. | Serves `GET /health`, `GET /sample-patient?risk=high|low`, and `POST /predict` for the frontend. |

## Browser app (`frontend/src/`)

| File | What it does and reads | Produces / connects to |
|---|---|---|
| `main.jsx` | React entry point; imports the application component and global CSS. | Mounts the dashboard into Vite’s HTML page. |
| `App.jsx` | Defines the six vital controls, high/low/random sample buttons, prediction request, score/badge display, and CSS SHAP bars. | Calls the Vite-proxied `/api/sample-patient` and `/api/predict` endpoints in `api.py`. |
| `styles.css` | Defines the responsive medical-dashboard layout, colours, cards, controls, and SHAP bars. | Styles the components rendered by `App.jsx`. |

`frontend/vite.config.js` is the bridge between the two sides: it proxies browser requests beginning `/api` to FastAPI on port 8000, avoiding a development-time cross-origin problem.
