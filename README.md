# Sepsis Prototype

Prototype project for sepsis prediction and analysis.

## Project structure

- `data/` — local dataset files (not tracked by Git)
- `src/` — Python source code
- `notebooks/` — exploratory notebooks
- `outputs/` — generated figures, models, and reports
- `frontend/` — React frontend application

## Step-by-Step Setup Guide

### Prerequisites
- Python 3.10 or higher
- Node.js 16 or higher
- PowerShell (on Windows)

### Step 1: Create and activate Python virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 2: Install Python dependencies

```powershell
pip install -r requirements.txt
```

### Step 3: Download and prepare the dataset

The project uses PhysioNet/CinC Challenge 2019 data. You need to:
1. Download the training-set-A data from PhysioNet
2. Extract PSV files to `data/raw/` directory

Or if you already have the processed data, skip to Step 4.

### Step 4: Process the dataset

```powershell
.\venv\Scripts\python.exe src\01_preprocessing.py
```

This creates `data/processed/data.npz` with normalized six-hour ICU sequences.

### Step 5: Train the models

```powershell
# Train baseline XGBoost model
.\venv\Scripts\python.exe src\02_ml_baseline.py

# Train Random Forest baseline model
.\venv\Scripts\python.exe src\08_train_random_forest.py

# Train additional models (Gradient Boosting, AdaBoost)
.\venv\Scripts\python.exe src\09_additional_models.py

# Optional: Train subtyping model
.\venv\Scripts\python.exe src\06_subtyping.py
```

This creates model files in `outputs/models/`:
- `xgb_baseline.pkl` (primary model)
- `rf_baseline.pkl` (Random Forest)
- `gb_baseline.pkl` (Gradient Boosting)
- `adaboost_baseline.pkl` (AdaBoost)
- `kmeans_subtypes.pkl` (optional, for patient subtypes)

### Step 6: Start the backend API

```powershell
uvicorn src.api:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### Step 7: Install frontend dependencies (in a new terminal)

```powershell
cd frontend
npm install
```

### Step 8: Start the frontend development server

```powershell
npm run dev
```

The frontend will be available at `http://localhost:5173` (or the URL shown in the terminal)

### Step 9: Access the application

Open your browser to the frontend URL (typically `http://localhost:5173`)

The frontend automatically proxies `/api` requests to the FastAPI backend on port 8000.

## Verification

To verify everything is working:

1. Check backend health: `http://localhost:8000/health`
2. Open the frontend: `http://localhost:5173`
3. Try loading a sample patient or entering vital measurements

## Troubleshooting

**Backend won't start:**
- Ensure processed data exists at `data/processed/data.npz`
- Ensure models exist in `outputs/models/`
- Check that port 8000 is not in use

**Frontend won't start:**
- Ensure Node.js is installed: `node --version`
- Ensure dependencies are installed: `npm install`
- Check that port 5173 is not in use

**Model prediction errors:**
- Ensure the model has the `selected_feature_names_` attribute
- Re-train models if they were trained with different feature sets

> **TensorFlow compatibility:** the available Python 3.14 interpreter does not
> currently have a TensorFlow wheel on PyPI. The remaining listed dependencies
> are installed in `venv`; use a TensorFlow-supported Python version to install
> the complete requirements set.
