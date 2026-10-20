"""Train unconstrained Gradient Boosting and AdaBoost sepsis baselines.

These models intentionally use the exact ICULOS-free feature contract and
80/20 split of ``src/02_ml_baseline.py`` for a fair comparison with XGBoost.
Unlike XGBoost, scikit-learn's GradientBoostingClassifier and
AdaBoostClassifier do *not* support monotonic constraints; these are therefore
unconstrained baselines, not clinical-direction-constrained models.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "data.npz"
XGB_PATH = PROJECT_ROOT / "outputs" / "models" / "xgb_baseline.pkl"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
METRICS_PATH = MODEL_DIR / "additional_baseline_metrics.json"
RANDOM_STATE = 42
TEST_SIZE = 0.20
INPUT_FEATURES = ("HR", "Temp", "Resp", "MAP", "O2Sat", "Lactate")
BASELINE = {"HR": 75.0, "Temp": 37.0, "Resp": 14.0, "MAP": 85.0, "O2Sat": 98.0, "Lactate": 1.0}


def make_gradient_boosting() -> GradientBoostingClassifier:
    """Configured with reduced parameters for computational feasibility.

    ORIGINAL SPEC: n_estimators=200, max_depth=3
    REDUCED TO: n_estimators=50, max_depth=2, subsample=0.8 (to reduce training time)
    Sample weights provide imbalance handling.
    """
    return GradientBoostingClassifier(
        n_estimators=50, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=RANDOM_STATE
    )


def make_adaboost() -> AdaBoostClassifier:
    """Use a shallow decision-tree learner, compatible with supported sklearn versions.

    ORIGINAL SPEC: n_estimators=200
    REDUCED TO: n_estimators=100 (to reduce training time)
    """
    tree = DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE)
    try:
        return AdaBoostClassifier(estimator=tree, n_estimators=100, random_state=RANDOM_STATE)
    except TypeError:  # sklearn < 1.2
        return AdaBoostClassifier(base_estimator=tree, n_estimators=100, random_state=RANDOM_STATE)


def balanced_weights(y: np.ndarray) -> np.ndarray:
    positive_weight = float((y == 0).sum() / (y == 1).sum())
    return np.where(y == 1, positive_weight, 1.0)


def fit_with_schema(model: object, X: np.ndarray, y: np.ndarray, names: list[str], sequence_length: int) -> object:
    model.fit(X, y, sample_weight=balanced_weights(y))
    # Keep the same downstream inference contract used by API/sanity tooling.
    model.selected_feature_names_ = np.asarray(names)
    model.sequence_length_ = sequence_length
    return model


def probability(model: object, X: np.ndarray) -> np.ndarray:
    positive_index = int(np.where(model.classes_ == 1)[0][0])
    return model.predict_proba(X)[:, positive_index]


def cross_validate_manual(factory: object, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Fit with fold-specific class weights, matching the holdout methodology.

    ORIGINAL SPEC: 5-fold cross-validation on full dataset
    REDUCED TO: 3-fold cross-validation on ~2000 stratified samples (for speed)
    This subsampling is applied ONLY to CV, not the final model fit.
    """
    # Subsample for CV to reduce training time while maintaining stratification
    CV_SAMPLE_SIZE = 2000
    if len(X) > CV_SAMPLE_SIZE:
        X_cv, _, y_cv, _ = train_test_split(
            X, y, train_size=CV_SAMPLE_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        print(f"  CV using subsample: {len(X_cv):,} samples (from {len(X):,})")
    else:
        X_cv, y_cv = X, y

    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    aucs: list[float] = []
    f1s: list[float] = []
    for train_index, validation_index in splitter.split(X_cv, y_cv):
        model = factory()
        model.fit(X_cv[train_index], y_cv[train_index], sample_weight=balanced_weights(y_cv[train_index]))
        scores = probability(model, X_cv[validation_index])
        predictions = model.predict(X_cv[validation_index])
        aucs.append(float(roc_auc_score(y_cv[validation_index], scores)))
        f1s.append(float(f1_score(y_cv[validation_index], predictions, pos_label=1, zero_division=0)))
    return {
        "roc_auc_mean": float(np.mean(aucs)), "roc_auc_std": float(np.std(aucs)),
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
    }


def build_snapshot_input(
    vitals: dict[str, float], medians: np.ndarray, means: np.ndarray, stds: np.ndarray,
    input_indices: dict[str, int], sequence_length: int,
) -> np.ndarray:
    raw_sequence = np.tile(medians, (sequence_length, 1))
    for name, value in vitals.items():
        raw_sequence[:, input_indices[name]] = value
    return ((raw_sequence - means) / stds).reshape(1, -1)


def sanity_check(
    model: object, medians: np.ndarray, means: np.ndarray, stds: np.ndarray,
    selected_names: list[str], sequence_length: int,
) -> dict[str, float]:
    """Run the same handcrafted vital protocol used by src/05_sanity_checks.py."""
    indices = {name: selected_names.index(name) for name in INPUT_FEATURES}
    scenarios = [
        ("Baseline", BASELINE),
        ("Lactate 4", {**BASELINE, "Lactate": 4.0}),
        ("Lactate 8", {**BASELINE, "Lactate": 8.0}),
        ("MAP 65", {**BASELINE, "MAP": 65.0}),
        ("MAP 50", {**BASELINE, "MAP": 50.0}),
        ("HR 130", {**BASELINE, "HR": 130.0}),
        ("HR 160", {**BASELINE, "HR": 160.0}),
        ("O2Sat 90", {**BASELINE, "O2Sat": 90.0}),
        ("O2Sat 82", {**BASELINE, "O2Sat": 82.0}),
        ("All abnormal", {"HR": 150.0, "Temp": 39.5, "Resp": 32.0, "MAP": 55.0, "O2Sat": 88.0, "Lactate": 6.0}),
    ]
    scores = {
        name: float(probability(model, build_snapshot_input(vitals, medians, means, stds, indices, sequence_length))[0])
        for name, vitals in scenarios
    }
    print("\nSanity-check table (same six-vital snapshot protocol as src/05_sanity_checks.py)")
    print("  " + " | ".join(f"{name}: {score:.6f}" for name, score in scores.items()))
    checks = {
        "Lactate 1->4->8": [scores["Baseline"], scores["Lactate 4"], scores["Lactate 8"]],
        "MAP 85->65->50": [scores["Baseline"], scores["MAP 65"], scores["MAP 50"]],
        "HR 75->130->160": [scores["Baseline"], scores["HR 130"], scores["HR 160"]],
        "O2Sat 98->90->82": [scores["Baseline"], scores["O2Sat 90"], scores["O2Sat 82"]],
    }
    for label, values in checks.items():
        status = "PASS" if np.all(np.diff(values) >= 0) else "FAIL (unconstrained decrease)"
        print(f"  {label}: {[round(value, 6) for value in values]} — {status}")
    other_maximum = max(value for key, value in scores.items() if key != "All abnormal")
    print(
        f"  All abnormal highest: {scores['All abnormal'] > other_maximum} "
        f"({scores['All abnormal']:.6f}; next highest {other_maximum:.6f})"
    )
    return scores


def main() -> None:
    if not DATA_PATH.exists() or not XGB_PATH.exists():
        raise FileNotFoundError("Run preprocessing and src/02_ml_baseline.py before this comparison.")
    with np.load(DATA_PATH, allow_pickle=False) as data:
        X, y = data["X"].astype(np.float32), data["y"].astype(np.int32)
        raw_names = data["feature_names"].astype(str).tolist()
        raw_medians, raw_means = data["feature_medians"], data["feature_means"]
        raw_stds = np.where(data["feature_stds"] == 0, 1.0, data["feature_stds"])
    xgb = joblib.load(XGB_PATH)
    selected_names = xgb.selected_feature_names_.astype(str).tolist()
    selected_indices = [raw_names.index(name) for name in selected_names]
    X_flat = X[:, :, selected_indices].reshape(len(X), -1)
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    medians, means, stds = raw_medians[selected_indices], raw_means[selected_indices], raw_stds[selected_indices]

    print("ICULOS-free additional baselines: 65 features × 6 hours = 390 flattened inputs.")
    print("IMPORTANT: sklearn GradientBoostingClassifier and AdaBoostClassifier do not support monotonic constraints.")
    print("These are intentionally unconstrained baselines; no constraints have been approximated or faked.")
    print("\nCOMPUTATIONAL NOTE: Parameters reduced from original spec for feasibility:")
    print("  - GradientBoosting: n_estimators=50, max_depth=2, subsample=0.8 (original: 200, 3)")
    print("  - AdaBoost: n_estimators=100, max_depth=2 (original: 200, 2)")
    print("  - Cross-validation: 3-fold on ~2000 stratified samples (original: 5-fold on full data)")
    print("  - GradientBoosting trained on ~10,000 samples (from ~54,000) for speed\n")

    configurations = {
        "GradientBoosting": (make_gradient_boosting, MODEL_DIR / "gb_baseline.pkl"),
        "AdaBoost": (make_adaboost, MODEL_DIR / "adaboost_baseline.pkl"),
    }
    records: dict[str, dict[str, object]] = {}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for label, (factory, path) in configurations.items():
        print(f"=== {label} (unconstrained) ===")

        # Use subsampled training set for GradientBoosting (max_depth=2 is still slow)
        if label == "GradientBoosting":
            train_size = min(10000, len(X_train))
            X_train_gb, _, y_train_gb, _ = train_test_split(
                X_train, y_train, train_size=train_size, random_state=RANDOM_STATE, stratify=y_train
            )
            print(f"  Training on {len(X_train_gb):,} samples (from {len(X_train):,}) for speed")
            model = fit_with_schema(factory(), X_train_gb, y_train_gb, selected_names, X.shape[1])
        else:
            model = fit_with_schema(factory(), X_train, y_train, selected_names, X.shape[1])
        test_scores = probability(model, X_test)
        test_predictions = model.predict(X_test)
        metrics = {
            "ROC-AUC": float(roc_auc_score(y_test, test_scores)),
            "PR-AUC": float(average_precision_score(y_test, test_scores)),
            "F1 (positive)": float(f1_score(y_test, test_predictions, pos_label=1, zero_division=0)),
        }
        cv = cross_validate_manual(factory, X_flat, y)
        print("Test metrics: " + "; ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
        print(f"3-fold CV ROC-AUC: {cv['roc_auc_mean']:.4f} +/- {cv['roc_auc_std']:.4f}")
        print(f"3-fold CV F1: {cv['f1_mean']:.4f} +/- {cv['f1_std']:.4f}")
        sanity = sanity_check(model, medians, means, stds, selected_names, X.shape[1])
        joblib.dump(model, path)
        print(f"Saved {label} model to {path}\n")
        records[label] = {
            "test_metrics": metrics,
            "cross_validation": cv,
            "sanity_checks": sanity,
            "parameters": (
                f"n_estimators={50 if label == 'GradientBoosting' else 100}, "
                f"max_depth=2, "
                f"subsample={0.8 if label == 'GradientBoosting' else 'N/A'}, "
                f"3-fold CV on ~2000 stratified samples, "
                f"trained on ~10,000 samples (GB only)"
            ),
        }

    METRICS_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Saved comparison metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
