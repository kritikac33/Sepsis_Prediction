"""Finish the ICULOS ablation without modifying existing models or the split.

Creates only new artifacts:
* outputs/models/gb_noiculos.pkl and adaboost_noiculos.pkl
* outputs/metrics/iculos_ablation_report.json
* data/processed/band_thresholds_noiculos.json

The held-out patient split in data/processed/split.json is read-only.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "features_v2.npz"
MODELS = ROOT / "outputs" / "models"
METRICS_DIR = ROOT / "outputs" / "metrics"
BANDS = ROOT / "data" / "processed" / "band_thresholds_noiculos.json"
RANDOM_SEED = 20261015
BOOTSTRAPS = 200


def filtered_features(data: np.lib.npyio.NpzFile):
    """Return fixed with-/without-ICULOS matrices without changing the source split."""
    names = data["feature_names"].astype(str).tolist()
    keep = np.array(
        [not any(token in name for token in ("Unit1", "Unit2", "HospAdmTime")) for name in names],
        dtype=bool,
    )
    with_names = [name for name, include in zip(names, keep) if include]
    no_iculos = np.array(["ICULOS" not in name for name in with_names], dtype=bool)
    return (
        data["X_train"][:, keep],
        data["X_test"][:, keep],
        with_names,
        data["X_train"][:, keep][:, no_iculos],
        data["X_test"][:, keep][:, no_iculos],
        [name for name, include in zip(with_names, no_iculos) if include],
    )


def ci(values: list[float]) -> list[float]:
    return [float(item) for item in np.percentile(values, [2.5, 97.5])]


def patient_bootstrap(
    y: np.ndarray, patient_ids: np.ndarray, score_pairs: dict[str, tuple[np.ndarray, np.ndarray]]
) -> dict[str, dict[str, object]]:
    """Patient-resample AUROC/PR CIs and paired with-minus-without AUROC CIs."""
    patient_to_rows: dict[str, np.ndarray] = {
        patient: np.flatnonzero(patient_ids == patient) for patient in np.unique(patient_ids)
    }
    patients = np.array(list(patient_to_rows))
    rng = np.random.default_rng(RANDOM_SEED)
    res = {name: {"with_auc": [], "without_auc": [], "with_pr_auc": [], "without_pr_auc": [], "delta_auc": []}
           for name in score_pairs}
    for _ in range(BOOTSTRAPS):
        chosen = rng.choice(patients, size=len(patients), replace=True)
        rows = np.concatenate([patient_to_rows[patient] for patient in chosen])
        y_sample = y[rows]
        # The supplied fixed split has positive patients in test; retain a guard for degenerate resamples.
        if np.unique(y_sample).size < 2:
            continue
        for name, (with_scores, no_scores) in score_pairs.items():
            with_auc = roc_auc_score(y_sample, with_scores[rows])
            without_auc = roc_auc_score(y_sample, no_scores[rows])
            res[name]["with_auc"].append(with_auc)
            res[name]["without_auc"].append(without_auc)
            res[name]["with_pr_auc"].append(average_precision_score(y_sample, with_scores[rows]))
            res[name]["without_pr_auc"].append(average_precision_score(y_sample, no_scores[rows]))
            res[name]["delta_auc"].append(with_auc - without_auc)
    return {
        name: {
            "with_iculos_auc_ci95": ci(values["with_auc"]),
            "without_iculos_auc_ci95": ci(values["without_auc"]),
            "with_iculos_pr_auc_ci95": ci(values["with_pr_auc"]),
            "without_iculos_pr_auc_ci95": ci(values["without_pr_auc"]),
            "paired_with_minus_without_auc_ci95": ci(values["delta_auc"]),
            "bootstrap_resamples": len(values["delta_auc"]),
        }
        for name, values in res.items()
    }


def model_record(with_model, without_model, x_with, x_without, y, bootstrap):
    with_scores = with_model.predict_proba(x_with)[:, 1]
    without_scores = without_model.predict_proba(x_without)[:, 1]
    return {
        "with_iculos": {
            "test_auroc": float(roc_auc_score(y, with_scores)),
            "test_pr_auc": float(average_precision_score(y, with_scores)),
            "model_file": None,
        },
        "without_iculos": {
            "test_auroc": float(roc_auc_score(y, without_scores)),
            "test_pr_auc": float(average_precision_score(y, without_scores)),
            "model_file": None,
        },
        "bootstrap": bootstrap,
    }, with_scores, without_scores


def oof_bands(model, x, y, groups) -> dict[str, float]:
    """Three-fold GroupKFold OOF scores on TRAIN patients only."""
    scores = np.empty(len(y), dtype=float)
    splitter = GroupKFold(n_splits=3)
    for fold, (fit_rows, valid_rows) in enumerate(splitter.split(x, y, groups=groups), start=1):
        fitted = clone(model)
        fitted.fit(x[fit_rows], y[fit_rows])
        scores[valid_rows] = fitted.predict_proba(x[valid_rows])[:, 1]
        print(f"  OOF fold {fold}/3 complete", flush=True)
    return {"yellow": float(np.percentile(scores, 75)), "red": float(np.percentile(scores, 90))}


def main() -> None:
    with np.load(DATA, allow_pickle=True) as data:
        x_train_with, x_test_with, with_names, x_train_no, x_test_no, no_names = filtered_features(data)
        y_train = data["y_train"].astype(int)
        y_test = data["y_test"].astype(int)
        groups_train = data["patient_ids_train"].astype(str)
        groups_test = data["patient_ids_test"].astype(str)

    print(f"Fixed split: train patients={len(np.unique(groups_train))}, test patients={len(np.unique(groups_test))}")
    print(f"Feature count: with ICULOS={len(with_names)}, without ICULOS={len(no_names)}")
    print(f"Test base rate={y_test.mean():.6f} ({y_test.sum()}/{len(y_test)})")

    with_paths = {
        "xgb": MODELS / "xgboost_v2.pkl",
        "rf": MODELS / "random_forest_v2.pkl",
        "gb": MODELS / "gradient_boosting_v2.pkl",
        "adaboost": MODELS / "adaboost_v2.pkl",
    }
    no_paths = {
        "xgb": MODELS / "xgb_noiculos.pkl",
        "rf": MODELS / "rf_noiculos.pkl",
        "gb": MODELS / "gb_noiculos.pkl",
        "adaboost": MODELS / "adaboost_noiculos.pkl",
    }
    with_models = {name: joblib.load(path) for name, path in with_paths.items()}
    no_models = {name: joblib.load(path) for name, path in no_paths.items() if path.exists()}

    # Clone the saved V2 estimators, preserving their actual hyperparameters.
    # If the requested new artifact already exists, it is deliberately reused:
    # this script must never overwrite a model generated in an earlier run.
    for name in ("gb", "adaboost"):
        if no_paths[name].exists():
            no_models[name] = joblib.load(no_paths[name])
            print(f"Reusing existing new artifact: {no_paths[name].name}", flush=True)
        else:
            print(f"Training {name} without ICULOS using saved estimator hyperparameters...", flush=True)
            fitted = clone(with_models[name])
            fitted.fit(x_train_no, y_train)
            joblib.dump(fitted, no_paths[name])
            no_models[name] = fitted
            print(f"Saved new artifact: {no_paths[name].name}", flush=True)

    records: dict[str, dict[str, object]] = {}
    score_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in ("xgb", "rf", "gb", "adaboost"):
        record, scores_with, scores_no = model_record(
            with_models[name], no_models[name], x_test_with, x_test_no, y_test, {}
        )
        record["with_iculos"]["model_file"] = with_paths[name].name
        record["without_iculos"]["model_file"] = no_paths[name].name
        records[name] = record
        score_pairs[name] = (scores_with, scores_no)

    bootstrap = patient_bootstrap(y_test, groups_test, score_pairs)
    for name in records:
        records[name]["bootstrap"] = bootstrap[name]

    xgb_constraints = with_models["xgb"].get_params().get("monotone_constraints")
    metadata = {
        "description": "Patient-level held-out test bootstrap (200 resamples); paired CI is with-ICULOS minus without-ICULOS AUROC.",
        "base_rate": float(y_test.mean()),
        "test_windows": int(len(y_test)),
        "train_patients": int(len(np.unique(groups_train))),
        "test_patients": int(len(np.unique(groups_test))),
        "with_iculos_feature_count": len(with_names),
        "without_iculos_feature_count": len(no_names),
        "xgb_monotone_constraints": xgb_constraints,
        "models": records,
    }
    METRICS_DIR.mkdir(exist_ok=True)
    report_path = METRICS_DIR / "iculos_ablation_report.json"
    report_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nStep 1 point estimates and 95% patient bootstrap CIs")
    for name, record in records.items():
        b = record["bootstrap"]
        print(
            f"{name}: with AUROC={record['with_iculos']['test_auroc']:.4f} "
            f"CI={b['with_iculos_auc_ci95']}; without AUROC={record['without_iculos']['test_auroc']:.4f} "
            f"CI={b['without_iculos_auc_ci95']}; paired with-minus-without={b['paired_with_minus_without_auc_ci95']}"
        )
        print(
            f"      with PR-AUC={record['with_iculos']['test_pr_auc']:.4f} "
            f"CI={b['with_iculos_pr_auc_ci95']}; without PR-AUC={record['without_iculos']['test_pr_auc']:.4f} "
            f"CI={b['without_iculos_pr_auc_ci95']}"
        )

    print("\nStep 2: computing 3-fold TRAIN-only OOF display bands...", flush=True)
    thresholds = {}
    for name in ("xgb", "rf", "gb", "adaboost"):
        print(f"{name}:", flush=True)
        thresholds[name] = oof_bands(no_models[name], x_train_no, y_train, groups_train)
    band_payload = {
        "label": "provisional display bands, not clinical thresholds",
        "method": "3-fold GroupKFold out-of-fold prediction on training patients only; yellow >= 75th percentile, red >= 90th percentile.",
        "feature_set": "ICULOS-derived, Unit1, Unit2, and HospAdmTime columns excluded",
        "thresholds": thresholds,
    }
    BANDS.write_text(json.dumps(band_payload, indent=2), encoding="utf-8")
    print(f"Saved {report_path}")
    print(f"Saved {BANDS}")


if __name__ == "__main__":
    main()
