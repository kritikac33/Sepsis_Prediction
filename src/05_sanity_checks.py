"""Check whether the ICULOS-free monotonic XGBoost model responds sensibly to key vitals.

This is a prototype behavior check, not clinical validation. It evaluates the
same median-filled, six-hour snapshot representation used by ``src/api.py``.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "data.npz"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "xgb_baseline.pkl"
INPUT_FEATURES = ("HR", "Temp", "Resp", "MAP", "O2Sat", "Lactate")

BASELINE = {"HR": 75.0, "Temp": 37.0, "Resp": 14.0, "MAP": 85.0, "O2Sat": 98.0, "Lactate": 1.0}


def load_predictor() -> tuple[object, dict[str, int], np.ndarray, np.ndarray, np.ndarray, int]:
    """Load the model schema and preprocessing statistics used for API inference."""
    if not MODEL_PATH.exists() or not DATA_PATH.exists():
        raise FileNotFoundError("Run preprocessing and src/02_ml_baseline.py before sanity checks.")

    model = joblib.load(MODEL_PATH)
    if not hasattr(model, "selected_feature_names_"):
        raise ValueError("Model has no saved feature schema. Re-run src/02_ml_baseline.py.")

    with np.load(DATA_PATH, allow_pickle=False) as data:
        raw_names = data["feature_names"].astype(str).tolist()
        selected_names = model.selected_feature_names_.astype(str).tolist()
        selected_indices = [raw_names.index(name) for name in selected_names]
        medians = data["feature_medians"][selected_indices].astype(np.float32)
        means = data["feature_means"][selected_indices].astype(np.float32)
        stds = data["feature_stds"][selected_indices].astype(np.float32)

    missing = set(INPUT_FEATURES).difference(selected_names)
    if missing:
        raise ValueError(f"Model schema is missing user-input features: {sorted(missing)}")
    sequence_length = int(model.sequence_length_)
    if model.n_features_in_ != sequence_length * len(selected_names):
        raise ValueError("Model feature count does not match the saved sequence schema.")

    return (
        model,
        {name: selected_names.index(name) for name in INPUT_FEATURES},
        medians,
        means,
        np.where(stds == 0, 1.0, stds),
        sequence_length,
    )


def predict_risk(
    vitals: dict[str, float],
    model: object,
    input_indices: dict[str, int],
    medians: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    sequence_length: int,
) -> float:
    """Create a median-filled sequence with current vitals repeated per hour."""
    raw_sequence = np.tile(medians, (sequence_length, 1))
    for name, value in vitals.items():
        raw_sequence[:, input_indices[name]] = value
    model_input = ((raw_sequence - means) / stds).reshape(1, -1)
    positive_class_index = int(np.where(model.classes_ == 1)[0][0])
    return float(model.predict_proba(model_input)[0, positive_class_index])


def print_case(name: str, vitals: dict[str, float], score: float) -> None:
    values = ", ".join(f"{feature}={vitals[feature]:g}" for feature in INPUT_FEATURES)
    print(f"{name}\n  {values}\n  risk_score={score:.6f}\n")


def assess_monotonicity(name: str, scores: list[float], direction: str = "increasing") -> bool:
    """Report strict increases, flat segments, and clinically unexpected decreases."""
    differences = np.diff(scores)
    if np.any(differences < 0):
        print(f"FAIL - {name}: risk decreased despite progressively more concerning values: {scores}")
        return False
    if np.any(differences == 0):
        print(f"WARNING - {name}: risk did not measurably increase at every step: {scores}")
        return True
    print(f"PASS - {name}: risk increased monotonically: {scores}")
    return True


def main() -> None:
    model, input_indices, medians, means, stds, sequence_length = load_predictor()
    print("Clinical-direction sanity checks (ICULOS-free monotonic XGBoost)\n")

    scenarios = [
        ("1. Baseline normal vitals", BASELINE),
        ("2a. Lactate 4.0", {**BASELINE, "Lactate": 4.0}),
        ("2b. Lactate 8.0", {**BASELINE, "Lactate": 8.0}),
        ("3a. MAP 65", {**BASELINE, "MAP": 65.0}),
        ("3b. MAP 50", {**BASELINE, "MAP": 50.0}),
        ("4a. HR 130", {**BASELINE, "HR": 130.0}),
        ("4b. HR 160", {**BASELINE, "HR": 160.0}),
        ("5a. O2Sat 90", {**BASELINE, "O2Sat": 90.0}),
        ("5b. O2Sat 82", {**BASELINE, "O2Sat": 82.0}),
        (
            "6. All vitals clearly abnormal",
            {"HR": 150.0, "Temp": 39.5, "Resp": 32.0, "MAP": 55.0, "O2Sat": 88.0, "Lactate": 6.0},
        ),
    ]

    scores: dict[str, float] = {}
    for name, vitals in scenarios:
        score = predict_risk(vitals, model, input_indices, medians, means, stds, sequence_length)
        scores[name] = score
        print_case(name, vitals, score)

    print("Direction checks")
    results = [
        assess_monotonicity("Lactate 1.0 -> 4.0 -> 8.0", [scores["1. Baseline normal vitals"], scores["2a. Lactate 4.0"], scores["2b. Lactate 8.0"]]),
        assess_monotonicity("MAP 85 -> 65 -> 50", [scores["1. Baseline normal vitals"], scores["3a. MAP 65"], scores["3b. MAP 50"]]),
        assess_monotonicity("HR 75 -> 130 -> 160", [scores["1. Baseline normal vitals"], scores["4a. HR 130"], scores["4b. HR 160"]]),
        assess_monotonicity("O2Sat 98 -> 90 -> 82", [scores["1. Baseline normal vitals"], scores["5a. O2Sat 90"], scores["5b. O2Sat 82"]]),
    ]

    all_abnormal = scores["6. All vitals clearly abnormal"]
    other_maximum = max(score for name, score in scores.items() if name != "6. All vitals clearly abnormal")
    if all_abnormal > other_maximum:
        print(f"PASS - combined abnormal case is highest: {all_abnormal:.6f}")
    else:
        results.append(False)
        print(
            "FAIL - combined abnormal case is not the highest risk score: "
            f"{all_abnormal:.6f} versus another case at {other_maximum:.6f}"
        )

    if all(results):
        print("\nSanity-check result: no directionally counterintuitive decreases were found.")
    else:
        print(
            "\nSanity-check result: at least one clinically counterintuitive response was found. "
            "This can result from sparse positive examples and learned boosted-tree interactions, "
            "and the median-filled snapshot representation; it needs model/data refinement rather than hiding the result."
        )


if __name__ == "__main__":
    main()
