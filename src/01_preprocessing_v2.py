"""Build normalized six-hour sepsis prediction sequences from PSV patient files.
Patient-level train/test split with statistics fit on train only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


SEQ_LEN = 6
RANDOM_STATE = 42
TEST_SIZE = 0.2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "data_v2.npz"
SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "split.json"
LABEL_COLUMN = "SepsisLabel"
NON_LAB_FEATURES = {
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
}


def load_and_clean_patient(path: Path, feature_names: list[str] | None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load one patient, impute values within that patient, and return arrays."""
    patient = pd.read_csv(path, sep="|")
    if patient.empty:
        raise ValueError("file has no ICU-hour rows")
    if patient.columns[-1] != LABEL_COLUMN:
        raise ValueError(
            f"expected {LABEL_COLUMN!r} as the last column, found {patient.columns[-1]!r}"
        )

    base_feature_names = patient.columns[:-1].tolist()
    raw_features = patient.loc[:, base_feature_names].apply(pd.to_numeric, errors="coerce")
    lab_feature_names = [name for name in base_feature_names if name not in NON_LAB_FEATURES]

    measurement_flags = raw_features.loc[:, lab_feature_names].notna().astype(np.float32)
    measurement_flags.columns = [f"{name}_was_measured" for name in lab_feature_names]

    clean_features = raw_features.ffill().bfill().fillna(0)
    augmented_features = pd.concat([clean_features, measurement_flags], axis=1)
    current_features = augmented_features.columns.tolist()
    if feature_names is not None and current_features != feature_names:
        raise ValueError("feature columns do not match the first patient file")

    labels = pd.to_numeric(patient[LABEL_COLUMN], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    features = augmented_features.to_numpy(dtype=np.float32)
    return features, labels, current_features


def make_windows(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return all fixed-length windows and their labels at each window's last hour."""
    if len(features) < SEQ_LEN:
        empty_X = np.empty((0, SEQ_LEN, features.shape[1]), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.int8)
        return empty_X, empty_y

    windows = np.lib.stride_tricks.sliding_window_view(features, SEQ_LEN, axis=0)
    windows = np.moveaxis(windows, -1, 1).copy()
    window_labels = labels[SEQ_LEN - 1 :]
    return windows, window_labels


def normalize(X: np.ndarray, means: np.ndarray | None = None, stds: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit z-score statistics on X and normalize it, or apply provided statistics."""
    flattened = X.reshape(-1, X.shape[-1])
    if means is None:
        means = flattened.mean(axis=0)
    if stds is None:
        stds = flattened.std(axis=0)
        stds[stds == 0] = 1.0
    return ((X - means) / stds).astype(np.float32), means.astype(np.float32), stds.astype(np.float32)


def main() -> None:
    patient_files = sorted(RAW_DIR.glob("*.psv"))
    if not patient_files:
        raise FileNotFoundError(
            f"No .psv files found in {RAW_DIR}. Add the sample patient files first."
        )

    print(f"Found {len(patient_files)} patient files. Building {SEQ_LEN}-hour windows...")
    all_windows: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_patient_ids: list[np.ndarray] = []
    feature_names: list[str] | None = None

    for index, path in enumerate(patient_files, start=1):
        try:
            features, labels, feature_names = load_and_clean_patient(path, feature_names)
            windows, window_labels = make_windows(features, labels)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            raise RuntimeError(f"Could not process {path.name}: {error}") from error

        all_windows.append(windows)
        all_labels.append(window_labels)
        all_patient_ids.append(np.full(len(windows), path.stem, dtype="U64"))
        print(
            f"[{index}/{len(patient_files)}] {path.name}: "
            f"{len(features)} hours -> {len(windows)} windows"
        )

    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    patient_ids = np.concatenate(all_patient_ids, axis=0)
    if len(X) == 0:
        raise ValueError(f"No {SEQ_LEN}-hour windows could be built from the patient files.")

    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    unique_patients = np.unique(patient_ids)
    print(f"Unique patients: {len(unique_patients)}")

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=patient_ids))

    train_patient_ids = patient_ids[train_idx]
    test_patient_ids = patient_ids[test_idx]
    unique_train_patients = np.unique(train_patient_ids)
    unique_test_patients = np.unique(test_patient_ids)

    print(f"Train patients: {len(unique_train_patients)}")
    print(f"Test patients: {len(unique_test_patients)}")

    overlap = set(unique_train_patients) & set(unique_test_patients)
    if overlap:
        raise ValueError(f"Patient leakage detected: {len(overlap)} patients in both splits")
    print("No patient leakage between train and test")

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    positive_train_patients = set()
    for pid in unique_train_patients:
        if y_train[train_patient_ids == pid].sum() > 0:
            positive_train_patients.add(pid)
    positive_test_patients = set()
    for pid in unique_test_patients:
        if y_test[test_patient_ids == pid].sum() > 0:
            positive_test_patients.add(pid)

    print(f"Train positive patients: {len(positive_train_patients)}/{len(unique_train_patients)}")
    print(f"Test positive patients: {len(positive_test_patients)}/{len(unique_test_patients)}")

    feature_medians = np.median(X_train.reshape(-1, X_train.shape[-1]), axis=0).astype(np.float32)
    X_train, feature_means, feature_stds = normalize(X_train)
    X_test, _, _ = normalize(X_test, feature_means, feature_stds)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        patient_ids_train=train_patient_ids,
        patient_ids_test=test_patient_ids,
        feature_names=np.asarray(feature_names),
        feature_medians=feature_medians,
        feature_means=feature_means,
        feature_stds=feature_stds,
    )
    print(f"Saved processed dataset to {OUTPUT_PATH}")

    split_info = {
        "train_patients": unique_train_patients.tolist(),
        "test_patients": unique_test_patients.tolist(),
        "train_patient_count": int(len(unique_train_patients)),
        "test_patient_count": int(len(unique_test_patients)),
        "train_positive_patient_count": int(len(positive_train_patients)),
        "test_positive_patient_count": int(len(positive_test_patients)),
        "train_window_count": int(len(X_train)),
        "test_window_count": int(len(X_test)),
    }
    with open(SPLIT_PATH, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"Saved split info to {SPLIT_PATH}")


if __name__ == "__main__":
    main()
