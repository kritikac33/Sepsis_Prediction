"""Build normalized six-hour sepsis prediction sequences from PSV patient files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# Change this value to build longer or shorter ICU-hour sequences.
SEQ_LEN = 6

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "data.npz"
LABEL_COLUMN = "SepsisLabel"
# Physiological measurements and administrative columns are not laboratory tests.
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

    # Capture whether a lab was observed before any within-patient imputation.
    measurement_flags = raw_features.loc[:, lab_feature_names].notna().astype(np.float32)
    measurement_flags.columns = [f"{name}_was_measured" for name in lab_feature_names]

    # Impute feature values only within this patient; labels are never propagated.
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

    # sliding_window_view places the window axis last, so move it to axis 1.
    windows = np.lib.stride_tricks.sliding_window_view(features, SEQ_LEN, axis=0)
    windows = np.moveaxis(windows, -1, 1).copy()
    window_labels = labels[SEQ_LEN - 1 :]
    return windows, window_labels


def normalize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit z-score statistics on the complete sequence dataset and normalize it."""
    flattened = X.reshape(-1, X.shape[-1])
    means = flattened.mean(axis=0)
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

    # Retain raw feature medians and z-score statistics for consistent API inference.
    feature_medians = np.median(X.reshape(-1, X.shape[-1]), axis=0).astype(np.float32)
    X, feature_means, feature_stds = normalize(X)
    positive_count = int(y.sum())
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(
        f"Class balance: {positive_count}/{len(y)} positive "
        f"({positive_count / len(y):.2%}), {len(y) - positive_count}/{len(y)} negative"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        X=X,
        y=y,
        patient_ids=patient_ids,
        feature_names=np.asarray(feature_names),
        feature_medians=feature_medians,
        feature_means=feature_means,
        feature_stds=feature_stds,
    )
    print(f"Saved processed dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
