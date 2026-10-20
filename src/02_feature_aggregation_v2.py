"""Aggregate 6-timestep windows to per-variable summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "data_v2.npz"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "features_v2.npz"


def compute_slope(values: np.ndarray) -> float:
    """Compute linear slope of values over time."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    try:
        slope = np.polyfit(x, values, 1)[0]
        return float(slope)
    except:
        return 0.0


def aggregate_window(window: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    """Aggregate 6 timesteps x features to per-variable summaries."""
    aggregated = {}
    n_timesteps, n_features = window.shape

    for i, name in enumerate(feature_names):
        values = window[:, i]

        if name.endswith("_was_measured"):
            aggregated[f"{name}_frac"] = float(values.mean())
        else:
            aggregated[f"{name}_last"] = float(values[-1])
            aggregated[f"{name}_mean"] = float(values.mean())
            aggregated[f"{name}_min"] = float(values.min())
            aggregated[f"{name}_max"] = float(values.max())
            aggregated[f"{name}_slope"] = compute_slope(values)

    return aggregated


def main() -> None:
    data = np.load(INPUT_PATH, allow_pickle=True)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    patient_ids_train = data["patient_ids_train"]
    patient_ids_test = data["patient_ids_test"]
    feature_names = data["feature_names"].tolist()

    print(f"Train windows: {len(X_train)}, shape: {X_train.shape}")
    print(f"Test windows: {len(X_test)}, shape: {X_test.shape}")
    print(f"Base features: {len(feature_names)}")

    aggregated_train = []
    for window in X_train:
        aggregated_train.append(aggregate_window(window, feature_names))
    df_train = pd.DataFrame(aggregated_train)

    aggregated_test = []
    for window in X_test:
        aggregated_test.append(aggregate_window(window, feature_names))
    df_test = pd.DataFrame(aggregated_test)

    X_train_agg = df_train.to_numpy(dtype=np.float32)
    X_test_agg = df_test.to_numpy(dtype=np.float32)
    new_feature_names = df_train.columns.tolist()

    print(f"Aggregated train shape: {X_train_agg.shape}")
    print(f"Aggregated test shape: {X_test_agg.shape}")
    print(f"New feature count: {len(new_feature_names)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        X_train=X_train_agg,
        X_test=X_test_agg,
        y_train=y_train,
        y_test=y_test,
        patient_ids_train=patient_ids_train,
        patient_ids_test=patient_ids_test,
        feature_names=np.asarray(new_feature_names),
    )
    print(f"Saved aggregated features to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
