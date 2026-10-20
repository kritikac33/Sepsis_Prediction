"""Cluster septic-patient six-hour profiles into data-driven subtypes.

This is an exploratory patient subtyping module, separate from the detection
pipeline. It uses one onset-window profile per patient: the first six-hour
window whose final-hour SepsisLabel is positive.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib"))

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "data.npz"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "kmeans_subtypes.pkl"
PLOT_PATH = PROJECT_ROOT / "outputs" / "plots" / "subtype_clusters.png"
RANDOM_STATE = 42
K_VALUES = range(2, 7)
MIN_CLUSTER_SIZE = 10
NON_CLINICAL_FEATURES = {"Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"}


def summarize_window(
    window: np.ndarray,
    measurement_mask: np.ndarray,
    measured_feature_mask: np.ndarray,
    fallback_values: np.ndarray,
) -> np.ndarray:
    """Summarize one clinical window without treating unmeasured labs as zero."""
    summaries: list[np.ndarray] = []
    for statistic in (np.mean, np.min, np.max):
        values: list[float] = []
        for index in range(window.shape[1]):
            observed = window[:, index]
            if measured_feature_mask[index]:
                observed = observed[measurement_mask[:, index]]
            if not len(observed):
                observed = np.asarray([fallback_values[index]])
            values.append(float(statistic(observed)))
        summaries.append(np.asarray(values, dtype=np.float32))
    return np.concatenate(summaries)


def summary_names(feature_names: list[str]) -> list[str]:
    return (
        [f"{name} (mean)" for name in feature_names]
        + [f"{name} (min)" for name in feature_names]
        + [f"{name} (max)" for name in feature_names]
    )


def describe_feature(name: str, value: float, overall: float) -> str:
    """Give a concise, presentation-friendly description of a cluster centroid."""
    base, statistic = name.rsplit(" (", 1)
    higher = value > overall
    direction = "higher" if higher else "lower"
    clinical_hints = {
        "HR": "tachycardia burden" if higher else "lower heart-rate burden",
        "Temp": "higher temperature burden" if higher else "lower temperature burden",
        "Resp": "respiratory-rate burden" if higher else "lower respiratory-rate burden",
        "MAP": "hemodynamic pressure" if higher else "lower arterial pressure",
        "O2Sat": "oxygen saturation" if higher else "reduced oxygen saturation",
        "Lactate": "lactate burden" if higher else "lower lactate burden",
    }
    concept = clinical_hints.get(base, f"{base} level")
    if concept.startswith("lower ") and not higher:
        return f"{concept.capitalize()} ({statistic.rstrip(')')}) than other subtype profiles."
    return f"{direction.capitalize()} {concept} ({statistic.rstrip(')')}) than other subtype profiles."


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Processed data missing. Run src/01_preprocessing.py first.")

    with np.load(DATA_PATH, allow_pickle=False) as data:
        required = {"X", "y", "patient_ids", "feature_names", "feature_means", "feature_stds"}
        missing = required.difference(data.files)
        if missing:
            raise RuntimeError(
                "Processed data lacks patient IDs/statistics. Re-run src/01_preprocessing.py. "
                f"Missing: {', '.join(sorted(missing))}"
            )
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int32)
        patient_ids = data["patient_ids"].astype(str)
        feature_names = data["feature_names"].astype(str).tolist()
        means = data["feature_means"].astype(np.float32)
        stds = np.where(data["feature_stds"] == 0, 1.0, data["feature_stds"]).astype(np.float32)

    # Subtyping uses physiological/laboratory columns, not demographics,
    # administrative fields, ICULOS, or the added missingness indicators.
    clinical_feature_names = [
        name for name in feature_names
        if not name.endswith("_was_measured") and name not in NON_CLINICAL_FEATURES
    ]
    clinical_indices = [feature_names.index(name) for name in clinical_feature_names]
    flag_indices = {
        name: feature_names.index(f"{name}_was_measured")
        for name in clinical_feature_names
        if f"{name}_was_measured" in feature_names
    }
    raw_all = X * stds + means
    raw_clinical = raw_all[:, :, clinical_indices]
    measured_feature_mask = np.asarray([name in flag_indices for name in clinical_feature_names])
    measurement_masks = np.ones_like(raw_clinical, dtype=bool)
    fallback_values = np.zeros(len(clinical_feature_names), dtype=np.float32)
    for index, name in enumerate(clinical_feature_names):
        if name not in flag_indices:
            fallback_values[index] = float(np.median(raw_clinical[:, :, index]))
            continue
        mask = raw_all[:, :, flag_indices[name]] > 0.5
        measurement_masks[:, :, index] = mask
        measured_values = raw_clinical[:, :, index][mask]
        fallback_values[index] = float(np.median(measured_values)) if len(measured_values) else 0.0

    positive_window_indices = np.flatnonzero(y == 1)
    if not len(positive_window_indices):
        raise ValueError("No SepsisLabel=1 windows found for subtyping.")
    # The first positive window is the earliest available positive profile per patient.
    _, first_positions = np.unique(patient_ids[positive_window_indices], return_index=True)
    onset_indices = positive_window_indices[np.sort(first_positions)]
    onset_patient_ids = patient_ids[onset_indices]
    onset_windows = raw_clinical[onset_indices]
    onset_masks = measurement_masks[onset_indices]
    summaries = np.vstack([
        summarize_window(window, mask, measured_feature_mask, fallback_values)
        for window, mask in zip(onset_windows, onset_masks)
    ])
    names = summary_names(clinical_feature_names)
    if len(summaries) < 3:
        raise ValueError("At least three septic patients are needed for silhouette-based clustering.")

    # Cap only extreme tails before scaling so one implausibly influential
    # laboratory value cannot become its own "subtype". The fitted scaler is
    # still StandardScaler, as used by the clustering pipeline.
    lower_bounds = np.percentile(summaries, 5, axis=0)
    upper_bounds = np.percentile(summaries, 95, axis=0)
    model_summaries = np.clip(summaries, lower_bounds, upper_bounds)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(model_summaries)
    candidate_scores: dict[int, float] = {}
    candidate_models: dict[int, KMeans] = {}
    for k in K_VALUES:
        if k >= len(summaries):
            continue
        model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = model.fit_predict(scaled)
        sizes = np.bincount(labels, minlength=k)
        if sizes.min() < MIN_CLUSTER_SIZE:
            print(f"k={k}: rejected (cluster sizes {sizes.tolist()}; minimum is {MIN_CLUSTER_SIZE})")
            continue
        candidate_models[k] = model
        candidate_scores[k] = float(silhouette_score(scaled, labels))
        print(f"k={k}: silhouette={candidate_scores[k]:.4f}; sizes={sizes.tolist()}")
    if not candidate_scores:
        raise ValueError("Not enough septic patients to evaluate k=2 through k=6.")

    best_k = max(candidate_scores, key=candidate_scores.get)
    kmeans = candidate_models[best_k]
    labels = kmeans.labels_
    best_silhouette = candidate_scores[best_k]
    print(f"\nChosen k={best_k}; silhouette score={best_silhouette:.4f}")
    print("Cluster sizes:", {int(cluster): int((labels == cluster).sum()) for cluster in range(best_k)})

    clinical_dimensions = np.ones(len(names), dtype=bool)
    cluster_records: list[dict[str, object]] = []
    overall_means = model_summaries.mean(axis=0)
    overall_stds = np.where(model_summaries.std(axis=0) == 0, 1.0, model_summaries.std(axis=0))
    for cluster in range(best_k):
        members = labels == cluster
        cluster_mean = model_summaries[members].mean(axis=0)
        other_mean = model_summaries[~members].mean(axis=0) if (~members).any() else overall_means
        effect = np.abs((cluster_mean - other_mean) / overall_stds)
        effect[~clinical_dimensions] = -np.inf
        # Present five distinct clinical variables, rather than repeating one
        # lab three times merely because its mean/min/max all differ.
        top_indices: list[int] = []
        used_bases: set[str] = set()
        for index in np.argsort(effect)[::-1]:
            base = names[index].split(" (")[0]
            if base not in used_bases:
                top_indices.append(int(index))
                used_bases.add(base)
            if len(top_indices) == 5:
                break
        top_features = [
            {
                "feature": names[index],
                "mean_value": float(cluster_mean[index]),
                "description": describe_feature(names[index], float(cluster_mean[index]), float(other_mean[index])),
            }
            for index in top_indices
        ]
        record = {
            "cluster_id": int(cluster),
            "size": int(members.sum()),
            "top_features": top_features,
            "example_patient_ids": onset_patient_ids[members][:5].tolist(),
        }
        cluster_records.append(record)
        print(f"\nCluster {cluster} (n={record['size']})")
        for feature in top_features:
            print(f"  {feature['feature']}: {feature['mean_value']:.3f} — {feature['description']}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "kmeans": kmeans,
            "scaler": scaler,
            "n_clusters": best_k,
            "silhouette_score": best_silhouette,
            "summary_feature_names": names,
            "cluster_records": cluster_records,
            "patient_ids": onset_patient_ids,
            "cluster_labels": labels,
        },
        MODEL_PATH,
    )
    print(f"Saved subtype model to {MODEL_PATH}")

    coordinates = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(scaled)
    figure, axis = plt.subplots(figsize=(8, 6.5))
    scatter = axis.scatter(coordinates[:, 0], coordinates[:, 1], c=labels, cmap="tab10", s=62, alpha=0.85, edgecolor="white", linewidth=0.7)
    axis.set(title=f"Septic patient subtypes (k={best_k}, silhouette={best_silhouette:.3f})", xlabel="PCA component 1", ylabel="PCA component 2")
    axis.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    figure.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(PLOT_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved cluster visualization to {PLOT_PATH}")


if __name__ == "__main__":
    main()
