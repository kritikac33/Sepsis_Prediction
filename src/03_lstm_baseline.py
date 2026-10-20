"""Train a compact LSTM baseline on six-hour sepsis prediction sequences."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib"))

import matplotlib.pyplot as plt
import tensorflow as tf


RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20  # Portion of the training split held out for early stopping.
MAX_EPOCHS = 20
BATCH_SIZE = 128
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "data.npz"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "lstm_baseline.keras"
HISTORY_PATH = PROJECT_ROOT / "outputs" / "plots" / "lstm_history.png"


def build_model(sequence_shape: tuple[int, int]) -> tf.keras.Sequential:
    """Build the intentionally small prototype LSTM architecture."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=sequence_shape),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="lstm_baseline",
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model


def plot_history(history: tf.keras.callbacks.History) -> None:
    """Save loss and AUC curves for the training and validation splits."""
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history.history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history.history["loss"], marker="o", label="Train")
    axes[0].plot(epochs, history.history["val_loss"], marker="o", label="Validation")
    axes[0].set(title="Binary cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()

    axes[1].plot(epochs, history.history["auc"], marker="o", label="Train")
    axes[1].plot(epochs, history.history["val_auc"], marker="o", label="Validation")
    axes[1].set(title="Area under ROC curve", xlabel="Epoch", ylabel="AUC", ylim=(0, 1))
    axes[1].legend()

    figure.suptitle("LSTM baseline training history")
    figure.tight_layout()
    figure.savefig(HISTORY_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved training history plot to {HISTORY_PATH}")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {DATA_PATH}. Run src/01_preprocessing.py first."
        )

    tf.keras.utils.set_random_seed(RANDOM_STATE)
    with np.load(DATA_PATH, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int32)

    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (samples, hours, features), got {X.shape}")
    if len(X) != len(y):
        raise ValueError(f"X and y have different sample counts: {len(X)} and {len(y)}")
    if len(np.unique(y)) != 2:
        raise ValueError("Both sepsis and non-sepsis samples are required for a stratified split.")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )
    print(f"Loaded sequences: X={X.shape}, y={y.shape}")
    print(f"Train/validation/test: {len(X_train):,}/{len(X_validation):,}/{len(X_test):,}")

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = {int(label): float(weight) for label, weight in zip(classes, weights)}
    print(f"Class weights: {class_weights}")

    model = build_model(sequence_shape=(X.shape[1], X.shape[2]))
    model.summary()
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=3, restore_best_weights=True, verbose=1
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=2,
    )

    probabilities = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(np.int32)
    test_auc = roc_auc_score(y_test, probabilities)
    test_f1 = f1_score(y_test, predictions, zero_division=0)
    print(f"\nTest AUC: {test_auc:.4f}")
    print(f"Test F1: {test_f1:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")
    plot_history(history)


if __name__ == "__main__":
    main()
