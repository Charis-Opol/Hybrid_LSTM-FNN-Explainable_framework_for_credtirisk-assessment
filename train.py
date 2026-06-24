"""Training pipeline for the hybrid credit risk model."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".matplotlib"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from config import MODELS_DIR, RANDOM_SEED
from evaluate import compute_binary_metrics
from hybrid_model import build_hybrid_model
from utils import ensure_directory, set_random_seed


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainArtifacts:
    """Paths written by the training pipeline."""

    trained_model: Path
    best_checkpoint: Path
    training_history: Path
    evaluation: Path
    history_plot: Path
    tensorboard_log_dir: Path


def split_arrays(
    X_temporal: np.ndarray,
    X_static: np.ndarray,
    labels: np.ndarray,
    random_seed: int = RANDOM_SEED,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Split arrays into 70/15/15 train, validation, and test partitions."""

    stratify = labels if len(np.unique(labels)) > 1 else None
    X_temp_train, X_temp_holdout, X_static_train, X_static_holdout, y_train, y_holdout = (
        train_test_split(
            X_temporal,
            X_static,
            labels,
            test_size=0.30,
            random_state=random_seed,
            stratify=stratify,
        )
    )
    holdout_stratify = y_holdout if len(np.unique(y_holdout)) > 1 else None
    X_temp_val, X_temp_test, X_static_val, X_static_test, y_val, y_test = (
        train_test_split(
            X_temp_holdout,
            X_static_holdout,
            y_holdout,
            test_size=0.50,
            random_state=random_seed,
            stratify=holdout_stratify,
        )
    )
    return (
        (X_temp_train, X_static_train, y_train),
        (X_temp_val, X_static_val, y_val),
        (X_temp_test, X_static_test, y_test),
    )


def train_model(
    X_temporal: np.ndarray,
    X_static: np.ndarray,
    labels: np.ndarray,
    output_dir: str | Path = MODELS_DIR,
    epochs: int = 100,
    batch_size: int = 32,
    random_seed: int = RANDOM_SEED,
) -> TrainArtifacts:
    """Train the hybrid model and save experiment artifacts.

    Args:
        X_temporal: Temporal input with shape ``(n, 12, temporal_features)``.
        X_static: Static input with shape ``(n, static_features)``.
        labels: Binary target labels.
        output_dir: Directory where model artifacts are saved.
        epochs: Maximum training epochs.
        batch_size: Batch size.
        random_seed: Reproducibility seed.

    Returns:
        Paths to saved training artifacts.
    """

    set_random_seed(random_seed)
    output_path = ensure_directory(Path(output_dir))
    plots_dir = ensure_directory(output_path / "plots")
    tensorboard_dir = ensure_directory(output_path / "tensorboard")

    labels = labels.astype(float)
    train_set, validation_set, test_set = split_arrays(
        X_temporal,
        X_static,
        labels,
        random_seed=random_seed,
    )
    X_temp_train, X_static_train, y_train = train_set
    X_temp_val, X_static_val, y_val = validation_set
    X_temp_test, X_static_test, y_test = test_set

    model = build_hybrid_model(
        sequence_length=X_temporal.shape[1],
        temporal_features=X_temporal.shape[2],
        static_features=X_static.shape[1],
    )

    checkpoint_path = output_path / "best_model.keras"
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train.astype(int)),
        y=y_train.astype(int),
    )
    class_weight = {
        int(cls): float(weight)
        for cls, weight in zip(np.unique(y_train.astype(int)), class_weights)
    }
    LOGGER.info("Using class weights: %s", class_weight)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=8,
            mode="max",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(tensorboard_dir)),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        ),
    ]

    LOGGER.info("Starting model training")
    history = model.fit(
        [X_temp_train, X_static_train],
        y_train,
        validation_data=([X_temp_val, X_static_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    trained_model_path = output_path / "trained_model.keras"
    model.save(trained_model_path)

    history_path = output_path / "training_history.csv"
    history_frame = pd.DataFrame(history.history)
    history_frame.to_csv(history_path, index=False)

    history_plot_path = plots_dir / "training_history.png"
    plot_training_history(history_frame, history_plot_path)

    probabilities = model.predict([X_temp_test, X_static_test]).ravel()
    metrics = compute_binary_metrics(y_test, probabilities)
    evaluation_path = output_path / "evaluation.json"
    evaluation_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    LOGGER.info("Training complete. Artifacts written to %s", output_path)
    return TrainArtifacts(
        trained_model=trained_model_path,
        best_checkpoint=checkpoint_path,
        training_history=history_path,
        evaluation=evaluation_path,
        history_plot=history_plot_path,
        tensorboard_log_dir=tensorboard_dir,
    )


def plot_training_history(history: pd.DataFrame, output_path: str | Path) -> Path:
    """Save training history curves for loss and common metrics."""

    output_path = Path(output_path)
    metrics = [column for column in ["loss", "auc", "precision", "recall", "f1"] if column in history]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 3 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]

    for axis, metric in zip(axes, metrics):
        axis.plot(history[metric], label=metric)
        validation_metric = f"val_{metric}"
        if validation_metric in history:
            axis.plot(history[validation_metric], label=validation_metric)
        axis.set_title(metric)
        axis.set_xlabel("Epoch")
        axis.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def train() -> None:
    """CLI placeholder for notebook-driven training.

    This framework expects prepared ``X_temporal``, ``X_static``, and labels
    from the dataset builders. Import ``train_model`` from this module in an
    experiment script or notebook and pass those arrays directly.
    """

    raise RuntimeError(
        "Use train_model(X_temporal, X_static, labels) after building datasets."
    )


if __name__ == "__main__":
    train()
