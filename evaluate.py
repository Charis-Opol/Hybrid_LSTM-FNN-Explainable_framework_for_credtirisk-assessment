"""Model evaluation utilities for credit risk experiments."""

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
import seaborn as sns
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import MODELS_DIR, RANDOM_SEED
from utils import ensure_directory, set_random_seed


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationArtifacts:
    """Paths written by the evaluation pipeline."""

    metrics_json: Path
    classification_report_csv: Path
    confusion_matrix_plot: Path
    roc_curve_plot: Path
    precision_recall_curve_plot: Path


def compute_binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute standard binary classification metrics."""

    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
    }
    metrics["roc_auc"] = (
        float(roc_auc_score(y_true, probabilities))
        if len(np.unique(y_true)) > 1
        else 0.0
    )
    return metrics


def evaluate_model(
    model: tf.keras.Model,
    X_temporal_test: np.ndarray,
    X_static_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str | Path = MODELS_DIR / "plots",
) -> EvaluationArtifacts:
    """Evaluate a trained hybrid model and generate plots."""

    output_path = ensure_directory(Path(output_dir))
    probabilities = model.predict([X_temporal_test, X_static_test]).ravel()
    metrics = compute_binary_metrics(y_test, probabilities)
    predictions = (probabilities >= 0.5).astype(int)

    metrics_path = output_path / "evaluation.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report_path = output_path / "classification_report.csv"
    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(report_path)

    confusion_plot = plot_confusion_matrix(y_test, predictions, output_path)
    roc_plot = plot_roc_curve(y_test, probabilities, output_path)
    pr_plot = plot_precision_recall_curve(y_test, probabilities, output_path)

    LOGGER.info("Evaluation artifacts written to %s", output_path)
    return EvaluationArtifacts(
        metrics_json=metrics_path,
        classification_report_csv=report_path,
        confusion_matrix_plot=confusion_plot,
        roc_curve_plot=roc_plot,
        precision_recall_curve_plot=pr_plot,
    )


def compare_baselines(
    X_temporal_train: np.ndarray,
    X_static_train: np.ndarray,
    y_train: np.ndarray,
    X_temporal_test: np.ndarray,
    X_static_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str | Path = MODELS_DIR / "baseline_comparison",
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Compare Logistic Regression, Random Forest, and Vanilla LSTM baselines."""

    set_random_seed(random_seed)
    output_path = ensure_directory(Path(output_dir))
    results: list[dict[str, float | str]] = []

    tabular_train = np.concatenate(
        [X_static_train, X_temporal_train.reshape(X_temporal_train.shape[0], -1)],
        axis=1,
    )
    tabular_test = np.concatenate(
        [X_static_test, X_temporal_test.reshape(X_temporal_test.shape[0], -1)],
        axis=1,
    )

    baselines = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_seed,
            class_weight="balanced",
        ),
    }
    for name, estimator in baselines.items():
        estimator.fit(tabular_train, y_train)
        probabilities = estimator.predict_proba(tabular_test)[:, 1]
        results.append({"model": name, **compute_binary_metrics(y_test, probabilities)})

    lstm_model = build_vanilla_lstm(
        sequence_length=X_temporal_train.shape[1],
        temporal_features=X_temporal_train.shape[2],
    )
    lstm_model.fit(
        X_temporal_train,
        y_train,
        validation_split=0.15,
        epochs=50,
        batch_size=32,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=8,
                restore_best_weights=True,
            )
        ],
    )
    lstm_probabilities = lstm_model.predict(X_temporal_test).ravel()
    results.append(
        {"model": "vanilla_lstm", **compute_binary_metrics(y_test, lstm_probabilities)}
    )

    comparison = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    comparison.to_csv(output_path / "baseline_comparison.csv", index=False)
    return comparison


def build_vanilla_lstm(
    sequence_length: int,
    temporal_features: int,
) -> tf.keras.Model:
    """Build a simple LSTM baseline classifier."""

    inputs = tf.keras.Input(shape=(sequence_length, temporal_features))
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=output, name="vanilla_lstm")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"), "accuracy"],
    )
    return model


def plot_confusion_matrix(
    y_true: np.ndarray,
    predictions: np.ndarray,
    output_dir: Path,
) -> Path:
    """Save a confusion matrix heatmap."""

    path = output_dir / "confusion_matrix.png"
    matrix = confusion_matrix(y_true, predictions)
    fig, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=axis)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    output_dir: Path,
) -> Path:
    """Save ROC curve plot."""

    path = output_dir / "roc_curve.png"
    fig, axis = plt.subplots(figsize=(6, 5))
    if len(np.unique(y_true)) > 1:
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probabilities)
        axis.plot(false_positive_rate, true_positive_rate, label="Hybrid model")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_precision_recall_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    output_dir: Path,
) -> Path:
    """Save precision-recall curve plot."""

    path = output_dir / "precision_recall_curve.png"
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(recall, precision, label="Hybrid model")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_training_history(
    history_df: pd.DataFrame, output_dir: Path
) -> Path:
    """Visualize training loss and metrics over epochs."""
    plots_dir = ensure_directory(output_dir / "plots")
    output_path = plots_dir / "training_history_detailed.png"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training and Validation Metrics Over Epochs", fontsize=16, fontweight="bold")

    # Loss plot
    axes[0, 0].plot(history_df["loss"], label="Train Loss", linewidth=2)
    axes[0, 0].plot(history_df["val_loss"], label="Val Loss", linewidth=2)
    axes[0, 0].set_title("Model Loss", fontsize=12)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # AUC plot
    axes[0, 1].plot(history_df["auc"], label="Train AUC", linewidth=2)
    axes[0, 1].plot(history_df["val_auc"], label="Val AUC", linewidth=2)
    axes[0, 1].set_title("Area Under Curve (AUC)", fontsize=12)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("AUC")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Recall plot
    axes[1, 0].plot(history_df["recall"], label="Train Recall", linewidth=2)
    axes[1, 0].plot(history_df["val_recall"], label="Val Recall", linewidth=2)
    axes[1, 0].set_title("Recall", fontsize=12)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Recall")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Precision plot
    axes[1, 1].plot(history_df["precision"], label="Train Precision", linewidth=2)
    axes[1, 1].plot(history_df["val_precision"], label="Val Precision", linewidth=2)
    axes[1, 1].set_title("Precision", fontsize=12)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Precision")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(f"Saved training history visualization: {output_path}")
    return output_path


def evaluate() -> None:
    """CLI placeholder for notebook-driven evaluation."""

    raise RuntimeError(
        "Use evaluate_model(model, X_temporal_test, X_static_test, y_test)."
    )


if __name__ == "__main__":
    evaluate()
