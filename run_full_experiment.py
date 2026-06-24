#!/usr/bin/env python
"""Run the full experiment: train, evaluate, and explain the credit risk model."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from config import RAW_DATA_DIR, RANDOM_SEED
from evaluate import evaluate_model
from explainability import HybridModelExplainer
from feature_engineering import engineer_features
from preprocessing import PreprocessingPipeline
from static_dataset import StaticDatasetBuilder
from temporal_dataset import TemporalDatasetBuilder
from train import split_arrays
from utils import ensure_directory, set_random_seed
from visualize_results import (  # Assuming this is for post-hoc visualization
    summarize_metrics,
    visualize_confusion_matrix,
    visualize_roc_auc,
    visualize_training_history,
)

RAW_COLUMN_RENAME_MAP = {
    "amount_ugx": "transaction_amount",
    "balance_after_ugx": "balance",
    "loan_amount_ugx": "loan_amount",
    "sacco_member": "sacco_membership",
    "location_type": "location",
    "network": "preferred_network",
    "channel": "preferred_channel",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


def load_best_model(model_dir: Path) -> tf.keras.Model:
    """Load the best model checkpoint saved during training."""
    model_path = model_dir / "best_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Best model checkpoint not found: {model_path}")
    LOGGER.info("Loading best model from %s", model_path)
    # We need to import F1Score from hybrid_model to load the model
    # with the custom metric.
    from hybrid_model import F1Score

    return tf.keras.models.load_model(
        model_path, custom_objects={"F1Score": F1Score}
    )


def load_final_model(model_dir: Path) -> tf.keras.Model:
    """Load the final model saved at the end of training."""
    model_path = model_dir / "trained_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Final trained model not found: {model_path}")
    LOGGER.info("Loading final model from %s", model_path)
    from hybrid_model import F1Score

    return tf.keras.models.load_model(model_path)


def main() -> None:
    set_random_seed(RANDOM_SEED)

    project_dir = Path(__file__).resolve().parent
    output_dir = ensure_directory(project_dir / "models" / "uganda_mobile_money")
    plots_dir = ensure_directory(output_dir / "plots")
    explain_dir = ensure_directory(output_dir / "explainability")

    raw_path = RAW_DATA_DIR / "Uganda_Mobile_Money_Logs_3000.xlsx"
    pipeline = PreprocessingPipeline(input_path=raw_path)
    raw = pipeline.load_data()
    raw = raw.rename(columns=RAW_COLUMN_RENAME_MAP)
    cleaned = pipeline.clean(raw)
    engineered = engineer_features(cleaned)

    temporal_dataset = TemporalDatasetBuilder().build(engineered)
    static_dataset = StaticDatasetBuilder().build(engineered)

    # Save feature names
    temporal_names_path = output_dir / "temporal_feature_names.txt"
    temporal_names_path.write_text("\n".join(temporal_dataset.feature_names), encoding="utf-8")
    static_names_path = output_dir / "static_feature_names.txt"
    static_names_path.write_text("\n".join(static_dataset.feature_names), encoding="utf-8")

    X_temp = temporal_dataset.X_temporal
    X_static = static_dataset.X_static
    labels = temporal_dataset.labels
    if labels is None:
        raise ValueError("Labels are required for training and evaluation.")

    train_set, val_set, test_set = split_arrays(X_temp, X_static, labels, random_seed=RANDOM_SEED)
    X_temp_train, X_static_train, y_train = train_set
    X_temp_val, X_static_val, y_val = val_set
    X_temp_test, X_static_test, y_test = test_set

    # The train_model function from train.py handles splitting, training,
    # and saving artifacts including the best model.
    LOGGER.info("Starting model training")
    from train import train_model
    artifacts = train_model(
        X_temporal=X_temp,
        X_static=X_static,
        labels=labels,
        output_dir=output_dir,
        epochs=100,
        batch_size=32,
        random_seed=RANDOM_SEED,
    )
    LOGGER.info("Model training completed")

    # Reload the best model based on `val_auc` for consistent evaluation
    model = load_best_model(output_dir)

    LOGGER.info("Evaluating on validation set")
    val_eval_dir = ensure_directory(output_dir / "evaluation" / "validation")
    val_artifacts = evaluate_model(
        model,
        X_temp_val,
        X_static_val,
        y_val,
        output_dir=val_eval_dir,
    )

    LOGGER.info("Evaluating on test set")
    test_eval_dir = ensure_directory(output_dir / "evaluation" / "test")
    test_artifacts = evaluate_model(
        model,
        X_temp_test,
        X_static_test,
        y_test,
        output_dir=test_eval_dir,
    )

    LOGGER.info("Generating SHAP explainability artifacts")
    explainer = HybridModelExplainer(
        model=model,
        temporal_feature_names=temporal_dataset.feature_names,
        static_feature_names=static_dataset.feature_names,
        output_dir=explain_dir, # Use the dedicated explainability directory
    )
    explain_artifacts = explainer.explain(
        X_temporal=X_temp_val,
        X_static=X_static_val,
        # Align borrower IDs with the validation set
        borrower_ids=val_set[2],
        background_size=100,
        explanation_size=50,
    )

    LOGGER.info("Generating final visualizations and summary")
    visualize_training_history(output_dir)
    visualize_roc_auc(output_dir, eval_type="validation")
    visualize_confusion_matrix(output_dir, eval_type="validation")
    visualize_roc_auc(output_dir, eval_type="test")
    visualize_confusion_matrix(output_dir, eval_type="test")
    summarize_metrics(output_dir)
    history_df = pd.read_csv(artifacts.training_history)
    history_plot_path = plot_training_history(history_df, output_dir)

    LOGGER.info("Full experiment completed")
    LOGGER.info("Training history plot: %s", plots_dir / "training_history_detailed.png")
    LOGGER.info("Validation plots in: %s", val_eval_dir)
    LOGGER.info("Test plots in: %s", test_eval_dir)
    LOGGER.info("=" * 80)
    LOGGER.info("ARTIFACTS SUMMARY")
    LOGGER.info("=" * 80)
    LOGGER.info("Training history plot: %s", history_plot_path)
    LOGGER.info("Validation evaluation artifacts in: %s", val_eval_dir)
    LOGGER.info("Test evaluation artifacts in: %s", test_eval_dir)
    LOGGER.info("  - Test metrics: %s", test_artifacts.metrics_json)
    LOGGER.info("  - Test ROC curve: %s", test_artifacts.roc_curve_plot)
    LOGGER.info("  - Test confusion matrix: %s", test_artifacts.confusion_matrix_plot)
    LOGGER.info("SHAP summary: %s", explainArtifacts.global_summary_plot)
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()
