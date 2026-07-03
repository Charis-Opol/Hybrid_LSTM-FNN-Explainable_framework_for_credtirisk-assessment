#!/usr/bin/env python
"""Full experiment on Uganda Mobile Money 15,000-row dataset.

Trains and cross-validates four models for comparison:
  1. Hybrid LSTM-GRU-Attention-FNN (the main model)
  2. XGBoost baseline
  3. Vanilla LSTM baseline (temporal-only, no attention/static branch)
  4. Logistic regression baseline

All four are evaluated with stratified 5-fold CV using pooled
out-of-fold predictions, since the positive class is very small (~2%,
~70 positive borrowers total) and a single train/val/test split does
not give stable metrics. A single-split train/evaluate run of the
hybrid model is also kept for continuity with earlier experiments.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import RANDOM_SEED
from evaluate import evaluate_model, plot_training_history as plot_detailed_history
from feature_engineering import FeatureColumns, engineer_features
from hybrid_model import F1Score
from Kfold_cv_hybrid import run_kfold_cv
from kfold_cv_vanilla_lstm import run_kfold_cv_vanilla_lstm
from logistic_regression_baseline import run_logistic_regression_baseline
from static_dataset import StaticDatasetBuilder
from temporal_dataset import TemporalDatasetBuilder
from train import split_arrays, train_model
from utils import ensure_directory, set_random_seed
from visualize_model_comparison import compare_all_models
from visualize_results import (
    summarize_metrics,
    visualize_confusion_matrix,
    visualize_roc_auc,
    visualize_training_history,
    visualize_validation_metrics,
)
from XGBoost_baseline import run_xgboost_baseline

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "uganda_mobile_money_master.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "models"
    / "uganda_mobile_money_708000_8_no_SHAP"
)
RESULT_IMAGES_DIR = OUTPUT_DIR / "training_result_images"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


def load_and_normalize_dataset(path: Path) -> pd.DataFrame:
    LOGGER.info("Reading CSV dataset...")

    data = pd.read_csv(path)

    rename_map = {
        "timestamp": "transaction_date",
        "amount": "transaction_amount",
        "balance_after": "balance",
        "default_label": "defaulted",
        "loan_amount": "loan_amount",
        "sacco_member": "sacco_membership",
        "network": "preferred_network",
        "channel": "preferred_channel",
    }

    existing = {
        k: v
        for k, v in rename_map.items()
        if k in data.columns
    }

    data = data.rename(columns=existing)
    data["transaction_date"] = pd.to_datetime(data["transaction_date"], format="ISO8601")

    return data


def build_training_arrays(data: pd.DataFrame):
    """Engineer features and build aligned temporal/static arrays."""

    feature_columns = FeatureColumns(
        amount="transaction_amount",
        balance="balance",
        target="defaulted"
    )
    engineered = engineer_features(data, columns=feature_columns)

    temporal = TemporalDatasetBuilder().build(engineered)
    static = StaticDatasetBuilder().build(engineered)

    static_by_borrower = {
        borrower_id: row_index
        for row_index, borrower_id in enumerate(static.borrower_ids.tolist())
    }
    static_indices = [
        static_by_borrower[borrower_id]
        for borrower_id in temporal.borrower_ids.tolist()
    ]
    X_static_aligned = static.X_static[static_indices]

    if temporal.labels is None:
        raise ValueError("The dataset must include default labels for training.")

    return (
        temporal.X_temporal,
        X_static_aligned,
        temporal.labels,
        engineered,
        temporal.feature_names,
        static.feature_names,
        temporal.borrower_ids,
    )


def copy_evaluation_images_to_results_dir(output_dir: Path, results_dir: Path) -> None:
    """Copy all evaluation, training, and comparison plots into training_result_images."""

    ensure_directory(results_dir)
    image_sources: list[tuple[Path, str]] = []

    plots_dir = output_dir / "plots"
    if plots_dir.exists():
        for image_path in plots_dir.glob("*.png"):
            image_sources.append((image_path, image_path.name))

    for eval_split in ("validation", "test"):
        eval_dir = output_dir / "evaluation" / eval_split
        if not eval_dir.exists():
            continue
        for image_path in eval_dir.glob("*.png"):
            image_sources.append((image_path, f"{eval_split}_{image_path.name}"))

    comparison_plot = output_dir / "model_comparison" / "model_comparison.png"
    if comparison_plot.exists():
        image_sources.append((comparison_plot, "model_comparison.png"))

    for source, dest_name in image_sources:
        shutil.copy2(source, results_dir / dest_name)
        LOGGER.info("Copied %s -> %s", source.name, results_dir / dest_name)


def write_experiment_summary(
    output_dir: Path,
    results_dir: Path,
    val_metrics: dict,
    test_metrics: dict,
    model_summaries: dict[str, dict],
) -> Path:
    """Write a text summary of the experiment, including CV results for all models."""

    summary_path = results_dir / "EXPERIMENT_SUMMARY.txt"
    lines = [
        "Uganda Mobile Money Credit Risk Experiment (708,000 transactions)",
        "=" * 60,
        f"Data source: {DATA_PATH.name}",
        f"Model output: {output_dir}",
        "",
        "Hybrid model - single-split validation metrics:",
    ]
    for key, value in val_metrics.items():
        lines.append(f"  {key}: {value:.4f}")
    lines.append("")
    lines.append("Hybrid model - single-split test metrics:")
    for key, value in test_metrics.items():
        lines.append(f"  {key}: {value:.4f}")

    for model_name, metrics in model_summaries.items():
        lines.append("")
        lines.append(f"{model_name} - pooled out-of-fold metrics (5-fold CV):")
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def main() -> None:
    set_random_seed(RANDOM_SEED)
    ensure_directory(OUTPUT_DIR)
    ensure_directory(RESULT_IMAGES_DIR)

    LOGGER.info("Loading dataset from %s", DATA_PATH)
    data = load_and_normalize_dataset(DATA_PATH)
    data = data.sort_values(["borrower_id", "transaction_date"]).reset_index(drop=True)
    LOGGER.info("Loaded %s transactions", f"{len(data):,}")
    LOGGER.info("Borrowers: %s", f"{data['borrower_id'].nunique():,}")

    (
        X_temporal,
        X_static,
        labels,
        engineered,
        temporal_names,
        static_names,
        borrower_ids,
    ) = build_training_arrays(data)

    LOGGER.info(
        "Built datasets: %d borrowers, %d temporal features/month, %d static features",
        len(labels), len(temporal_names), len(static_names),
    )

    # --- Diagnostics: overall label distribution ---
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    LOGGER.info("Overall label distribution: %s", dict(zip(unique_labels.tolist(), label_counts.tolist())))
    LOGGER.info("Total borrowers: %d, positive rate: %.4f", len(labels), labels.mean())

    engineered.to_csv(OUTPUT_DIR / "engineered_features.csv", index=False)
    (OUTPUT_DIR / "temporal_feature_names.txt").write_text("\n".join(temporal_names), encoding="utf-8")
    (OUTPUT_DIR / "static_feature_names.txt").write_text("\n".join(static_names), encoding="utf-8")

    # --- Stratified k-fold CV for all four models ---
    LOGGER.info("Running 5-fold CV for hybrid model")
    hybrid_cv_summary = run_kfold_cv(
        X_temporal=X_temporal, X_static=X_static, labels=labels,
        output_dir=OUTPUT_DIR / "kfold_hybrid", n_splits=5,
    )

    LOGGER.info("Running 5-fold CV for XGBoost baseline")
    xgb_summary = run_xgboost_baseline(
        X_temporal=X_temporal, X_static=X_static, labels=labels,
        temporal_feature_names=temporal_names, static_feature_names=static_names,
        output_dir=OUTPUT_DIR / "xgboost_baseline", n_splits=5,
    )

    LOGGER.info("Running 5-fold CV for vanilla LSTM baseline")
    vanilla_lstm_summary = run_kfold_cv_vanilla_lstm(
        X_temporal=X_temporal, labels=labels,
        output_dir=OUTPUT_DIR / "kfold_vanilla_lstm", n_splits=5,
    )

    LOGGER.info("Running 5-fold CV for logistic regression baseline")
    logreg_summary = run_logistic_regression_baseline(
        X_temporal=X_temporal, X_static=X_static, labels=labels,
        temporal_feature_names=temporal_names, static_feature_names=static_names,
        output_dir=OUTPUT_DIR / "logistic_regression_baseline", n_splits=5,
    )

    model_summaries = {
        "Hybrid LSTM-GRU-Attention-FNN": hybrid_cv_summary["pooled_metrics"],
        "XGBoost": xgb_summary["pooled_metrics"],
        "Vanilla LSTM": vanilla_lstm_summary["pooled_metrics"],
        "Logistic Regression": logreg_summary["pooled_metrics"],
    }

    LOGGER.info("Building model comparison chart and table")
    comparison_dir = OUTPUT_DIR / "model_comparison"
    chart_path, table_path = compare_all_models(
        model_result_dirs={
            "Hybrid LSTM-GRU-Attention-FNN": OUTPUT_DIR / "kfold_hybrid",
            "XGBoost": OUTPUT_DIR / "xgboost_baseline",
            "Vanilla LSTM": OUTPUT_DIR / "kfold_vanilla_lstm",
            "Logistic Regression": OUTPUT_DIR / "logistic_regression_baseline",
        },
        output_dir=comparison_dir,
    )
    LOGGER.info("Comparison chart: %s", chart_path)
    LOGGER.info("Comparison table: %s", table_path)

    # --- Single-split training/evaluation for the hybrid model (kept for continuity) ---
    LOGGER.info("Starting single-split model training (epochs=100, batch_size=128)")
    artifacts = train_model(
        X_temporal=X_temporal, X_static=X_static, labels=labels,
        output_dir=OUTPUT_DIR, epochs=100, batch_size=128, random_seed=RANDOM_SEED,
    )
    LOGGER.info("Training complete")

    _, val_set, test_set = split_arrays(X_temporal, X_static, labels, random_seed=RANDOM_SEED)
    X_temp_val, X_static_val, y_val = val_set
    X_temp_test, X_static_test, y_test = test_set

    LOGGER.info("Val positives: %d / %d (%.2f%%)", int(y_val.sum()), len(y_val), 100 * y_val.mean())
    LOGGER.info("Test positives: %d / %d (%.2f%%)", int(y_test.sum()), len(y_test), 100 * y_test.mean())

    model_path = OUTPUT_DIR / "best_model.keras"
    model = tf.keras.models.load_model(model_path, custom_objects={"F1Score": F1Score})

    LOGGER.info("Evaluating on validation and test sets")
    val_eval_dir = ensure_directory(OUTPUT_DIR / "evaluation" / "validation")
    test_eval_dir = ensure_directory(OUTPUT_DIR / "evaluation" / "test")
    val_artifacts, test_artifacts = evaluate_model(
        model,
        X_temp_val, X_static_val, y_val,
        X_temp_test, X_static_test, y_test,
        val_output_dir=val_eval_dir,
        test_output_dir=test_eval_dir,
    )

    with open(val_artifacts.metrics_json, encoding="utf-8") as handle:
        val_metrics = json.load(handle)
    with open(test_artifacts.metrics_json, encoding="utf-8") as handle:
        test_metrics = json.load(handle)

    LOGGER.info("Generating evaluation visualizations")
    visualize_training_history(OUTPUT_DIR)
    visualize_validation_metrics(OUTPUT_DIR)
    visualize_roc_auc(OUTPUT_DIR, eval_type="validation")
    visualize_confusion_matrix(OUTPUT_DIR, eval_type="validation")
    visualize_roc_auc(OUTPUT_DIR, eval_type="test")
    visualize_confusion_matrix(OUTPUT_DIR, eval_type="test")
    summarize_metrics(OUTPUT_DIR)

    history_df = pd.read_csv(artifacts.training_history)
    plot_detailed_history(history_df, OUTPUT_DIR)

    copy_evaluation_images_to_results_dir(OUTPUT_DIR, RESULT_IMAGES_DIR)
    summary_path = write_experiment_summary(
        OUTPUT_DIR, RESULT_IMAGES_DIR, val_metrics, test_metrics, model_summaries,
    )

    LOGGER.info("=" * 80)
    LOGGER.info("EXPERIMENT COMPLETE")
    LOGGER.info("=" * 80)
    LOGGER.info("Best model: %s", model_path)
    LOGGER.info("Training history: %s", artifacts.training_history)
    LOGGER.info("Validation evaluation: %s", val_eval_dir)
    LOGGER.info("Test evaluation: %s", test_eval_dir)
    LOGGER.info("Model comparison chart: %s", chart_path)
    LOGGER.info("Model comparison table: %s", table_path)
    LOGGER.info("Result images folder: %s", RESULT_IMAGES_DIR)
    LOGGER.info("Experiment summary: %s", summary_path)
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()