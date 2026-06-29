#!/usr/bin/env python
"""Full experiment on Uganda Mobile Money 15,000-row dataset.

Train hybrid LSTM-FNN model, evaluate, visualize results, and generate SHAP
explainability artifacts.
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
from explainability import HybridModelExplainer
from feature_engineering import FeatureColumns, engineer_features
from hybrid_model import F1Score
from static_dataset import StaticDatasetBuilder
from temporal_dataset import TemporalDatasetBuilder
from train import split_arrays, train_model
from utils import ensure_directory, set_random_seed
from visualize_results import (
    summarize_metrics,
    visualize_confusion_matrix,
    visualize_roc_auc,
    visualize_training_history,
    visualize_validation_metrics,
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "Uganda_Mobile_Money_Logs_15000.xlsx"
OUTPUT_DIR = PROJECT_DIR / "models" / "uganda_mobile_money_15000"
RESULT_IMAGES_DIR = OUTPUT_DIR / "training_result_images"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


def load_and_normalize_dataset(path: Path) -> pd.DataFrame:
    """Load workbook and map columns to framework names."""

    workbook = pd.ExcelFile(path)
    sheet_name = (
        "mobile_money_logs"
        if "mobile_money_logs" in workbook.sheet_names
        else workbook.sheet_names[0]
    )
    LOGGER.info("Reading sheet: %s", sheet_name)
    data = pd.read_excel(workbook, sheet_name=sheet_name)
    return data.rename(
        columns={
            "amount_ugx": "transaction_amount",
            "balance_after_ugx": "balance",
            "default_label": "defaulted",
            "loan_amount_ugx": "loan_amount",
            "sacco_member": "sacco_membership",
            "location_type": "location",
            "network": "preferred_network",
            "channel": "preferred_channel",
        }
    )


def build_training_arrays(data: pd.DataFrame):
    """Engineer features and build aligned temporal/static arrays."""

    feature_columns = FeatureColumns(
        amount="transaction_amount",
        balance="balance",
        target="defaulted",
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


def split_borrower_ids(
    borrower_ids: np.ndarray,
    labels: np.ndarray,
    random_seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Split borrower IDs with the same 70/15/15 scheme as split_arrays."""

    stratify = labels if len(np.unique(labels)) > 1 else None
    _, holdout_ids, _, holdout_labels = train_test_split(
        borrower_ids,
        labels,
        test_size=0.30,
        random_state=random_seed,
        stratify=stratify,
    )
    holdout_stratify = holdout_labels if len(np.unique(holdout_labels)) > 1 else None
    val_ids, test_ids = train_test_split(
        holdout_ids,
        test_size=0.50,
        random_state=random_seed,
        stratify=holdout_stratify,
    )
    return val_ids, test_ids


def copy_evaluation_images_to_results_dir(output_dir: Path, results_dir: Path) -> None:
    """Copy all evaluation and training plots into training_result_images."""

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
            image_sources.append(
                (image_path, f"{eval_split}_{image_path.name}")
            )

    shap_summary = output_dir / "explainability" / "global_shap_summary.png"
    if shap_summary.exists():
        image_sources.append((shap_summary, "global_shap_summary.png"))

    for source, dest_name in image_sources:
        shutil.copy2(source, results_dir / dest_name)
        LOGGER.info("Copied %s -> %s", source.name, results_dir / dest_name)


def write_experiment_summary(
    output_dir: Path,
    results_dir: Path,
    val_metrics: dict,
    test_metrics: dict,
    explain_dir: Path,
) -> Path:
    """Write a text summary of the experiment."""

    summary_path = results_dir / "EXPERIMENT_SUMMARY.txt"
    lines = [
        "Uganda Mobile Money Credit Risk Experiment (15,000 rows)",
        "=" * 60,
        f"Data source: {DATA_PATH.name}",
        f"Model output: {output_dir}",
        "",
        "Validation metrics:",
    ]
    for key, value in val_metrics.items():
        lines.append(f"  {key}: {value:.4f}")
    lines.append("")
    lines.append("Test metrics:")
    for key, value in test_metrics.items():
        lines.append(f"  {key}: {value:.4f}")
    lines.append("")
    lines.append("SHAP artifacts:")
    lines.append(f"  Global summary: {explain_dir / 'global_shap_summary.png'}")
    lines.append(f"  Feature importance: {explain_dir / 'feature_importance.csv'}")
    lines.append(f"  Local explanations: {explain_dir / 'local_explanations.json'}")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def main() -> None:
    set_random_seed(RANDOM_SEED)
    ensure_directory(OUTPUT_DIR)
    ensure_directory(RESULT_IMAGES_DIR)

    LOGGER.info("Loading dataset from %s", DATA_PATH)
    data = load_and_normalize_dataset(DATA_PATH)
    LOGGER.info("Loaded %d transaction rows", len(data))

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
        len(labels),
        len(temporal_names),
        len(static_names),
    )

    engineered.to_csv(OUTPUT_DIR / "engineered_features.csv", index=False)
    (OUTPUT_DIR / "temporal_feature_names.txt").write_text(
        "\n".join(temporal_names), encoding="utf-8"
    )
    (OUTPUT_DIR / "static_feature_names.txt").write_text(
        "\n".join(static_names), encoding="utf-8"
    )

    LOGGER.info("Starting model training (epochs=100, batch_size=32)")
    artifacts = train_model(
        X_temporal=X_temporal,
        X_static=X_static,
        labels=labels,
        output_dir=OUTPUT_DIR,
        epochs=100,
        batch_size=32,
        random_seed=RANDOM_SEED,
    )
    LOGGER.info("Training complete")

    _, val_set, test_set = split_arrays(
        X_temporal, X_static, labels, random_seed=RANDOM_SEED
    )
    X_temp_val, X_static_val, y_val = val_set
    X_temp_test, X_static_test, y_test = test_set
    val_borrower_ids, _ = split_borrower_ids(borrower_ids, labels, RANDOM_SEED)

    model_path = OUTPUT_DIR / "best_model.keras"
    model = tf.keras.models.load_model(
        model_path, custom_objects={"F1Score": F1Score}
    )

    LOGGER.info("Evaluating on validation set")
    val_eval_dir = ensure_directory(OUTPUT_DIR / "evaluation" / "validation")
    val_artifacts = evaluate_model(
        model, X_temp_val, X_static_val, y_val, output_dir=val_eval_dir
    )

    LOGGER.info("Evaluating on test set")
    test_eval_dir = ensure_directory(OUTPUT_DIR / "evaluation" / "test")
    test_artifacts = evaluate_model(
        model, X_temp_test, X_static_test, y_test, output_dir=test_eval_dir
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

    LOGGER.info("Generating SHAP explainability artifacts (this may take several minutes)")
    explain_dir = ensure_directory(OUTPUT_DIR / "explainability")
    explainer = HybridModelExplainer(
        model=model,
        temporal_feature_names=temporal_names,
        static_feature_names=static_names,
        output_dir=explain_dir,
    )
    explain_artifacts = explainer.explain(
        X_temporal=X_temp_val,
        X_static=X_static_val,
        borrower_ids=val_borrower_ids,
        background_size=min(100, len(y_val)),
        explanation_size=min(50, len(y_val)),
    )

    copy_evaluation_images_to_results_dir(OUTPUT_DIR, RESULT_IMAGES_DIR)
    summary_path = write_experiment_summary(
        OUTPUT_DIR, RESULT_IMAGES_DIR, val_metrics, test_metrics, explain_dir
    )

    LOGGER.info("=" * 80)
    LOGGER.info("EXPERIMENT COMPLETE")
    LOGGER.info("=" * 80)
    LOGGER.info("Best model: %s", model_path)
    LOGGER.info("Training history: %s", artifacts.training_history)
    LOGGER.info("Validation evaluation: %s", val_eval_dir)
    LOGGER.info("Test evaluation: %s", test_eval_dir)
    LOGGER.info("Result images folder: %s", RESULT_IMAGES_DIR)
    LOGGER.info("SHAP global summary: %s", explain_artifacts.global_summary_plot)
    LOGGER.info("SHAP feature importance: %s", explain_artifacts.feature_importance_csv)
    LOGGER.info("SHAP local explanations: %s", explain_artifacts.local_explanations_json)
    LOGGER.info("Experiment summary: %s", summary_path)
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()
