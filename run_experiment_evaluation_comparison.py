#!/usr/bin/env python
"""Evaluation comparison: hybrid, logistic regression, XGBoost, and both
transformer variants.

Trains and evaluates five models with stratified 5-fold CV using pooled
out-of-fold predictions (required given the very small positive class,
~2.3%, ~70 positive borrowers total). None of these models are
hyperparameter-tuned -- this is a clean, like-for-like comparison at
each model family's default settings:

  1. Hybrid LSTM-GRU-Attention-FNN     -- original architecture
  2. Logistic Regression (untuned)     -- linear baseline
  3. XGBoost (untuned)                 -- tree baseline
  4. Transformer-Encoder Hybrid        -- GRU replaced by self-attention
  5. Cross-Attention Fusion Hybrid     -- static queries temporal sequence

Outputs, all under one folder:
  - model_comparison.png   grouped bar chart across accuracy/precision/
                            recall/F1/AP/ROC-AUC
  - model_comparison.md    same metrics as a markdown table
  - pr_curves_overlay.png  all five PR curves + random baseline on one plot
  - lift_over_random.md    average precision vs. random-guessing baseline,
                            expressed as a lift multiplier, per model
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import RANDOM_SEED
from feature_engineering import FeatureColumns, engineer_features
from kfold_cv_cross_attention import run_kfold_cv_cross_attention
from Kfold_cv_hybrid import run_kfold_cv
from logistic_regression_baseline import run_logistic_regression_baseline
from static_dataset import StaticDatasetBuilder
from temporal_dataset import TemporalDatasetBuilder
from transformer_model import run_kfold_cv_transformer
from utils import ensure_directory, set_random_seed
from visualize_model_comparison import compare_all_models, compute_lift_table
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
    / "uganda_mobile_money_evaluation_comparison"
)

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


def write_experiment_summary(
    results_dir: Path,
    model_summaries: dict[str, dict],
    lift_table: pd.DataFrame,
) -> Path:
    """Write a plain-text summary alongside the markdown/chart artifacts."""

    summary_path = results_dir / "EXPERIMENT_SUMMARY.txt"
    lines = [
        "Evaluation Comparison: Hybrid vs. Logistic Regression vs. XGBoost vs. Transformers",
        "=" * 80,
        f"Data source: {DATA_PATH.name}",
        "All models untuned (default hyperparameters). 5-fold CV, pooled OOF metrics.",
        "",
    ]
    for model_name, metrics in model_summaries.items():
        lines.append(f"{model_name} - pooled out-of-fold metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
        lines.append("")

    lines.append("Lift over random baseline (Average Precision / Base Rate):")
    for _, row in lift_table.iterrows():
        lines.append(
            f"  {row['Model']}: base_rate={row['Base Rate']:.4f}, "
            f"AP={row['Avg. Precision']:.4f}, lift={row['Lift over Random']:.2f}x"
        )

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def main() -> None:
    set_random_seed(RANDOM_SEED)
    ensure_directory(OUTPUT_DIR)

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

    unique_labels, label_counts = np.unique(labels, return_counts=True)
    LOGGER.info("Overall label distribution: %s", dict(zip(unique_labels.tolist(), label_counts.tolist())))
    LOGGER.info("Total borrowers: %d, positive rate: %.4f", len(labels), labels.mean())

    engineered.to_csv(OUTPUT_DIR / "engineered_features.csv", index=False)
    (OUTPUT_DIR / "temporal_feature_names.txt").write_text("\n".join(temporal_names), encoding="utf-8")
    (OUTPUT_DIR / "static_feature_names.txt").write_text("\n".join(static_names), encoding="utf-8")

    # --- 5-fold CV: all five models, none hyperparameter-tuned ---
    LOGGER.info("Running 5-fold CV: hybrid LSTM-GRU-Attention-FNN")
    hybrid_summary = run_kfold_cv(
        X_temporal=X_temporal, X_static=X_static, labels=labels,
        output_dir=OUTPUT_DIR / "kfold_hybrid", n_splits=5,
    )

    LOGGER.info("Running 5-fold CV: logistic regression (untuned)")
    logreg_summary = run_logistic_regression_baseline(
        X_temporal=X_temporal, X_static=X_static, labels=labels,
        temporal_feature_names=temporal_names, static_feature_names=static_names,
        output_dir=OUTPUT_DIR / "logistic_regression", n_splits=5,
    )

    LOGGER.info("Running 5-fold CV: XGBoost (untuned)")
    xgb_summary = run_xgboost_baseline(
        X_temporal=X_temporal, X_static=X_static, labels=labels,
        temporal_feature_names=temporal_names, static_feature_names=static_names,
        output_dir=OUTPUT_DIR / "xgboost", n_splits=5,
    )

    LOGGER.info("Running 5-fold CV: Transformer-encoder hybrid")
    transformer_summary = run_kfold_cv_transformer(
        X_temporal=X_temporal, X_static=X_static, labels=labels,
        output_dir=OUTPUT_DIR / "kfold_transformer", n_splits=5,
    )

    LOGGER.info("Running 5-fold CV: cross-attention fusion hybrid")
    cross_attention_summary = run_kfold_cv_cross_attention(
        X_temporal=X_temporal, X_static=X_static, labels=labels,
        output_dir=OUTPUT_DIR / "kfold_cross_attention", n_splits=5,
    )

    model_summaries = {
        "Hybrid LSTM-GRU-Attention-FNN": hybrid_summary["pooled_metrics"],
        "Logistic Regression": logreg_summary["pooled_metrics"],
        "XGBoost": xgb_summary["pooled_metrics"],
        "Transformer-Encoder Hybrid": transformer_summary["pooled_metrics"],
        "Cross-Attention Fusion Hybrid": cross_attention_summary["pooled_metrics"],
    }

    LOGGER.info("Building comparison chart, table, PR overlay, and lift table")
    comparison_dir = OUTPUT_DIR / "model_comparison"
    chart_path, table_path, pr_overlay_path, lift_table_path = compare_all_models(
        model_result_dirs={
            "Hybrid LSTM-GRU-Attention-FNN": OUTPUT_DIR / "kfold_hybrid",
            "Logistic Regression": OUTPUT_DIR / "logistic_regression",
            "XGBoost": OUTPUT_DIR / "xgboost",
            "Transformer-Encoder Hybrid": OUTPUT_DIR / "kfold_transformer",
            "Cross-Attention Fusion Hybrid": OUTPUT_DIR / "kfold_cross_attention",
        },
        output_dir=comparison_dir,
    )

    lift_table = compute_lift_table(model_summaries)
    summary_path = write_experiment_summary(OUTPUT_DIR, model_summaries, lift_table)

    LOGGER.info("=" * 80)
    LOGGER.info("EXPERIMENT COMPLETE")
    LOGGER.info("=" * 80)
    LOGGER.info("Comparison chart: %s", chart_path)
    LOGGER.info("Comparison table: %s", table_path)
    LOGGER.info("PR curves overlay: %s", pr_overlay_path)
    LOGGER.info("Lift over random table: %s", lift_table_path)
    LOGGER.info("Experiment summary: %s", summary_path)
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()
