#!/usr/bin/env python
"""Hybrid LSTM vs. Transformer-encoder hybrid, head-to-head comparison.

Trains and evaluates exactly two models with stratified 5-fold CV using
pooled out-of-fold predictions (required given the very small positive
class, ~2.3%, ~70 positive borrowers total):

  1. Hybrid LSTM-GRU-Attention-FNN (kfold_cv_hybrid.py)   -- original model
  2. Transformer-Encoder Hybrid (transformer_model.py)     -- GRU replaced
     by a Transformer encoder (positional embeddings + self-attention +
     feed-forward), static branch and fusion unchanged

Reuses the same data pipeline, k-fold CV, and comparison visualization
modules as the other experiments. All outputs are written under one
folder: models/uganda_mobile_money_hybrid_vs_transformer/
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
from static_dataset import StaticDatasetBuilder
from temporal_dataset import TemporalDatasetBuilder
from transformer_model import run_kfold_cv_transformer
from utils import ensure_directory, set_random_seed
from visualize_model_comparison import compare_all_models

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
    / "uganda_mobile_money_hybrid_vs_transformer2"
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


def write_experiment_summary(results_dir: Path, model_summaries: dict[str, dict]) -> Path:
    """Write a text summary comparing the two models."""

    summary_path = results_dir / "EXPERIMENT_SUMMARY.txt"
    lines = [
        "Hybrid LSTM vs. Transformer-Encoder Hybrid",
        "=" * 60,
        f"Data source: {DATA_PATH.name}",
        "",
    ]
    for model_name, metrics in model_summaries.items():
        lines.append(f"{model_name} - pooled out-of-fold metrics (5-fold CV):")
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
        lines.append("")

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

    # --- 5-fold CV: hybrid LSTM (original) ---
    LOGGER.info("Running 5-fold CV: hybrid LSTM-GRU-Attention-FNN")
    hybrid_summary = run_kfold_cv(
        X_temporal=X_temporal, X_static=X_static, labels=labels,
        output_dir=OUTPUT_DIR / "kfold_hybrid", n_splits=5,
    )

    # --- 5-fold CV: Transformer-encoder hybrid ---
    LOGGER.info("Running 5-fold CV: Transformer-encoder hybrid")
    transformer_summary = run_kfold_cv_transformer(
        X_temporal=X_temporal, X_static=X_static, labels=labels,
        output_dir=OUTPUT_DIR / "kfold_transformer", n_splits=5,
    )

    # --- 5-fold CV: Cross-attention fusion hybrid ---
    LOGGER.info("Running 5-fold CV: Cross-attention fusion hybrid")
    cross_attention_summary = run_kfold_cv_cross_attention(
        X_temporal=X_temporal, X_static=X_static, labels=labels,
        output_dir=OUTPUT_DIR / "kfold_cross_attention", n_splits=5,
    )

    model_summaries = {
        "Hybrid LSTM-GRU-Attention-FNN": hybrid_summary["pooled_metrics"],
        "Transformer-Encoder Hybrid": transformer_summary["pooled_metrics"],
        "Cross-Attention Fusion Hybrid": cross_attention_summary["pooled_metrics"],
    }

    LOGGER.info("Building model comparison chart and table")
    comparison_dir = OUTPUT_DIR / "model_comparison"
    chart_path, table_path = compare_all_models(
        model_result_dirs={
            "Hybrid LSTM-GRU-Attention-FNN": OUTPUT_DIR / "kfold_hybrid",
            "Transformer-Encoder Hybrid": OUTPUT_DIR / "kfold_transformer",
            "Cross-Attention Fusion Hybrid": OUTPUT_DIR / "kfold_cross_attention",
        },
        output_dir=comparison_dir,
    )
    LOGGER.info("Comparison chart: %s", chart_path)
    LOGGER.info("Comparison table: %s", table_path)

    summary_path = write_experiment_summary(OUTPUT_DIR, model_summaries)

    LOGGER.info("=" * 80)
    LOGGER.info("EXPERIMENT COMPLETE")
    LOGGER.info("=" * 80)
    LOGGER.info("Hybrid CV results: %s", OUTPUT_DIR / "kfold_hybrid" / "kfold_summary.json")
    LOGGER.info("Transformer CV results: %s", OUTPUT_DIR / "kfold_transformer" / "kfold_summary.json")
    LOGGER.info("Cross-attention CV results: %s", OUTPUT_DIR / "kfold_cross_attention" / "kfold_summary.json")
    LOGGER.info("Comparison chart: %s", chart_path)
    LOGGER.info("Comparison table: %s", table_path)
    LOGGER.info("Experiment summary: %s", summary_path)
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()