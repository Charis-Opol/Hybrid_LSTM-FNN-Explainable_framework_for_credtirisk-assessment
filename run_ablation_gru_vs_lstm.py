#!/usr/bin/env python
"""Ablation: GRU vs LSTM as the hybrid model's recurrent cell.

Every prior "hybrid vs vanilla LSTM baseline" comparison in this repo
changes three things at once -- cell type, attention, and the static
feature branch -- so it can't tell you whether GRU specifically helps.
This script holds architecture, hyperparameters, data, folds, and seed
fixed and swaps only the recurrent cell (GRU vs LSTM) inside
TemporalEncoder, via Kfold_cv_hybrid.run_kfold_cv(..., cell_type=...).

Reuses the engineered feature set already produced by the
708000_8_no_SHAP experiment so this doesn't redo feature engineering
on the raw 708k-row transaction log.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from config import RANDOM_SEED
from Kfold_cv_hybrid import run_kfold_cv
from static_dataset import StaticDatasetBuilder
from temporal_dataset import TemporalDatasetBuilder

PROJECT_DIR = Path(__file__).resolve().parent
ENGINEERED_FEATURES_PATH = (
    PROJECT_DIR
    / "models"
    / "uganda_mobile_money_708000_8_no_SHAP"
    / "engineered_features.csv"
)
OUTPUT_DIR = PROJECT_DIR / "models" / "ablation_gru_vs_lstm"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


def build_arrays(engineered: pd.DataFrame):
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
        raise ValueError("Engineered features must include default labels.")

    return temporal.X_temporal, X_static_aligned, temporal.labels


def main() -> None:
    LOGGER.info("Loading cached engineered features from %s", ENGINEERED_FEATURES_PATH)
    engineered = pd.read_csv(ENGINEERED_FEATURES_PATH)

    X_temporal, X_static, labels = build_arrays(engineered)
    LOGGER.info(
        "Built arrays: %d borrowers, temporal shape %s, static shape %s, positive rate %.4f",
        len(labels), X_temporal.shape, X_static.shape, labels.mean(),
    )

    results = {}
    for cell_type in ("gru", "lstm"):
        LOGGER.info("=" * 80)
        LOGGER.info("Running 5-fold CV for hybrid model with cell_type=%s", cell_type)
        LOGGER.info("=" * 80)
        summary = run_kfold_cv(
            X_temporal=X_temporal,
            X_static=X_static,
            labels=labels,
            output_dir=OUTPUT_DIR / f"kfold_{cell_type}",
            n_splits=5,
            random_seed=RANDOM_SEED,
            cell_type=cell_type,
        )
        results[cell_type] = summary["pooled_metrics"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "ablation_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    metric_keys = [
        "accuracy", "precision", "recall", "f1",
        "average_precision", "roc_auc", "selected_threshold",
    ]
    lines = [
        "# GRU vs LSTM Ablation -- Hybrid Model (attention + static branch held fixed)",
        "",
        "| Cell type | " + " | ".join(metric_keys) + " |",
        "| --- | " + " | ".join(["---"] * len(metric_keys)) + " |",
    ]
    for cell_type, metrics in results.items():
        row = [cell_type.upper()] + [f"{metrics[key]:.4f}" for key in metric_keys]
        lines.append("| " + " | ".join(row) + " |")

    table_path = OUTPUT_DIR / "ablation_table.md"
    table_path.write_text("\n".join(lines), encoding="utf-8")

    LOGGER.info("=" * 80)
    LOGGER.info("ABLATION COMPLETE")
    LOGGER.info("=" * 80)
    for cell_type, metrics in results.items():
        LOGGER.info("%s: %s", cell_type.upper(), metrics)
    LOGGER.info("Summary: %s", OUTPUT_DIR / "ablation_summary.json")
    LOGGER.info("Table: %s", table_path)


if __name__ == "__main__":
    main()
