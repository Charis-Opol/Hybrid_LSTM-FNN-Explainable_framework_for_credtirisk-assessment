#!/usr/bin/env python
"""Ablation: how much do the temporal (GRU/LSTM) and static (FNN) streams
each contribute to the hybrid model's classification performance?

Trains three single-stream models -- temporal-only (GRU), temporal-only
(LSTM), static-only (FNN) -- with the exact same classification head as
the full hybrid model (see stream_ablation_model.py), under the same
5-fold CV protocol, same seed, same cached engineered feature set as
run_ablation_gru_vs_lstm.py. Combines them with the full hybrid
GRU/LSTM results already produced by that script (models/ablation_gru_vs_lstm/)
into one comparison table, so all five configurations are directly
comparable:

  1. Static-only (FNN)
  2. Temporal-only (GRU)
  3. Temporal-only (LSTM)
  4. Full hybrid (GRU + static)
  5. Full hybrid (LSTM + static)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from config import RANDOM_SEED
from kfold_cv_stream_ablation import run_kfold_cv_stream
from static_dataset import StaticDatasetBuilder
from temporal_dataset import TemporalDatasetBuilder

PROJECT_DIR = Path(__file__).resolve().parent
ENGINEERED_FEATURES_PATH = (
    PROJECT_DIR
    / "models"
    / "uganda_mobile_money_708000_8_no_SHAP"
    / "engineered_features.csv"
)
FULL_HYBRID_DIR = PROJECT_DIR / "models" / "ablation_gru_vs_lstm"
OUTPUT_DIR = PROJECT_DIR / "models" / "ablation_streams"

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


def load_cached_summary(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["pooled_metrics"]


def main() -> None:
    LOGGER.info("Loading cached engineered features from %s", ENGINEERED_FEATURES_PATH)
    engineered = pd.read_csv(ENGINEERED_FEATURES_PATH)

    X_temporal, X_static, labels = build_arrays(engineered)
    LOGGER.info(
        "Built arrays: %d borrowers, temporal shape %s, static shape %s, positive rate %.4f",
        len(labels), X_temporal.shape, X_static.shape, labels.mean(),
    )

    results: dict[str, dict] = {}

    LOGGER.info("=" * 80)
    LOGGER.info("Running 5-fold CV: static-only (FNN)")
    LOGGER.info("=" * 80)
    summary = run_kfold_cv_stream(
        stream="static", X=X_static, labels=labels,
        output_dir=OUTPUT_DIR / "kfold_static_only", n_splits=5,
        random_seed=RANDOM_SEED,
    )
    results["Static-only (FNN)"] = summary["pooled_metrics"]

    for cell_type in ("gru", "lstm"):
        LOGGER.info("=" * 80)
        LOGGER.info("Running 5-fold CV: temporal-only (%s)", cell_type.upper())
        LOGGER.info("=" * 80)
        summary = run_kfold_cv_stream(
            stream="temporal", X=X_temporal, labels=labels,
            output_dir=OUTPUT_DIR / f"kfold_temporal_only_{cell_type}", n_splits=5,
            random_seed=RANDOM_SEED, cell_type=cell_type,
        )
        results[f"Temporal-only ({cell_type.upper()})"] = summary["pooled_metrics"]

    LOGGER.info("Loading cached full-hybrid GRU/LSTM results from %s", FULL_HYBRID_DIR)
    results["Full Hybrid (GRU + static)"] = load_cached_summary(
        FULL_HYBRID_DIR / "kfold_gru" / "kfold_summary.json"
    )
    results["Full Hybrid (LSTM + static)"] = load_cached_summary(
        FULL_HYBRID_DIR / "kfold_lstm" / "kfold_summary.json"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "ablation_streams_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    metric_keys = [
        "accuracy", "precision", "recall", "f1",
        "average_precision", "roc_auc", "selected_threshold",
    ]
    order = [
        "Static-only (FNN)",
        "Temporal-only (GRU)",
        "Temporal-only (LSTM)",
        "Full Hybrid (GRU + static)",
        "Full Hybrid (LSTM + static)",
    ]
    lines = [
        "# Stream Ablation -- Temporal (GRU/LSTM) vs. Static (FNN) Contribution",
        "",
        "Classification head (Dense64 -> BN -> Dropout -> Dense32 -> BN -> Dropout "
        "-> sigmoid) held identical across all rows; only the input stream(s) change.",
        "",
        "| Configuration | " + " | ".join(metric_keys) + " |",
        "| --- | " + " | ".join(["---"] * len(metric_keys)) + " |",
    ]
    for name in order:
        metrics = results[name]
        row = [name] + [f"{metrics[key]:.4f}" for key in metric_keys]
        lines.append("| " + " | ".join(row) + " |")

    table_path = OUTPUT_DIR / "ablation_streams_table.md"
    table_path.write_text("\n".join(lines), encoding="utf-8")

    LOGGER.info("=" * 80)
    LOGGER.info("STREAM ABLATION COMPLETE")
    LOGGER.info("=" * 80)
    for name in order:
        LOGGER.info("%s: %s", name, results[name])
    LOGGER.info("Summary: %s", OUTPUT_DIR / "ablation_streams_summary.json")
    LOGGER.info("Table: %s", table_path)


if __name__ == "__main__":
    main()
