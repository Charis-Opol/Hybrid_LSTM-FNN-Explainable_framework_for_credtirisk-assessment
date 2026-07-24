#!/usr/bin/env python
"""Add the vanilla LSTM baseline to the master.csv comparison, and sweep
class-weight strength for the two deep models (vanilla LSTM, hybrid
LSTM-GRU-Attention-FNN) to see whether a gentler weighting than the
"balanced" default recovers precision.

SMOTE is intentionally not applied to the temporal sequence models here --
oversampling a (sequence_length, features) tensor by interpolating between
borrower sequences has no established, defensible formulation the way it
does for flat tabular rows, so this sweep only varies class-weight strength.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import RANDOM_SEED
from hybrid_model import build_hybrid_model
from imbalance_sweep import class_weight_at_strength, pooled_metrics_from_oof
from kfold_cv_vanilla_lstm import build_vanilla_lstm, run_kfold_cv_vanilla_lstm
from run_experiment_evaluation_comparison import (
    DATA_PATH,
    OUTPUT_DIR,
    build_training_arrays,
    load_and_normalize_dataset,
)
from utils import ensure_directory, set_random_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

SWEEP_DIR = OUTPUT_DIR / "imbalance_sweep"
WEIGHT_STRENGTHS = [0.0, 0.5, 1.5]  # 1.0 ("balanced") is the existing baseline run


def run_vanilla_lstm_weight_sweep(X_temporal, labels, weight_strength, n_splits, random_seed, epochs=100, batch_size=128):
    set_random_seed(random_seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    oof = np.zeros(len(labels), dtype=float)

    for fold_index, (train_index, holdout_index) in enumerate(skf.split(X_temporal, labels), start=1):
        X_trainfold, y_trainfold = X_temporal[train_index], labels[train_index]
        X_holdout = X_temporal[holdout_index]

        stratify = y_trainfold if len(np.unique(y_trainfold)) > 1 else None
        X_tr, X_es, y_tr, y_es = train_test_split(
            X_trainfold, y_trainfold, test_size=0.15, random_state=random_seed, stratify=stratify,
        )

        model = build_vanilla_lstm(sequence_length=X_temporal.shape[1], temporal_features=X_temporal.shape[2])
        class_weight = class_weight_at_strength(y_tr.astype(int), weight_strength)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=8, mode="max", restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        ]
        model.fit(
            X_tr, y_tr, validation_data=(X_es, y_es), epochs=epochs, batch_size=batch_size,
            class_weight=class_weight, callbacks=callbacks, verbose=0,
        )
        oof[holdout_index] = model.predict(X_holdout, verbose=0).ravel()
        LOGGER.info("  fold %d/%d done", fold_index, n_splits)

    return oof


def run_hybrid_weight_sweep(X_temporal, X_static, labels, weight_strength, n_splits, random_seed, epochs=100, batch_size=128):
    set_random_seed(random_seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    oof = np.zeros(len(labels), dtype=float)

    for fold_index, (train_index, holdout_index) in enumerate(skf.split(X_static, labels), start=1):
        X_temp_trainfold, X_static_trainfold, y_trainfold = X_temporal[train_index], X_static[train_index], labels[train_index]
        X_temp_holdout, X_static_holdout = X_temporal[holdout_index], X_static[holdout_index]

        stratify = y_trainfold if len(np.unique(y_trainfold)) > 1 else None
        (X_temp_tr, X_temp_es, X_static_tr, X_static_es, y_tr, y_es) = train_test_split(
            X_temp_trainfold, X_static_trainfold, y_trainfold,
            test_size=0.15, random_state=random_seed, stratify=stratify,
        )

        model = build_hybrid_model(
            sequence_length=X_temporal.shape[1], temporal_features=X_temporal.shape[2], static_features=X_static.shape[1],
        )
        class_weight = class_weight_at_strength(y_tr.astype(int), weight_strength)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=8, mode="max", restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        ]
        model.fit(
            [X_temp_tr, X_static_tr], y_tr, validation_data=([X_temp_es, X_static_es], y_es),
            epochs=epochs, batch_size=batch_size, class_weight=class_weight, callbacks=callbacks, verbose=0,
        )
        oof[holdout_index] = model.predict([X_temp_holdout, X_static_holdout], verbose=0).ravel()
        LOGGER.info("  fold %d/%d done", fold_index, n_splits)

    return oof


def main() -> None:
    ensure_directory(SWEEP_DIR)

    data = load_and_normalize_dataset(DATA_PATH)
    data = data.sort_values(["borrower_id", "transaction_date"]).reset_index(drop=True)
    X_temporal, X_static, labels, *_ = build_training_arrays(data)
    labels = labels.astype(float)
    LOGGER.info("n=%d, positives=%d (%.2f%%)", len(labels), labels.sum(), 100 * labels.mean())

    LOGGER.info("=== Running vanilla LSTM baseline (balanced weighting) for main comparison ===")
    vanilla_summary = run_kfold_cv_vanilla_lstm(
        X_temporal=X_temporal, labels=labels, output_dir=OUTPUT_DIR / "kfold_vanilla_lstm", n_splits=5,
    )
    LOGGER.info("Vanilla LSTM baseline pooled metrics: %s", vanilla_summary["pooled_metrics"])

    rows = [{"model": "vanilla_lstm", "config": "balanced (baseline)", **pooled_metrics_from_oof(
        np.load(OUTPUT_DIR / "kfold_vanilla_lstm" / "oof_true.npy"),
        np.load(OUTPUT_DIR / "kfold_vanilla_lstm" / "oof_probabilities.npy"),
    )}]
    rows.append({"model": "hybrid", "config": "balanced (baseline)", **pooled_metrics_from_oof(
        np.load(OUTPUT_DIR / "kfold_hybrid" / "oof_true.npy"),
        np.load(OUTPUT_DIR / "kfold_hybrid" / "oof_probabilities.npy"),
    )})

    for strength in WEIGHT_STRENGTHS:
        config_name = f"weight_{strength}"

        LOGGER.info("=== vanilla_lstm / %s ===", config_name)
        oof = run_vanilla_lstm_weight_sweep(X_temporal, labels, strength, n_splits=5, random_seed=RANDOM_SEED)
        np.save(SWEEP_DIR / f"vanilla_lstm__{config_name}__oof_probabilities.npy", oof)
        metrics = pooled_metrics_from_oof(labels, oof)
        metrics.update({"model": "vanilla_lstm", "config": config_name})
        rows.append(metrics)
        LOGGER.info("  AP=%.4f AUC=%.4f precision@F1=%.4f recall@F1=%.4f precision@recall0.5=%.4f",
                    metrics["average_precision"], metrics["roc_auc"], metrics["precision"], metrics["recall"], metrics["precision_at_recall_0.5"])

        LOGGER.info("=== hybrid / %s ===", config_name)
        oof = run_hybrid_weight_sweep(X_temporal, X_static, labels, strength, n_splits=5, random_seed=RANDOM_SEED)
        np.save(SWEEP_DIR / f"hybrid__{config_name}__oof_probabilities.npy", oof)
        metrics = pooled_metrics_from_oof(labels, oof)
        metrics.update({"model": "hybrid", "config": config_name})
        rows.append(metrics)
        LOGGER.info("  AP=%.4f AUC=%.4f precision@F1=%.4f recall@F1=%.4f precision@recall0.5=%.4f",
                    metrics["average_precision"], metrics["roc_auc"], metrics["precision"], metrics["recall"], metrics["precision_at_recall_0.5"])

    frame = pd.DataFrame(rows)
    frame = frame[[
        "model", "config", "average_precision", "roc_auc", "precision", "recall", "f1",
        "precision_at_recall_0.5", "n_flagged_at_recall_0.5", "selected_threshold",
    ]]
    frame.to_csv(SWEEP_DIR / "deep_sweep_results.csv", index=False)
    LOGGER.info("Saved sweep results to %s", SWEEP_DIR / "deep_sweep_results.csv")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
