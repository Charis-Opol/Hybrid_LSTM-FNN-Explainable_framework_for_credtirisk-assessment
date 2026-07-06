"""Stratified k-fold cross-validation for the Transformer-encoder variant.

Same OOF-pooling approach as kfold_cv_hybrid.py, so results are directly
comparable in the model comparison table.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

from config import MODELS_DIR, RANDOM_SEED
from hybrid_transformer_encoder_model import build_transformer_hybrid_model
from utils import ensure_directory, set_random_seed

LOGGER = logging.getLogger(__name__)


def find_best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Find the probability threshold that maximizes F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, probabilities)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def run_kfold_cv_transformer_encoder(
    X_temporal: np.ndarray,
    X_static: np.ndarray,
    labels: np.ndarray,
    output_dir: str | Path = MODELS_DIR / "kfold_transformer_encoder",
    n_splits: int = 5,
    epochs: int = 100,
    batch_size: int = 128,
    random_seed: int = RANDOM_SEED,
) -> dict:
    """Run stratified k-fold CV over the Transformer-encoder hybrid model.

    Args:
        X_temporal: Temporal input, shape (n, sequence_length, temporal_features).
        X_static: Static input, shape (n, static_features).
        labels: Binary labels, shape (n,).
        output_dir: Where fold artifacts and the final summary are written.
        n_splits: Number of stratified folds.
        epochs: Max epochs per fold (early stopping still applies per fold).
        batch_size: Batch size per fold.
        random_seed: Reproducibility seed.

    Returns:
        Dict with pooled out-of-fold metrics and per-fold breakdown.
    """
    set_random_seed(random_seed)
    output_path = ensure_directory(Path(output_dir))
    labels = labels.astype(float)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    oof_probabilities = np.zeros(len(labels), dtype=float)
    oof_true = labels.copy()
    fold_summaries: list[dict] = []

    for fold_index, (train_index, holdout_index) in enumerate(
        skf.split(X_static, labels), start=1
    ):
        LOGGER.info("=== Fold %d/%d ===", fold_index, n_splits)

        X_temp_trainfold = X_temporal[train_index]
        X_static_trainfold = X_static[train_index]
        y_trainfold = labels[train_index]

        X_temp_holdout = X_temporal[holdout_index]
        X_static_holdout = X_static[holdout_index]
        y_holdout = labels[holdout_index]

        stratify = y_trainfold if len(np.unique(y_trainfold)) > 1 else None
        (
            X_temp_tr, X_temp_es, X_static_tr, X_static_es, y_tr, y_es
        ) = train_test_split(
            X_temp_trainfold, X_static_trainfold, y_trainfold,
            test_size=0.15, random_state=random_seed, stratify=stratify,
        )

        model = build_transformer_hybrid_model(
            sequence_length=X_temporal.shape[1],
            temporal_features=X_temporal.shape[2],
            static_features=X_static.shape[1],
        )

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_tr.astype(int)),
            y=y_tr.astype(int),
        )
        class_weight = {
            int(cls): float(weight)
            for cls, weight in zip(np.unique(y_tr.astype(int)), class_weights)
        }

        fold_dir = ensure_directory(output_path / f"fold_{fold_index}")
        checkpoint_path = fold_dir / "best_model.keras"

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", patience=8, mode="max", restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path), monitor="val_auc", mode="max", save_best_only=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
            ),
        ]

        model.fit(
            [X_temp_tr, X_static_tr], y_tr,
            validation_data=([X_temp_es, X_static_es], y_es),
            epochs=epochs, batch_size=batch_size,
            class_weight=class_weight, callbacks=callbacks, verbose=1,
        )

        fold_probabilities = model.predict([X_temp_holdout, X_static_holdout]).ravel()
        oof_probabilities[holdout_index] = fold_probabilities

        fold_auc = (
            roc_auc_score(y_holdout, fold_probabilities)
            if len(np.unique(y_holdout)) > 1 else float("nan")
        )
        fold_ap = average_precision_score(y_holdout, fold_probabilities)
        LOGGER.info(
            "Fold %d: holdout positives=%d/%d, AUC=%.4f, AP=%.4f",
            fold_index, int(y_holdout.sum()), len(y_holdout), fold_auc, fold_ap,
        )
        fold_summaries.append({
            "fold": fold_index,
            "holdout_positives": int(y_holdout.sum()),
            "holdout_size": len(y_holdout),
            "roc_auc": fold_auc,
            "average_precision": fold_ap,
        })

        model.save(fold_dir / "trained_model.keras")

    best_threshold, best_f1 = find_best_threshold(oof_true, oof_probabilities)
    oof_predictions = (oof_probabilities >= best_threshold).astype(int)

    pooled_metrics = {
        "n_positives_total": int(oof_true.sum()),
        "n_total": len(oof_true),
        "selected_threshold": best_threshold,
        "accuracy": float(accuracy_score(oof_true, oof_predictions)),
        "precision": float(precision_score(oof_true, oof_predictions, zero_division=0)),
        "recall": float(recall_score(oof_true, oof_predictions, zero_division=0)),
        "f1": float(f1_score(oof_true, oof_predictions, zero_division=0)),
        "average_precision": float(average_precision_score(oof_true, oof_probabilities)),
        "roc_auc": float(roc_auc_score(oof_true, oof_probabilities)),
    }
    LOGGER.info("Pooled out-of-fold metrics: %s", pooled_metrics)

    report = classification_report(oof_true, oof_predictions, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(output_path / "oof_classification_report.csv")

    np.save(output_path / "oof_probabilities.npy", oof_probabilities)
    np.save(output_path / "oof_true.npy", oof_true)

    summary = {"pooled_metrics": pooled_metrics, "fold_summaries": fold_summaries}
    (output_path / "kfold_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


if __name__ == "__main__":
    raise RuntimeError(
        "Import run_kfold_cv_transformer_encoder(X_temporal, X_static, labels) "
        "and call it after building datasets, e.g. from a run_experiment script."
    )
