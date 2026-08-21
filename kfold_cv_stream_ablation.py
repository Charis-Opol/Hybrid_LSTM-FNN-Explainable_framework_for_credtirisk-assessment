"""Stratified k-fold cross-validation for single-stream ablation models.

Mirrors Kfold_cv_hybrid.run_kfold_cv exactly (same fold strategy, early
stopping, class weighting, threshold selection, pooled OOF metrics) but
trains a model on only one input array (temporal or static) instead of
both, using stream_ablation_model.build_stream_ablation_model.
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
from stream_ablation_model import Stream, build_stream_ablation_model
from utils import ensure_directory, set_random_seed

LOGGER = logging.getLogger(__name__)


def find_best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, probabilities)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def run_kfold_cv_stream(
    stream: Stream,
    X: np.ndarray,
    labels: np.ndarray,
    output_dir: str | Path,
    n_splits: int = 5,
    epochs: int = 100,
    batch_size: int = 128,
    random_seed: int = RANDOM_SEED,
    cell_type: str = "gru",
) -> dict:
    """Run stratified k-fold CV over a single-stream ablation model.

    Args:
        stream: "temporal" or "static" -- which branch to keep.
        X: Input array -- X_temporal (n, seq_len, n_features) for
            stream="temporal", or X_static (n, n_features) for
            stream="static".
        labels: Binary labels, shape (n,).
        output_dir: Where fold artifacts and the final summary are written.
        n_splits: Number of stratified folds.
        epochs: Max epochs per fold (early stopping still applies per fold).
        batch_size: Batch size per fold.
        random_seed: Reproducibility seed.
        cell_type: Recurrent cell for stream="temporal", "gru" or "lstm".

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
        skf.split(X.reshape(len(X), -1), labels), start=1
    ):
        LOGGER.info("=== Fold %d/%d ===", fold_index, n_splits)

        X_trainfold = X[train_index]
        y_trainfold = labels[train_index]

        X_holdout = X[holdout_index]
        y_holdout = labels[holdout_index]

        stratify = y_trainfold if len(np.unique(y_trainfold)) > 1 else None
        X_tr, X_es, y_tr, y_es = train_test_split(
            X_trainfold,
            y_trainfold,
            test_size=0.15,
            random_state=random_seed,
            stratify=stratify,
        )

        if stream == "temporal":
            model = build_stream_ablation_model(
                stream="temporal",
                sequence_length=X.shape[1],
                temporal_features=X.shape[2],
                cell_type=cell_type,
            )
        else:
            model = build_stream_ablation_model(
                stream="static",
                static_features=X.shape[1],
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
        LOGGER.info("Fold %d class weights: %s", fold_index, class_weight)

        fold_dir = ensure_directory(output_path / f"fold_{fold_index}")
        checkpoint_path = fold_dir / "best_model.keras"

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", patience=8, mode="max", restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_auc",
                mode="max",
                save_best_only=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
            ),
        ]

        model.fit(
            X_tr,
            y_tr,
            validation_data=(X_es, y_es),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )

        fold_probabilities = model.predict(X_holdout).ravel()
        oof_probabilities[holdout_index] = fold_probabilities

        fold_auc = (
            roc_auc_score(y_holdout, fold_probabilities)
            if len(np.unique(y_holdout)) > 1
            else float("nan")
        )
        fold_ap = average_precision_score(y_holdout, fold_probabilities)
        LOGGER.info(
            "Fold %d: holdout positives=%d/%d, AUC=%.4f, AP=%.4f",
            fold_index,
            int(y_holdout.sum()),
            len(y_holdout),
            fold_auc,
            fold_ap,
        )
        fold_summaries.append(
            {
                "fold": fold_index,
                "holdout_positives": int(y_holdout.sum()),
                "holdout_size": len(y_holdout),
                "roc_auc": fold_auc,
                "average_precision": fold_ap,
            }
        )

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

    report = classification_report(
        oof_true, oof_predictions, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(output_path / "oof_classification_report.csv")

    np.save(output_path / "oof_probabilities.npy", oof_probabilities)
    np.save(output_path / "oof_true.npy", oof_true)

    summary = {"pooled_metrics": pooled_metrics, "fold_summaries": fold_summaries}
    (output_path / "kfold_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return summary


if __name__ == "__main__":
    raise RuntimeError(
        "Import run_kfold_cv_stream(stream, X, labels) and call it after "
        "building datasets, e.g. from a run_experiment script."
    )
