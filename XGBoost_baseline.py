"""XGBoost baseline for the credit risk classification task.

Flattens the temporal features and concatenates them with the static
features into a single tabular representation, then evaluates with
stratified k-fold CV -- the same OOF-pooling approach as the hybrid
model's CV script, for a fair, stable comparison given the very small
positive class (~2%, ~69 positive borrowers total).

Requires xgboost: pip install xgboost
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
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
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from config import MODELS_DIR, RANDOM_SEED
from utils import ensure_directory

LOGGER = logging.getLogger(__name__)


def find_best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Find the probability threshold that maximizes F1 score.

    Args:
        y_true: Ground-truth binary labels.
        probabilities: Predicted probabilities.

    Returns:
        Tuple of (best_threshold, best_f1_score).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, probabilities)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def flatten_temporal(X_temporal: np.ndarray) -> np.ndarray:
    """Flatten (n, sequence_length, features) into (n, sequence_length * features)."""
    return X_temporal.reshape(X_temporal.shape[0], -1)


def run_xgboost_baseline(
    X_temporal: np.ndarray,
    X_static: np.ndarray,
    labels: np.ndarray,
    temporal_feature_names: list[str] | None = None,
    static_feature_names: list[str] | None = None,
    output_dir: str | Path = MODELS_DIR / "xgboost_baseline",
    n_splits: int = 5,
    random_seed: int = RANDOM_SEED,
) -> dict:
    """Train and evaluate an XGBoost baseline with stratified k-fold CV.

    Args:
        X_temporal: Temporal input, shape (n, sequence_length, temporal_features).
        X_static: Static input, shape (n, static_features).
        labels: Binary labels, shape (n,).
        temporal_feature_names: Names for temporal features (for importance report).
        static_feature_names: Names for static features (for importance report).
        output_dir: Where fold artifacts and the final summary are written.
        n_splits: Number of stratified folds.
        random_seed: Reproducibility seed.

    Returns:
        Dict with pooled out-of-fold metrics and per-fold breakdown.
    """
    output_path = ensure_directory(Path(output_dir))
    labels = labels.astype(int)

    X_temporal_flat = flatten_temporal(X_temporal)
    X = np.concatenate([X_static, X_temporal_flat], axis=1)

    if static_feature_names is not None and temporal_feature_names is not None:
        sequence_length = X_temporal.shape[1]
        flat_temporal_names = [
            f"{name}_t{step}"
            for step in range(sequence_length)
            for name in temporal_feature_names
        ]
        feature_names = list(static_feature_names) + flat_temporal_names
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    oof_probabilities = np.zeros(len(labels), dtype=float)
    fold_summaries: list[dict] = []
    importances = np.zeros(X.shape[1], dtype=float)

    for fold_index, (train_index, holdout_index) in enumerate(skf.split(X, labels), start=1):
        LOGGER.info("=== Fold %d/%d ===", fold_index, n_splits)

        X_train, y_train = X[train_index], labels[train_index]
        X_holdout, y_holdout = X[holdout_index], labels[holdout_index]

        positive_count = y_train.sum()
        negative_count = len(y_train) - positive_count
        scale_pos_weight = negative_count / max(positive_count, 1)
        LOGGER.info("Fold %d scale_pos_weight=%.2f", fold_index, scale_pos_weight)

        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=random_seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train, verbose=False)

        fold_probabilities = model.predict_proba(X_holdout)[:, 1]
        oof_probabilities[holdout_index] = fold_probabilities
        importances += model.feature_importances_

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

    best_threshold, best_f1 = find_best_threshold(labels, oof_probabilities)
    oof_predictions = (oof_probabilities >= best_threshold).astype(int)

    pooled_metrics = {
        "n_positives_total": int(labels.sum()),
        "n_total": len(labels),
        "selected_threshold": best_threshold,
        "accuracy": float(accuracy_score(labels, oof_predictions)),
        "precision": float(precision_score(labels, oof_predictions, zero_division=0)),
        "recall": float(recall_score(labels, oof_predictions, zero_division=0)),
        "f1": float(f1_score(labels, oof_predictions, zero_division=0)),
        "average_precision": float(average_precision_score(labels, oof_probabilities)),
        "roc_auc": float(roc_auc_score(labels, oof_probabilities)),
    }
    LOGGER.info("Pooled out-of-fold metrics: %s", pooled_metrics)

    report = classification_report(labels, oof_predictions, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(output_path / "oof_classification_report.csv")

    mean_importances = importances / n_splits
    importance_frame = pd.DataFrame(
        {"feature": feature_names, "importance": mean_importances}
    ).sort_values("importance", ascending=False)
    importance_frame.to_csv(output_path / "feature_importance.csv", index=False)

    np.save(output_path / "oof_probabilities.npy", oof_probabilities)
    np.save(output_path / "oof_true.npy", labels)

    summary = {"pooled_metrics": pooled_metrics, "fold_summaries": fold_summaries}
    (output_path / "kfold_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return summary


if __name__ == "__main__":
    raise RuntimeError(
        "Import run_xgboost_baseline(X_temporal, X_static, labels) and call it "
        "after building datasets, e.g. from a run_experiment script."
    )