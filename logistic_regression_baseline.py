"""Logistic regression baseline for the credit risk classification task.

Simplest possible baseline: linear model over the same flattened
temporal + static tabular representation used by the XGBoost baseline.
Evaluated with stratified k-fold CV and OOF-pooled metrics, matching
the other model scripts for a fair, apples-to-apples comparison.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import MODELS_DIR, RANDOM_SEED
from utils import ensure_directory

LOGGER = logging.getLogger(__name__)


def find_best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Find the probability threshold that maximizes F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, probabilities)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def flatten_temporal(X_temporal: np.ndarray) -> np.ndarray:
    """Flatten (n, sequence_length, features) into (n, sequence_length * features)."""
    return X_temporal.reshape(X_temporal.shape[0], -1)


def run_logistic_regression_baseline(
    X_temporal: np.ndarray,
    X_static: np.ndarray,
    labels: np.ndarray,
    temporal_feature_names: list[str] | None = None,
    static_feature_names: list[str] | None = None,
    output_dir: str | Path = MODELS_DIR / "logistic_regression_baseline",
    n_splits: int = 5,
    random_seed: int = RANDOM_SEED,
) -> dict:
    """Train and evaluate a logistic regression baseline with stratified k-fold CV.

    Args:
        X_temporal: Temporal input, shape (n, sequence_length, temporal_features).
        X_static: Static input, shape (n, static_features).
        labels: Binary labels, shape (n,).
        temporal_feature_names: Names for temporal features (for coefficient report).
        static_feature_names: Names for static features (for coefficient report).
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
    coefficients = np.zeros(X.shape[1], dtype=float)

    for fold_index, (train_index, holdout_index) in enumerate(skf.split(X, labels), start=1):
        LOGGER.info("=== Fold %d/%d ===", fold_index, n_splits)

        X_train, y_train = X[train_index], labels[train_index]
        X_holdout, y_holdout = X[holdout_index], labels[holdout_index]

        model = Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    C=1.0,
                    random_state=random_seed,
                ),
            ),
        ])
        model.fit(X_train, y_train)

        fold_probabilities = model.predict_proba(X_holdout)[:, 1]
        oof_probabilities[holdout_index] = fold_probabilities
        coefficients += model.named_steps["classifier"].coef_.ravel()

        fold_auc = (
            roc_auc_score(y_holdout, fold_probabilities)
            if len(np.unique(y_holdout)) > 1
            else float("nan")
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

    mean_coefficients = coefficients / n_splits
    coefficient_frame = pd.DataFrame(
        {"feature": feature_names, "coefficient": mean_coefficients}
    ).sort_values("coefficient", key=np.abs, ascending=False)
    coefficient_frame.to_csv(output_path / "coefficients.csv", index=False)

    np.save(output_path / "oof_probabilities.npy", oof_probabilities)
    np.save(output_path / "oof_true.npy", labels)

    summary = {"pooled_metrics": pooled_metrics, "fold_summaries": fold_summaries}
    (output_path / "kfold_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return summary


if __name__ == "__main__":
    raise RuntimeError(
        "Import run_logistic_regression_baseline(X_temporal, X_static, labels) "
        "and call it after building datasets, e.g. from a run_experiment script."
    )