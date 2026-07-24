"""Shared utilities for sweeping imbalance-handling strategies.

Given a very small positive class (~2.3%, ~70 borrowers in the master.csv
dataset), the F1-optimal threshold used elsewhere in this project tends to
land at a high-recall/low-precision operating point. This module isolates
whether a different imbalance-handling strategy -- SMOTE oversampling within
each training fold, or a tuned (rather than fully "balanced") class-weight
strength -- shifts the precision/recall trade-off at a fixed recall level,
versus just picking a different threshold on the same model.
"""

from __future__ import annotations

import logging

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

LOGGER = logging.getLogger(__name__)


def find_best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Find the probability threshold that maximizes F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, probabilities)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def precision_at_recall(y_true: np.ndarray, probabilities: np.ndarray, target_recall: float) -> tuple[float, float, int]:
    """Precision at the first threshold whose recall reaches ``target_recall``.

    Returns (precision, actual_recall_at_that_point, n_flagged).
    """
    order = np.argsort(-probabilities)
    y_sorted = y_true[order]
    total_positive = y_true.sum()
    cumulative_true_positive = np.cumsum(y_sorted)
    cumulative_n = np.arange(1, len(y_sorted) + 1)
    recall = cumulative_true_positive / total_positive
    index = min(int(np.searchsorted(recall, target_recall)), len(recall) - 1)
    precision = cumulative_true_positive[index] / cumulative_n[index]
    return float(precision), float(recall[index]), int(cumulative_n[index])


def class_weight_at_strength(y: np.ndarray, strength: float) -> dict[int, float]:
    """Interpolate between no weighting (strength=0) and full balanced weighting (strength=1).

    "Balanced" weighting (used everywhere in this project) sets the minority
    weight to n_negative / n_positive -- the theoretical optimum for
    maximizing *balanced accuracy*, not precision. Weaker weighting trades
    some recall back for precision.
    """
    positive_count = int(y.sum())
    negative_count = len(y) - positive_count
    full_balanced_weight = negative_count / max(positive_count, 1)
    positive_weight = 1.0 + strength * (full_balanced_weight - 1.0)
    return {0: 1.0, 1: positive_weight}


def smote_resample(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample the minority class with SMOTE up to ``sampling_strategy``
    (minority:majority ratio). Must only be called on a training fold, never
    on the holdout, or the reported metrics would leak synthetic neighbors
    of holdout points into training.
    """
    positive_count = int(y.sum())
    k_neighbors = max(1, min(5, positive_count - 1))
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=k_neighbors,
    )
    return smote.fit_resample(X, y)


def pooled_metrics_from_oof(oof_true: np.ndarray, oof_probabilities: np.ndarray) -> dict:
    best_threshold, _ = find_best_threshold(oof_true, oof_probabilities)
    oof_predictions = (oof_probabilities >= best_threshold).astype(int)
    precision_at_50, recall_at_50, flagged_at_50 = precision_at_recall(oof_true, oof_probabilities, 0.5)
    return {
        "n_positives_total": int(oof_true.sum()),
        "n_total": len(oof_true),
        "selected_threshold": best_threshold,
        "accuracy": float(accuracy_score(oof_true, oof_predictions)),
        "precision": float(precision_score(oof_true, oof_predictions, zero_division=0)),
        "recall": float(recall_score(oof_true, oof_predictions, zero_division=0)),
        "f1": float(f1_score(oof_true, oof_predictions, zero_division=0)),
        "average_precision": float(average_precision_score(oof_true, oof_probabilities)),
        "roc_auc": float(roc_auc_score(oof_true, oof_probabilities)) if len(np.unique(oof_true)) > 1 else float("nan"),
        "precision_at_recall_0.5": precision_at_50,
        "recall_at_recall_0.5": recall_at_50,
        "n_flagged_at_recall_0.5": flagged_at_50,
    }
