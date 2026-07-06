"""Hyperparameter tuning for the XGBoost baseline.

Uses RandomizedSearchCV with stratified k-fold, scored on average
precision (more informative than accuracy or ROC-AUC for a ~2.3%
positive rate). Saves the best parameters and a full results table.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from config import MODELS_DIR, RANDOM_SEED
from utils import ensure_directory
from XGBoost_baseline import flatten_temporal

LOGGER = logging.getLogger(__name__)

PARAM_DISTRIBUTIONS = {
    "n_estimators": randint(100, 600),
    "max_depth": randint(2, 6),
    "learning_rate": uniform(0.01, 0.19),       # 0.01 - 0.20
    "subsample": uniform(0.6, 0.4),              # 0.6 - 1.0
    "colsample_bytree": uniform(0.6, 0.4),        # 0.6 - 1.0
    "min_child_weight": randint(1, 10),
    "gamma": uniform(0.0, 0.5),
    "reg_alpha": uniform(0.0, 1.0),
    "reg_lambda": uniform(0.5, 2.5),
}


def tune_xgboost(
    X_temporal: np.ndarray,
    X_static: np.ndarray,
    labels: np.ndarray,
    output_dir: str | Path = MODELS_DIR / "tuning" / "xgboost",
    n_iter: int = 60,
    n_splits: int = 5,
    random_seed: int = RANDOM_SEED,
) -> dict:
    """Run RandomizedSearchCV over XGBoost hyperparameters.

    Args:
        X_temporal: Temporal input, shape (n, sequence_length, temporal_features).
        X_static: Static input, shape (n, static_features).
        labels: Binary labels, shape (n,).
        output_dir: Where results are written.
        n_iter: Number of random parameter combinations to try.
        n_splits: Number of stratified folds used inside the search.
        random_seed: Reproducibility seed.

    Returns:
        Dict with best_params, best_score, and the full results table path.
    """
    output_path = ensure_directory(Path(output_dir))
    labels = labels.astype(int)

    X = np.concatenate([X_static, flatten_temporal(X_temporal)], axis=1)

    positive_count = labels.sum()
    negative_count = len(labels) - positive_count
    scale_pos_weight = negative_count / max(positive_count, 1)

    base_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=random_seed,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv,
        random_state=random_seed,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    LOGGER.info("Starting XGBoost random search: %d candidates x %d folds", n_iter, n_splits)
    search.fit(X, labels)

    LOGGER.info("Best average precision: %.4f", search.best_score_)
    LOGGER.info("Best params: %s", search.best_params_)

    results_frame = pd.DataFrame(search.cv_results_).sort_values(
        "mean_test_score", ascending=False
    )
    results_path = output_path / "xgboost_search_results.csv"
    results_frame.to_csv(results_path, index=False)

    best_params_path = output_path / "xgboost_best_params.json"
    best_params_path.write_text(
        json.dumps({"best_params": search.best_params_, "best_score": search.best_score_}, indent=2),
        encoding="utf-8",
    )

    return {
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "results_path": results_path,
        "best_params_path": best_params_path,
    }


if __name__ == "__main__":
    raise RuntimeError(
        "Import tune_xgboost(X_temporal, X_static, labels) and call it after "
        "building datasets, e.g. from a run_experiment script."
    )