"""Hyperparameter tuning for the logistic regression baseline.

Grid search over regularization strength and penalty type, scored on
average precision. Exhaustive grid is affordable here since logistic
regression trains in milliseconds even with many candidates.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import MODELS_DIR, RANDOM_SEED
from utils import ensure_directory
from XGBoost_baseline import flatten_temporal

LOGGER = logging.getLogger(__name__)

PARAM_GRID = [
    {
        # Block 1: Pure L2 Regularization (using lbfgs)
        "classifier__penalty": ["l2"],
        "classifier__l1_ratio": [0.0],
        "classifier__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "classifier__solver": ["lbfgs"],
    },
    {
        # Block 2: Pure L1 Regularization (using liblinear)
        "classifier__penalty": ["l1"],
        "classifier__l1_ratio": [1.0],
        "classifier__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "classifier__solver": ["liblinear"],
    },
    {
        # Block 3: ElasticNet Regularization (using saga)
        "classifier__penalty": ["elasticnet"],
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__l1_ratio": [0.25, 0.5, 0.75],
        "classifier__solver": ["saga"],
    },
]


def tune_logistic_regression(
    X_temporal: np.ndarray,
    X_static: np.ndarray,
    labels: np.ndarray,
    output_dir: str | Path = MODELS_DIR / "tuning" / "logistic_regression",
    n_splits: int = 5,
    random_seed: int = RANDOM_SEED,
) -> dict:
    """Run GridSearchCV over logistic regression hyperparameters.

    Args:
        X_temporal: Temporal input, shape (n, sequence_length, temporal_features).
        X_static: Static input, shape (n, static_features).
        labels: Binary labels, shape (n,).
        output_dir: Where results are written.
        n_splits: Number of stratified folds used inside the search.
        random_seed: Reproducibility seed.

    Returns:
        Dict with best_params, best_score, and the full results table path.
    """
    output_path = ensure_directory(Path(output_dir))
    labels = labels.astype(int)

    X = np.concatenate([X_static, flatten_temporal(X_temporal)], axis=1)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=10000,
                random_state=random_seed,
            ),
        ),
    ])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRID,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    LOGGER.info("Starting logistic regression grid search")
    search.fit(X, labels)

    LOGGER.info("Best average precision: %.4f", search.best_score_)
    LOGGER.info("Best params: %s", search.best_params_)

    results_frame = pd.DataFrame(search.cv_results_).sort_values(
        "mean_test_score", ascending=False
    )
    results_path = output_path / "logreg_search_results.csv"
    results_frame.to_csv(results_path, index=False)

    best_params_path = output_path / "logreg_best_params.json"
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
        "Import tune_logistic_regression(X_temporal, X_static, labels) and call "
        "it after building datasets, e.g. from a run_experiment script."
    )