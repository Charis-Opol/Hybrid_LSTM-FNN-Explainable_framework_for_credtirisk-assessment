#!/usr/bin/env python
"""Sweep imbalance-handling strategies for the tabular baselines (logistic
regression, XGBoost) on the master.csv dataset (~2.3% positive rate, ~70
positive borrowers of 3000).

Compares, for each model:
  - no_weight            plain fit, no class weighting or resampling
  - balanced (baseline)  the class_weight="balanced" / scale_pos_weight
                         setting already used elsewhere in this project
  - weight_0.5           half-strength class weighting
  - smote_0.3            SMOTE oversampling to a 0.3 minority:majority
                         ratio within each training fold, no extra weighting
  - smote_0.5            SMOTE oversampling to a 0.5 ratio
  - smote_0.3_weight_0.5 SMOTE to 0.3 plus half-strength weighting on top

SMOTE is always fit on the training fold only (never the holdout), pooled
out-of-fold predictions are used for all metrics, matching every other
script in this project.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from imbalance_sweep import class_weight_at_strength, pooled_metrics_from_oof, smote_resample
from run_experiment_evaluation_comparison import (
    DATA_PATH,
    build_training_arrays,
    load_and_normalize_dataset,
)
from config import RANDOM_SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent / "models" / "uganda_mobile_money_evaluation_comparison" / "imbalance_sweep"

CONFIGS = [
    {"name": "no_weight", "weight_strength": 0.0, "smote_ratio": None},
    {"name": "balanced (baseline)", "weight_strength": 1.0, "smote_ratio": None},
    {"name": "weight_0.5", "weight_strength": 0.5, "smote_ratio": None},
    {"name": "smote_0.3", "weight_strength": 0.0, "smote_ratio": 0.3},
    {"name": "smote_0.5", "weight_strength": 0.0, "smote_ratio": 0.5},
    {"name": "smote_0.3_weight_0.5", "weight_strength": 0.5, "smote_ratio": 0.3},
]


def flatten_inputs(X_temporal: np.ndarray, X_static: np.ndarray) -> np.ndarray:
    return np.concatenate([X_static, X_temporal.reshape(X_temporal.shape[0], -1)], axis=1)


def run_logreg_config(X: np.ndarray, y: np.ndarray, config: dict, n_splits: int, random_seed: int) -> np.ndarray:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    oof = np.zeros(len(y), dtype=float)

    for train_index, holdout_index in skf.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_holdout = X[holdout_index]

        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_holdout_scaled = scaler.transform(X_holdout)

        if config["smote_ratio"] is not None:
            X_train_scaled, y_train = smote_resample(X_train_scaled, y_train, config["smote_ratio"], random_seed)

        weight = class_weight_at_strength(y_train, config["weight_strength"])
        model = LogisticRegression(class_weight=weight, max_iter=2000, C=1.0, random_state=random_seed)
        model.fit(X_train_scaled, y_train)
        oof[holdout_index] = model.predict_proba(X_holdout_scaled)[:, 1]

    return oof


def run_xgboost_config(X: np.ndarray, y: np.ndarray, config: dict, n_splits: int, random_seed: int) -> np.ndarray:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    oof = np.zeros(len(y), dtype=float)

    for train_index, holdout_index in skf.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_holdout = X[holdout_index]

        if config["smote_ratio"] is not None:
            X_train, y_train = smote_resample(X_train, y_train, config["smote_ratio"], random_seed)

        weight = class_weight_at_strength(y_train, config["weight_strength"])
        scale_pos_weight = weight[1] / weight[0]
        model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
            random_state=random_seed, n_jobs=-1,
        )
        model.fit(X_train, y_train, verbose=False)
        oof[holdout_index] = model.predict_proba(X_holdout)[:, 1]

    return oof


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_and_normalize_dataset(DATA_PATH)
    data = data.sort_values(["borrower_id", "transaction_date"]).reset_index(drop=True)
    X_temporal, X_static, labels, *_ = build_training_arrays(data)
    X = flatten_inputs(X_temporal, X_static)
    y = labels.astype(int)
    LOGGER.info("n=%d, positives=%d (%.2f%%)", len(y), y.sum(), 100 * y.mean())

    rows = []
    for model_name, runner in [("logistic_regression", run_logreg_config), ("xgboost", run_xgboost_config)]:
        for config in CONFIGS:
            LOGGER.info("=== %s / %s ===", model_name, config["name"])
            oof = runner(X, y, config, n_splits=5, random_seed=RANDOM_SEED)
            metrics = pooled_metrics_from_oof(y, oof)
            metrics["model"] = model_name
            metrics["config"] = config["name"]
            rows.append(metrics)
            np.save(OUTPUT_DIR / f"{model_name}__{config['name']}__oof_probabilities.npy", oof)
            LOGGER.info(
                "  AP=%.4f AUC=%.4f precision@F1=%.4f recall@F1=%.4f precision@recall0.5=%.4f",
                metrics["average_precision"], metrics["roc_auc"],
                metrics["precision"], metrics["recall"], metrics["precision_at_recall_0.5"],
            )

    frame = pd.DataFrame(rows)
    frame = frame[[
        "model", "config", "average_precision", "roc_auc", "precision", "recall", "f1",
        "precision_at_recall_0.5", "n_flagged_at_recall_0.5", "selected_threshold",
    ]]
    frame.to_csv(OUTPUT_DIR / "tabular_sweep_results.csv", index=False)
    LOGGER.info("Saved sweep results to %s", OUTPUT_DIR / "tabular_sweep_results.csv")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
