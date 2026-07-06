"""Hyperparameter tuning for the hybrid LSTM-GRU-Attention-FNN model.

Full 5-fold CV per candidate would mean training N_TRIALS x 5 models --
too expensive for a deep model. Instead this runs a random search over
a single stratified train/validation split (with early stopping), then
you re-validate only the winning configuration with the existing
kfold_cv_hybrid.py script for a trustworthy final number.

Architecture mirrors fnn_encoder.py / lstm_encoder.py / hybrid_model.py
but exposes the key structural choices as tunable hyperparameters
instead of hardcoded constants.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from config import MODELS_DIR, RANDOM_SEED
from utils import ensure_directory, set_random_seed

LOGGER = logging.getLogger(__name__)

SEARCH_SPACE = {
    "gru_units": [32, 64, 96, 128],
    "attention_heads": [1, 2, 4],
    "attention_key_dim": [16, 32, 64],
    "static_dense_units": [32, 64, 96],
    "fusion_dense_units": [32, 64, 96, 128],
    "second_dense_units": [16, 32, 64],
    "dropout_rate": [0.2, 0.3, 0.4, 0.5],
    "l2_reg": [1e-4, 1e-3, 1e-2],
    "learning_rate": [1e-4, 5e-4, 1e-3, 2e-3],
    "batch_size": [32, 64, 128],
}


def sample_config(rng: random.Random) -> dict:
    """Sample one random hyperparameter configuration from SEARCH_SPACE."""
    return {key: rng.choice(values) for key, values in SEARCH_SPACE.items()}


def build_tunable_hybrid_model(
    sequence_length: int,
    temporal_features: int,
    static_features: int,
    config: dict,
) -> tf.keras.Model:
    """Build the hybrid model with hyperparameters from `config`."""

    l2 = tf.keras.regularizers.l2(config["l2_reg"])

    # Temporal branch: Masking -> GRU -> MultiHeadAttention -> pooling -> Dense
    temporal_input = tf.keras.Input(
        shape=(sequence_length, temporal_features), name="temporal_input"
    )
    x = tf.keras.layers.Masking(mask_value=0.0)(temporal_input)
    x = tf.keras.layers.GRU(
        config["gru_units"],
        return_sequences=True,
        dropout=config["dropout_rate"],
        recurrent_dropout=0.2,
    )(x)
    attention = tf.keras.layers.MultiHeadAttention(
        num_heads=config["attention_heads"],
        key_dim=config["attention_key_dim"],
    )(x, x)
    attention = tf.keras.layers.Add()([x, attention])
    attention = tf.keras.layers.LayerNormalization()(attention)
    attention = tf.keras.layers.GlobalAveragePooling1D()(attention)
    temporal_embedding = tf.keras.layers.Dense(
        config["gru_units"], activation="relu", kernel_regularizer=l2
    )(attention)

    # Static branch: Dense -> BatchNorm -> Dropout -> Dense
    static_input = tf.keras.Input(shape=(static_features,), name="static_input")
    s = tf.keras.layers.Dense(
        config["static_dense_units"], activation="relu", kernel_regularizer=l2
    )(static_input)
    s = tf.keras.layers.BatchNormalization()(s)
    s = tf.keras.layers.Dropout(config["dropout_rate"])(s)
    static_embedding = tf.keras.layers.Dense(
        config["static_dense_units"], activation="relu", kernel_regularizer=l2
    )(s)

    # Fusion head
    fused = tf.keras.layers.Concatenate()([temporal_embedding, static_embedding])
    f = tf.keras.layers.Dense(
        config["fusion_dense_units"], activation="relu", kernel_regularizer=l2
    )(fused)
    f = tf.keras.layers.BatchNormalization()(f)
    f = tf.keras.layers.Dropout(config["dropout_rate"])(f)
    f = tf.keras.layers.Dense(
        config["second_dense_units"], activation="relu", kernel_regularizer=l2
    )(f)
    f = tf.keras.layers.BatchNormalization()(f)
    f = tf.keras.layers.Dropout(max(config["dropout_rate"] - 0.1, 0.1))(f)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="risk_score")(f)

    model = tf.keras.Model(inputs=[temporal_input, static_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


def tune_hybrid_model(
    X_temporal: np.ndarray,
    X_static: np.ndarray,
    labels: np.ndarray,
    output_dir: str | Path = MODELS_DIR / "tuning" / "hybrid",
    n_trials: int = 20,
    epochs: int = 60,
    random_seed: int = RANDOM_SEED,
) -> dict:
    """Random search over hybrid model hyperparameters on a single split.

    Args:
        X_temporal: Temporal input, shape (n, sequence_length, temporal_features).
        X_static: Static input, shape (n, static_features).
        labels: Binary labels, shape (n,).
        output_dir: Where trial results and the best config are written.
        n_trials: Number of random configurations to try.
        epochs: Max epochs per trial (early stopping applies).
        random_seed: Reproducibility seed.

    Returns:
        Dict with best_config, best_score, and the full trial results table.
    """
    set_random_seed(random_seed)
    output_path = ensure_directory(Path(output_dir))
    labels = labels.astype(float)
    rng = random.Random(random_seed)

    stratify = labels if len(np.unique(labels)) > 1 else None
    X_temp_train, X_temp_val, X_static_train, X_static_val, y_train, y_val = train_test_split(
        X_temporal, X_static, labels,
        test_size=0.2, random_state=random_seed, stratify=stratify,
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train.astype(int)),
        y=y_train.astype(int),
    )
    class_weight = {
        int(cls): float(weight)
        for cls, weight in zip(np.unique(y_train.astype(int)), class_weights)
    }

    trial_results: list[dict] = []
    best_score = -np.inf
    best_config: dict | None = None

    for trial_index in range(1, n_trials + 1):
        config = sample_config(rng)
        LOGGER.info("=== Trial %d/%d: %s ===", trial_index, n_trials, config)

        tf.keras.backend.clear_session()
        model = build_tunable_hybrid_model(
            sequence_length=X_temporal.shape[1],
            temporal_features=X_temporal.shape[2],
            static_features=X_static.shape[1],
            config=config,
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", patience=8, mode="max", restore_best_weights=True
            ),
        ]

        try:
            model.fit(
                [X_temp_train, X_static_train],
                y_train,
                validation_data=([X_temp_val, X_static_val], y_val),
                epochs=epochs,
                batch_size=config["batch_size"],
                class_weight=class_weight,
                callbacks=callbacks,
                verbose=0,
            )
            val_probabilities = model.predict([X_temp_val, X_static_val], verbose=0).ravel()
            val_auc = roc_auc_score(y_val, val_probabilities)
            val_ap = average_precision_score(y_val, val_probabilities)
        except Exception as exc:  # noqa: BLE001 - log and continue the search
            LOGGER.warning("Trial %d failed: %s", trial_index, exc)
            val_auc, val_ap = float("nan"), float("nan")

        LOGGER.info("Trial %d: val_auc=%.4f, val_ap=%.4f", trial_index, val_auc, val_ap)
        trial_results.append({**config, "trial": trial_index, "val_auc": val_auc, "val_ap": val_ap})

        # Rank trials by average precision -- more informative than AUC given
        # the severe class imbalance.
        if not np.isnan(val_ap) and val_ap > best_score:
            best_score = val_ap
            best_config = config

    results_frame = pd.DataFrame(trial_results).sort_values("val_ap", ascending=False)
    results_path = output_path / "hybrid_search_results.csv"
    results_frame.to_csv(results_path, index=False)

    best_config_path = output_path / "hybrid_best_config.json"
    best_config_path.write_text(
        json.dumps({"best_config": best_config, "best_val_ap": best_score}, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Best val AP: %.4f, config: %s", best_score, best_config)

    return {
        "best_config": best_config,
        "best_score": float(best_score),
        "results_path": results_path,
        "best_config_path": best_config_path,
    }


if __name__ == "__main__":
    raise RuntimeError(
        "Import tune_hybrid_model(X_temporal, X_static, labels) and call it "
        "after building datasets, e.g. from a run_experiment script."
    )