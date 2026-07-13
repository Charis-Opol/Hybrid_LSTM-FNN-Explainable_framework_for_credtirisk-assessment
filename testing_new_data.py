"""Transformer-encoder hybrid model: architecture + k-fold CV training.

Standard Transformer encoder block (Vaswani et al., 2017) applied to
the monthly transaction sequence -- learned positional embeddings +
multi-head self-attention + feed-forward sublayer, each with residual
connections and layer normalization, no recurrence at all. The static
branch and fusion (concatenation) mirror the original hybrid_model.py,
so this isolates the effect of recurrence (GRU) vs. pure attention.

This file contains both the model architecture and its stratified
k-fold cross-validation runner (pooled out-of-fold metrics), so it can
be trained and evaluated with a single import.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Union

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
import argparse
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

from config import MODELS_DIR, RANDOM_SEED
from feature_engineering import FeatureColumns, engineer_features
from hybrid_model import F1Score
from static_dataset import StaticDatasetBuilder
from temporal_dataset import TemporalDatasetBuilder
from utils import configure_logging, ensure_directory, set_random_seed

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Architecture
# --------------------------------------------------------------------------

def transformer_encoder_block(
    x: tf.Tensor,
    attention_mask: tf.Tensor,
    num_heads: int,
    key_dim: int,
    feed_forward_dim: int,
    dropout_rate: float,
    l2_reg: float,
    name: str,
) -> tf.Tensor:
    """One Transformer encoder block: self-attention + feed-forward, each
    with a residual connection and layer normalization.

    Args:
        x: Input sequence, shape (batch, sequence_length, model_dim).
        attention_mask: Explicit boolean mask broadcastable to
            (batch, sequence_length, sequence_length), True where a key
            position is valid (not padding).
        num_heads: Attention heads.
        key_dim: Key dimension per head.
        feed_forward_dim: Hidden size of the feed-forward sublayer.
        dropout_rate: Dropout applied after attention and after feed-forward.
        l2_reg: L2 regularization strength.
        name: Prefix for this block's layer names.
    """

    l2 = tf.keras.regularizers.l2(l2_reg)
    model_dim = x.shape[-1]

    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, name=f"{name}_self_attention"
    )(x, x, attention_mask=attention_mask)
    attention_output = tf.keras.layers.Dropout(dropout_rate)(attention_output)
    x = tf.keras.layers.Add()([x, attention_output])
    x = tf.keras.layers.LayerNormalization(name=f"{name}_attention_norm")(x)

    feed_forward = tf.keras.layers.Dense(
        feed_forward_dim, activation="relu", kernel_regularizer=l2
    )(x)
    feed_forward = tf.keras.layers.Dense(model_dim, kernel_regularizer=l2)(feed_forward)
    feed_forward = tf.keras.layers.Dropout(dropout_rate)(feed_forward)
    x = tf.keras.layers.Add()([x, feed_forward])
    x = tf.keras.layers.LayerNormalization(name=f"{name}_feedforward_norm")(x)

    return x


def build_transformer_hybrid_model(
    sequence_length: int,
    temporal_features: int,
    static_features: int,
    model_dim: int = 64,
    num_heads: int = 4,
    key_dim: int = 16,
    feed_forward_dim: int = 128,
    num_encoder_layers: int = 2,
    static_dense_units: int = 64,
    fusion_dense_units: int = 64,
    second_dense_units: int = 32,
    dropout_rate: float = 0.3,
    l2_reg: float = 1e-3,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """Build the Transformer-encoder hybrid model.

    Padding is handled with an explicit mask computed directly from the
    raw temporal input (True where any feature at that time step is
    nonzero), rather than Keras's automatic mask propagation -- the
    positional embedding tensor has no batch dimension, which breaks
    automatic mask propagation through Add/MultiHeadAttention/pooling.

    Args:
        sequence_length: Number of monthly time steps in the temporal input.
        temporal_features: Number of features per monthly time step.
        static_features: Number of static borrower features.
        model_dim: Dimension the raw temporal features are projected to
            before positional embeddings and the encoder stack.
        num_heads: Attention heads per encoder block.
        key_dim: Key dimension per attention head.
        feed_forward_dim: Hidden size of each encoder block's feed-forward
            sublayer.
        num_encoder_layers: Number of stacked Transformer encoder blocks.
        static_dense_units: Units in the static (FNN) branch's Dense layers.
        fusion_dense_units: Units in the first post-fusion Dense layer.
        second_dense_units: Units in the second post-fusion Dense layer.
        dropout_rate: Dropout applied throughout.
        l2_reg: L2 regularization strength applied to Dense layers.
        learning_rate: Adam optimizer learning rate.

    Returns:
        Compiled Keras model with AUC, precision, recall, accuracy, and F1.
    """
    l2 = tf.keras.regularizers.l2(l2_reg)

    temporal_input = tf.keras.Input(
        shape=(sequence_length, temporal_features), name="temporal_input"
    )

    # FIX: Explicitly passing 'tf' into arguments ensures the saved graph package 
    # has access to the module namespace when unpickled/loaded via load_model().
    padding_mask = tf.keras.layers.Lambda(
        lambda t, tf_backend: tf_backend.reduce_any(tf_backend.not_equal(t, 0.0), axis=-1),
        arguments={"tf_backend": tf},
        output_shape=(sequence_length,),
        name="compute_padding_mask",
    )(temporal_input)  # shape: (batch, sequence_length), boolean

    x = tf.keras.layers.Dense(model_dim, kernel_regularizer=l2, name="temporal_projection")(
        temporal_input
    )

    positions = tf.range(start=0, limit=sequence_length, delta=1)
    position_embeddings = tf.keras.layers.Embedding(
        input_dim=sequence_length, output_dim=model_dim, name="positional_embedding"
    )(positions)
    
    # FIX: Native addition element-wise avoids dynamic batch shape inferencing breakdown
    x = x + position_embeddings

    # FIX: Using pure Python slicing 'None' instead of 'tf.newaxis' eliminates 
    # external runtime dependency errors inside the Lambda serialization boundaries.
    attention_mask = tf.keras.layers.Lambda(
        lambda m: m[:, None, :],
        output_shape=(1, sequence_length),
        name="expand_attention_mask",
    )(padding_mask)

    for layer_index in range(num_encoder_layers):
        x = transformer_encoder_block(
            x,
            attention_mask=attention_mask,
            num_heads=num_heads,
            key_dim=key_dim,
            feed_forward_dim=feed_forward_dim,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            name=f"encoder_block_{layer_index + 1}",
        )

    # Manual masked average pooling (padded time steps excluded explicitly,
    # rather than relying on GlobalAveragePooling1D's automatic masking).
    pooled = tf.keras.layers.Lambda(
        lambda inputs, tf_backend: (
            tf_backend.reduce_sum(inputs[0] * tf_backend.cast(inputs[1], tf_backend.float32)[..., tf_backend.newaxis], axis=1)
            / tf_backend.maximum(
                tf_backend.reduce_sum(tf_backend.cast(inputs[1], tf_backend.float32)[..., tf_backend.newaxis], axis=1), 1e-6
            )
        ),
        arguments={"tf_backend": tf},
        output_shape=(model_dim,),
        name="masked_global_average_pool",
    )([x, padding_mask])

    temporal_embedding = tf.keras.layers.Dense(
        model_dim, activation="relu", kernel_regularizer=l2, name="temporal_embedding"
    )(pooled)

    static_input = tf.keras.Input(shape=(static_features,), name="static_input")
    s = tf.keras.layers.Dense(
        static_dense_units, activation="relu", kernel_regularizer=l2
    )(static_input)
    s = tf.keras.layers.BatchNormalization()(s)
    s = tf.keras.layers.Dropout(dropout_rate)(s)
    static_embedding = tf.keras.layers.Dense(
        static_dense_units, activation="relu", kernel_regularizer=l2, name="static_embedding"
    )(s)

    fused = tf.keras.layers.Concatenate(name="hybrid_embedding")(
        [temporal_embedding, static_embedding]
    )
    f = tf.keras.layers.Dense(
        fusion_dense_units, activation="relu", kernel_regularizer=l2
    )(fused)
    f = tf.keras.layers.BatchNormalization()(f)
    f = tf.keras.layers.Dropout(dropout_rate)(f)
    f = tf.keras.layers.Dense(
        second_dense_units, activation="relu", kernel_regularizer=l2
    )(f)
    f = tf.keras.layers.BatchNormalization()(f)
    f = tf.keras.layers.Dropout(max(dropout_rate - 0.1, 0.1))(f)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="risk_score")(f)

    model = tf.keras.Model(
        inputs=[temporal_input, static_input],
        outputs=output,
        name="transformer_encoder_hybrid_model",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            F1Score(name="f1"),
        ],
    )
    return model


# --------------------------------------------------------------------------
# Stratified k-fold cross-validation (pooled out-of-fold metrics)
# --------------------------------------------------------------------------

def find_best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Find the probability threshold that maximizes F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, probabilities)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def run_kfold_cv_transformer(
    X_temporal: np.ndarray,
    X_static: np.ndarray,
    labels: np.ndarray,
    output_dir: Union[str, Path] = MODELS_DIR / "kfold_transformer",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run transformer encoder hybrid k-fold CV on a labeled borrower dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data") / "raw" / "uganda_mobile_money_master.csv",
        help="Path to the raw transaction dataset (CSV or Excel).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODELS_DIR / "uganda_mobile_money_test_new_data",
        help="Directory for model artifacts and summaries.",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Number of stratified folds.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs per fold.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    return parser.parse_args()


def _normalize_column_names(data: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "timestamp": "transaction_date",
        "amount": "transaction_amount",
        "balance_after": "balance",
        "default_label": "defaulted",
        "loan_amount": "loan_amount",
        "sacco_member": "sacco_membership",
        "network": "preferred_network",
        "channel": "preferred_channel",
    }
    existing = {k: v for k, v in rename_map.items() if k in data.columns}
    if existing:
        data = data.rename(columns=existing)
    return data


def load_and_normalize_dataset(path: Path) -> pd.DataFrame:
    LOGGER.info("Loading dataset from %s", path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        data = pd.read_excel(path)
    else:
        data = pd.read_csv(path)

    data = _normalize_column_names(data)
    if "transaction_date" in data.columns:
        data["transaction_date"] = pd.to_datetime(
            data["transaction_date"], errors="coerce"
        )
    return data


def build_training_arrays(data: pd.DataFrame):
    feature_columns = FeatureColumns(
        amount="transaction_amount",
        balance="balance",
        target="defaulted",
    )
    engineered = engineer_features(data, columns=feature_columns)
    temporal = TemporalDatasetBuilder().build(engineered)
    static = StaticDatasetBuilder().build(engineered)

    static_by_borrower = {
        borrower_id: row_index
        for row_index, borrower_id in enumerate(static.borrower_ids.tolist())
    }
    static_indices = [
        static_by_borrower[borrower_id]
        for borrower_id in temporal.borrower_ids.tolist()
    ]
    X_static_aligned = static.X_static[static_indices]

    if temporal.labels is None:
        raise ValueError("The dataset must include default labels for training.")

    return (
        temporal.X_temporal,
        X_static_aligned,
        temporal.labels,
        engineered,
        temporal.feature_names,
        static.feature_names,
    )


def main() -> None:
    configure_logging()
    args = parse_args()

    args.output_dir = ensure_directory(args.output_dir)
    data = load_and_normalize_dataset(args.data)
    data = data.sort_values(["borrower_id", "transaction_date"]).reset_index(drop=True)

    X_temporal, X_static, labels, engineered, temporal_names, static_names = (
        build_training_arrays(data)
    )

    logging.info(
        "Built datasets: %d borrowers, %d temporal features/month, %d static features",
        len(labels), X_temporal.shape[2], X_static.shape[1],
    )

    unique_labels, label_counts = np.unique(labels, return_counts=True)
    logging.info(
        "Label distribution: %s",
        dict(zip(unique_labels.tolist(), label_counts.tolist())),
    )

    engineered.to_csv(args.output_dir / "engineered_features.csv", index=False)
    (args.output_dir / "temporal_feature_names.txt").write_text(
        "\n".join(temporal_names), encoding="utf-8"
    )
    (args.output_dir / "static_feature_names.txt").write_text(
        "\n".join(static_names), encoding="utf-8"
    )

    summary = run_kfold_cv_transformer(
        X_temporal=X_temporal,
        X_static=X_static,
        labels=labels,
        output_dir=args.output_dir / "kfold_transformer",
        n_splits=args.n_splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    logging.info("Finished transformer k-fold CV. Summary: %s", summary)


if __name__ == "__main__":
    main()