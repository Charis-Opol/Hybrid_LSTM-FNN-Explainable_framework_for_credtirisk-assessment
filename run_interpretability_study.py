#!/usr/bin/env python
"""Interpretability study for the hybrid GRU-attention model and the
Transformer-encoder hybrid model, on the master.csv dataset.

Trains the attention-exposing twins from attention_visualization.py
directly (loss on the risk_score output only; attention weights and the
pre-output fused embedding are free extra outputs with no loss term),
using the same stratified 5-fold split, class weighting, and early
stopping as the production training scripts. Pools attention weights and
embeddings out-of-fold (every borrower interpreted by the fold that held
it out), then renders:

  - attention_by_month_{model}.png     mean attention received per month,
                                        defaulters vs. non-defaulters
  - attention_heatmap_{model}.png      per-borrower attention-over-months,
                                        sampled defaulters vs. non-defaulters
  - attention_matrix_{model}.png       full month-x-month attention,
                                        averaged over borrowers, per label
  - embedding_pca_{model}.png          2D PCA of the fused embedding,
                                        colored by true label

Training this twin is an independently-initialized instance of the same
architecture, not the canonical benchmark model -- its own pooled OOF
metrics are logged and saved so any drift from the benchmark numbers is
visible and auditable, but the canonical kfold_hybrid/kfold_transformer
results already reported are untouched.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

from attention_visualization import (
    build_hybrid_model_with_attention,
    build_transformer_hybrid_model_with_attention,
)
from config import RANDOM_SEED
from imbalance_sweep import pooled_metrics_from_oof
from run_experiment_evaluation_comparison import (
    DATA_PATH,
    OUTPUT_DIR,
    build_training_arrays,
    load_and_normalize_dataset,
)
from utils import ensure_directory, set_random_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

INTERP_DIR = OUTPUT_DIR / "interpretability"
N_SPLITS = 5


def train_oof_with_attention(model_name, build_fn, X_temporal, X_static, labels, n_layers):
    """Train the attention-exposing twin with stratified 5-fold CV, pooling
    OOF risk scores, attention weights (one array per encoder layer), and
    fused embeddings.
    """
    output_dir = ensure_directory(INTERP_DIR / model_name)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    n = len(labels)
    sequence_length = X_temporal.shape[1]
    oof_risk = np.zeros(n, dtype=float)
    oof_attention = [np.zeros((n, sequence_length, sequence_length), dtype=float) for _ in range(n_layers)]
    oof_fused = np.zeros((n, 128), dtype=float)

    for fold_index, (train_index, holdout_index) in enumerate(skf.split(X_static, labels), start=1):
        LOGGER.info("=== %s fold %d/%d ===", model_name, fold_index, N_SPLITS)
        set_random_seed(RANDOM_SEED)

        X_temp_train, X_static_train, y_train = X_temporal[train_index], X_static[train_index], labels[train_index]
        X_temp_holdout, X_static_holdout = X_temporal[holdout_index], X_static[holdout_index]

        stratify = y_train if len(np.unique(y_train)) > 1 else None
        (X_temp_tr, X_temp_es, X_static_tr, X_static_es, y_tr, y_es) = train_test_split(
            X_temp_train, X_static_train, y_train, test_size=0.15, random_state=RANDOM_SEED, stratify=stratify,
        )

        model = build_fn(sequence_length=sequence_length, temporal_features=X_temporal.shape[2], static_features=X_static.shape[1])
        n_outputs = len(model.output_names)
        loss = [tf.keras.losses.BinaryCrossentropy()] + [None] * (n_outputs - 1)
        metrics = [[tf.keras.metrics.AUC(name="auc")]] + [None] * (n_outputs - 1)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=loss, metrics=metrics)

        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_tr.astype(int)), y=y_tr.astype(int))
        class_weight = {int(cls): float(weight) for cls, weight in zip(np.unique(y_tr.astype(int)), class_weights)}

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_risk_score_auc", patience=8, mode="max", restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        ]
        model.fit(
            [X_temp_tr, X_static_tr], y_tr,
            validation_data=([X_temp_es, X_static_es], y_es),
            epochs=100, batch_size=128, class_weight=class_weight, callbacks=callbacks, verbose=0,
        )

        outputs = model.predict([X_temp_holdout, X_static_holdout], verbose=0)
        risk, attention_layers, fused = outputs[0], outputs[1:1 + n_layers], outputs[-1]
        oof_risk[holdout_index] = risk.ravel()
        oof_fused[holdout_index] = fused
        for layer_index in range(n_layers):
            oof_attention[layer_index][holdout_index] = attention_layers[layer_index].mean(axis=1)  # average over heads

        fold_dir = ensure_directory(output_dir / f"fold_{fold_index}")
        model.save(fold_dir / "trained_model_with_attention.keras")

    metrics = pooled_metrics_from_oof(labels, oof_risk)
    LOGGER.info("%s (interpretability twin) pooled OOF metrics: %s", model_name, metrics)

    np.save(output_dir / "oof_risk.npy", oof_risk)
    np.save(output_dir / "oof_true.npy", labels)
    for layer_index in range(n_layers):
        np.save(output_dir / f"oof_attention_layer{layer_index + 1}.npy", oof_attention[layer_index])
    np.save(output_dir / "oof_fused_embedding.npy", oof_fused)

    return oof_risk, oof_attention, oof_fused, metrics


def plot_attention_by_month(model_name, oof_attention, labels, n_layers, output_dir):
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 4.5), squeeze=False)
    axes = axes[0]
    months = np.arange(1, oof_attention[0].shape[1] + 1)

    for layer_index in range(n_layers):
        received = oof_attention[layer_index].mean(axis=1)  # (n, seq_len): avg over query positions
        defaulters = received[labels == 1].mean(axis=0)
        non_defaulters = received[labels == 0].mean(axis=0)

        axis = axes[layer_index]
        axis.plot(months, non_defaulters, marker="o", label="Non-defaulters", color="#4C72B0")
        axis.plot(months, defaulters, marker="o", label="Defaulters", color="#C44E52")
        axis.set_xlabel("Month index (1 = oldest in 12-month window, 12 = most recent)")
        axis.set_ylabel("Mean attention received")
        title = f"{model_name}" if n_layers == 1 else f"{model_name} — encoder layer {layer_index + 1}"
        axis.set_title(title)
        axis.legend()
        axis.grid(alpha=0.3)

    fig.suptitle(f"Attention-by-month profile: {model_name}")
    fig.tight_layout()
    path = output_dir / f"attention_by_month_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved %s", path)


def plot_attention_heatmap(model_name, oof_attention, labels, oof_risk, n_layers, output_dir, n_per_group=10):
    layer_to_show = n_layers - 1  # last encoder layer / only layer
    received = oof_attention[layer_to_show].mean(axis=1)  # (n, seq_len)

    positive_index = np.where(labels == 1)[0]
    negative_index = np.where(labels == 0)[0]
    top_positive = positive_index[np.argsort(-oof_risk[positive_index])[:n_per_group]]
    top_negative = negative_index[np.argsort(oof_risk[negative_index])[:n_per_group]]
    selected = np.concatenate([top_positive, top_negative])
    group_labels = ["Defaulter"] * len(top_positive) + ["Non-defaulter"] * len(top_negative)

    fig, axis = plt.subplots(figsize=(8, 0.35 * len(selected) + 2))
    image = axis.imshow(received[selected], aspect="auto", cmap="viridis")
    axis.set_yticks(range(len(selected)))
    axis.set_yticklabels([f"{group_labels[i]} #{i}" for i in range(len(selected))], fontsize=7)
    axis.set_xlabel("Month index (1 = oldest, 12 = most recent)")
    axis.set_title(f"Per-borrower attention received per month: {model_name}\n(top-risk defaulters vs. lowest-risk non-defaulters)")
    fig.colorbar(image, ax=axis, label="Mean attention received")
    fig.tight_layout()
    path = output_dir / f"attention_heatmap_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved %s", path)


def plot_attention_matrix(model_name, oof_attention, labels, n_layers, output_dir):
    fig, axes = plt.subplots(n_layers, 2, figsize=(9, 4 * n_layers), squeeze=False)

    for layer_index in range(n_layers):
        for column, (mask, group_name) in enumerate([(labels == 0, "Non-defaulters"), (labels == 1, "Defaulters")]):
            matrix = oof_attention[layer_index][mask].mean(axis=0)
            axis = axes[layer_index][column]
            image = axis.imshow(matrix, cmap="magma")
            axis.set_xlabel("Key month")
            axis.set_ylabel("Query month")
            layer_label = "" if n_layers == 1 else f" — layer {layer_index + 1}"
            axis.set_title(f"{model_name}{layer_label}: {group_name}")
            fig.colorbar(image, ax=axis, fraction=0.046)

    fig.tight_layout()
    path = output_dir / f"attention_matrix_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved %s", path)


def plot_embedding_pca(model_name, oof_fused, labels, output_dir):
    coordinates = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(oof_fused)

    fig, axis = plt.subplots(figsize=(6, 5))
    for label_value, color, name in [(0, "#4C72B0", "Non-defaulter"), (1, "#C44E52", "Defaulter")]:
        mask = labels == label_value
        axis.scatter(coordinates[mask, 0], coordinates[mask, 1], s=14 if label_value == 0 else 28,
                     alpha=0.5 if label_value == 0 else 0.9, color=color, label=name)
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.set_title(f"Fused embedding (PCA): {model_name}")
    axis.legend()
    axis.grid(alpha=0.3)
    fig.tight_layout()
    path = output_dir / f"embedding_pca_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved %s", path)


def main() -> None:
    ensure_directory(INTERP_DIR)

    data = load_and_normalize_dataset(DATA_PATH)
    data = data.sort_values(["borrower_id", "transaction_date"]).reset_index(drop=True)
    X_temporal, X_static, labels, *_ = build_training_arrays(data)
    labels = labels.astype(int)
    LOGGER.info("n=%d, positives=%d (%.2f%%)", len(labels), labels.sum(), 100 * labels.mean())

    for model_name, build_fn, n_layers in [
        ("hybrid", build_hybrid_model_with_attention, 1),
        ("transformer", build_transformer_hybrid_model_with_attention, 2),
    ]:
        oof_risk, oof_attention, oof_fused, metrics = train_oof_with_attention(
            model_name, build_fn, X_temporal, X_static, labels, n_layers,
        )
        plot_attention_by_month(model_name, oof_attention, labels, n_layers, INTERP_DIR)
        plot_attention_heatmap(model_name, oof_attention, labels, oof_risk, n_layers, INTERP_DIR)
        plot_attention_matrix(model_name, oof_attention, labels, n_layers, INTERP_DIR)
        plot_embedding_pca(model_name, oof_fused, labels, INTERP_DIR)

    LOGGER.info("Interpretability study complete. Artifacts in %s", INTERP_DIR)


if __name__ == "__main__":
    main()
