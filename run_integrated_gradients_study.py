#!/usr/bin/env python
"""Integrated Gradients study for the hybrid and transformer models on
master.csv, using the interpretability-twin checkpoints already trained
in run_interpretability_study.py (same stratified 5-fold split, so each
borrower is explained by the fold model that held it out).

Baseline is the training fold's own per-feature mean ("average borrower"),
not zero -- an all-zero temporal sequence fully masks every time step
(Masking(mask_value=0.0)), and GlobalAveragePooling1D over zero valid
steps divides by zero, producing NaN attributions. This was verified
directly against the trained model before settling on the mean baseline.

Explains all 70 positive borrowers (the whole minority class) plus a
random sample of negatives, matching the subsampling approach already
used for the SHAP explainer -- full-dataset attribution here would mean
~3000 borrowers x 51 forward/backward passes per model, unnecessary for
what this study is trying to show.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

import attention_visualization  # noqa: F401  registers PaddingMaskLayer/ExpandMaskLayer/MaskedGlobalAveragePooling1D for model loading
from config import RANDOM_SEED
from integrated_gradients import integrated_gradients_batch
from run_experiment_evaluation_comparison import (
    DATA_PATH,
    OUTPUT_DIR,
    build_training_arrays,
    load_and_normalize_dataset,
)
from utils import ensure_directory

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

INTERP_DIR = OUTPUT_DIR / "interpretability"
IG_DIR = OUTPUT_DIR / "integrated_gradients"
N_SPLITS = 5
M_STEPS = 50
N_NEGATIVE_SAMPLE = 200
# Predictions are sigmoid-bounded to [0, 1], so a well-behaved convergence
# delta (sum of attributions vs. actual F(x) - F(baseline)) should be a
# small fraction of that range. A handful of borrowers with extreme raw
# feature values (temporal inputs aren't scaled before the GRU) push the
# interpolation path through a numerically unstable region and produce
# wildly inflated gradients; those explanations are discarded rather than
# silently averaged into the aggregate plots.
RELIABLE_DELTA_THRESHOLD = 1.0


def run_ig_study(model_name, X_temporal, X_static, labels, temporal_feature_names, static_feature_names, sample_index):
    output_dir = ensure_directory(IG_DIR / model_name)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    sample_mask = np.zeros(len(labels), dtype=bool)
    sample_mask[sample_index] = True

    sequence_length = X_temporal.shape[1]
    ig_temporal = np.full((len(labels), sequence_length, X_temporal.shape[2]), np.nan, dtype=np.float32)
    ig_static = np.full((len(labels), X_static.shape[1]), np.nan, dtype=np.float32)
    convergence_deltas = np.full(len(labels), np.nan, dtype=np.float32)

    for fold_index, (train_index, holdout_index) in enumerate(skf.split(X_static, labels), start=1):
        explain_index = np.array([i for i in holdout_index if sample_mask[i]])
        if len(explain_index) == 0:
            continue
        LOGGER.info("=== %s fold %d/%d: explaining %d borrowers ===", model_name, fold_index, N_SPLITS, len(explain_index))

        model_path = INTERP_DIR / model_name / f"fold_{fold_index}" / "trained_model_with_attention.keras"
        model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)

        baseline_temporal = X_temporal[train_index].mean(axis=0)
        baseline_static = X_static[train_index].mean(axis=0)

        temporal_attr, static_attr, deltas = integrated_gradients_batch(
            model, X_temporal[explain_index], X_static[explain_index],
            baseline_temporal, baseline_static, m_steps=M_STEPS,
        )
        ig_temporal[explain_index] = temporal_attr
        ig_static[explain_index] = static_attr
        convergence_deltas[explain_index] = deltas

    np.save(output_dir / "ig_temporal.npy", ig_temporal)
    np.save(output_dir / "ig_static.npy", ig_static)
    np.save(output_dir / "convergence_deltas.npy", convergence_deltas)

    computed = ~np.isnan(convergence_deltas)
    valid = computed & (np.abs(convergence_deltas) < RELIABLE_DELTA_THRESHOLD)
    n_discarded = int(computed.sum() - valid.sum())
    LOGGER.info(
        "%s: explained %d borrowers, median |convergence_delta|=%.5f, discarded %d as numerically unreliable (|delta|>=%.1f)",
        model_name, computed.sum(), np.median(np.abs(convergence_deltas[computed])), n_discarded, RELIABLE_DELTA_THRESHOLD,
    )

    plot_feature_importance(model_name, ig_temporal, ig_static, temporal_feature_names, static_feature_names, valid, output_dir)
    plot_attribution_by_month(model_name, ig_temporal, labels, valid, output_dir)
    write_local_explanations(model_name, ig_temporal, ig_static, temporal_feature_names, static_feature_names, valid, labels, output_dir)

    return ig_temporal, ig_static, convergence_deltas


def plot_feature_importance(model_name, ig_temporal, ig_static, temporal_names, static_names, valid, output_dir, top_n=20):
    temporal_importance = np.abs(ig_temporal[valid]).mean(axis=(0, 1))  # averaged over borrowers and months
    static_importance = np.abs(ig_static[valid]).mean(axis=0)

    names = list(temporal_names) + list(static_names)
    importances = np.concatenate([temporal_importance, static_importance])
    order = np.argsort(-importances)[:top_n]

    fig, axis = plt.subplots(figsize=(8, 0.35 * top_n + 1))
    axis.barh(range(top_n), importances[order][::-1], color="#4C72B0")
    axis.set_yticks(range(top_n))
    axis.set_yticklabels([names[i] for i in order][::-1], fontsize=8)
    axis.set_xlabel("Mean |Integrated Gradients attribution|")
    axis.set_title(f"Top {top_n} features by Integrated Gradients: {model_name}")
    fig.tight_layout()
    path = output_dir / f"ig_feature_importance_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved %s", path)

    import pandas as pd
    pd.DataFrame({"feature": names, "mean_abs_ig": importances}).sort_values(
        "mean_abs_ig", ascending=False
    ).to_csv(output_dir / f"ig_feature_importance_{model_name}.csv", index=False)


def plot_attribution_by_month(model_name, ig_temporal, labels, valid, output_dir):
    per_month = np.abs(ig_temporal).sum(axis=2)  # (n, seq_len): total attribution magnitude per month
    months = np.arange(1, ig_temporal.shape[1] + 1)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    for label_value, color, name in [(0, "#4C72B0", "Non-defaulters"), (1, "#C44E52", "Defaulters")]:
        mask = valid & (labels == label_value)
        axis.plot(months, per_month[mask].mean(axis=0), marker="o", label=name, color=color)
    axis.set_xlabel("Month index (1 = oldest in 12-month window, 12 = most recent)")
    axis.set_ylabel("Mean |IG attribution| (summed over features)")
    axis.set_title(f"Integrated Gradients attribution by month: {model_name}")
    axis.legend()
    axis.grid(alpha=0.3)
    fig.tight_layout()
    path = output_dir / f"ig_by_month_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved %s", path)


def write_local_explanations(model_name, ig_temporal, ig_static, temporal_names, static_names, valid, labels, output_dir, top_n=5):
    sequence_length = ig_temporal.shape[1]
    flat_temporal_names = [f"month_{m + 1}_{name}" for m in range(sequence_length) for name in temporal_names]

    explanations = []
    for index in np.where(valid)[0]:
        temporal_flat = ig_temporal[index].reshape(-1)
        combined_values = np.concatenate([temporal_flat, ig_static[index]])
        combined_names = flat_temporal_names + list(static_names)

        top_indices = np.argsort(-np.abs(combined_values))[:top_n]
        drivers = [
            {
                "feature": combined_names[i],
                "ig_attribution": float(combined_values[i]),
                "direction": "increased" if combined_values[i] > 0 else "reduced",
            }
            for i in top_indices
        ]
        explanations.append({
            "borrower_index": int(index),
            "true_label": int(labels[index]),
            "main_drivers": drivers,
        })

    path = output_dir / f"ig_local_explanations_{model_name}.json"
    path.write_text(json.dumps(explanations, indent=2), encoding="utf-8")
    LOGGER.info("Saved %s", path)


def main() -> None:
    ensure_directory(IG_DIR)

    data = load_and_normalize_dataset(DATA_PATH)
    data = data.sort_values(["borrower_id", "transaction_date"]).reset_index(drop=True)
    X_temporal, X_static, labels, _, temporal_names, static_names, _ = build_training_arrays(data)
    labels = labels.astype(int)

    rng = np.random.default_rng(RANDOM_SEED)
    positive_index = np.where(labels == 1)[0]
    negative_index = rng.choice(np.where(labels == 0)[0], size=min(N_NEGATIVE_SAMPLE, (labels == 0).sum()), replace=False)
    sample_index = np.concatenate([positive_index, negative_index])
    LOGGER.info("Explaining %d borrowers (%d positive, %d negative)", len(sample_index), len(positive_index), len(negative_index))

    for model_name in ["hybrid", "transformer"]:
        run_ig_study(model_name, X_temporal, X_static, labels, temporal_names, static_names, sample_index)

    LOGGER.info("Integrated Gradients study complete. Artifacts in %s", IG_DIR)


if __name__ == "__main__":
    main()
