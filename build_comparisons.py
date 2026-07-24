#!/usr/bin/env python
"""Regenerate the master.csv model comparison (now with vanilla LSTM), and
build a separate baseline-vs-best-imbalance-config comparison from the
sweep results in imbalance_sweep/.
"""

from __future__ import annotations

import numpy as np

from imbalance_sweep import pooled_metrics_from_oof
from run_experiment_evaluation_comparison import OUTPUT_DIR
from visualize_model_comparison import (
    plot_model_comparison,
    plot_pr_curves_overlay,
    write_comparison_table,
    write_lift_table,
)

SWEEP_DIR = OUTPUT_DIR / "imbalance_sweep"

# --- 1. Regenerate the main 6-model comparison (adds Vanilla LSTM) ---
main_model_dirs = {
    "Hybrid LSTM-GRU-Attention-FNN": OUTPUT_DIR / "kfold_hybrid",
    "Logistic Regression": OUTPUT_DIR / "logistic_regression",
    "XGBoost": OUTPUT_DIR / "xgboost",
    "Vanilla LSTM": OUTPUT_DIR / "kfold_vanilla_lstm",
    "Transformer-Encoder Hybrid": OUTPUT_DIR / "kfold_transformer",
    "Cross-Attention Fusion Hybrid": OUTPUT_DIR / "kfold_cross_attention",
}

import json
main_summaries = {}
for name, result_dir in main_model_dirs.items():
    with open(result_dir / "kfold_summary.json", encoding="utf-8") as handle:
        main_summaries[name] = json.load(handle)["pooled_metrics"]

comparison_dir = OUTPUT_DIR / "model_comparison"
plot_model_comparison(main_summaries, comparison_dir / "model_comparison.png")
write_comparison_table(main_summaries, comparison_dir / "model_comparison.md")
plot_pr_curves_overlay(main_model_dirs, comparison_dir / "pr_curves_overlay.png")
write_lift_table(main_summaries, comparison_dir / "lift_over_random.md")
print("Regenerated main 6-model comparison at", comparison_dir)

# --- 2. Baseline vs. best imbalance-handling config, per model ---
oof_true = np.load(OUTPUT_DIR / "kfold_hybrid" / "oof_true.npy")

best_config_dirs = {
    "Logistic Regression (baseline)": OUTPUT_DIR / "logistic_regression",
    "Logistic Regression (no_weight)": SWEEP_DIR / "logistic_regression__no_weight__oof_probabilities.npy",
    "XGBoost (baseline)": OUTPUT_DIR / "xgboost",
    "XGBoost (no_weight)": SWEEP_DIR / "xgboost__no_weight__oof_probabilities.npy",
    "XGBoost (smote_0.5)": SWEEP_DIR / "xgboost__smote_0.5__oof_probabilities.npy",
    "Vanilla LSTM (baseline)": OUTPUT_DIR / "kfold_vanilla_lstm",
    "Vanilla LSTM (weight_1.5)": SWEEP_DIR / "vanilla_lstm__weight_1.5__oof_probabilities.npy",
    "Hybrid (baseline)": OUTPUT_DIR / "kfold_hybrid",
    "Hybrid (weight_1.5)": SWEEP_DIR / "hybrid__weight_1.5__oof_probabilities.npy",
}

best_summaries = {}
best_model_result_dirs = {}
best_config_out = SWEEP_DIR / "best_config_comparison"
best_config_out.mkdir(parents=True, exist_ok=True)

for name, path in best_config_dirs.items():
    if path.suffix == ".npy":
        oof = np.load(path)
        best_summaries[name] = pooled_metrics_from_oof(oof_true, oof)
        # Write throwaway oof pair so plot_pr_curves_overlay can read it uniformly.
        pair_dir = best_config_out / name.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
        pair_dir.mkdir(parents=True, exist_ok=True)
        np.save(pair_dir / "oof_probabilities.npy", oof)
        np.save(pair_dir / "oof_true.npy", oof_true)
        best_model_result_dirs[name] = pair_dir
    else:
        with open(path / "kfold_summary.json", encoding="utf-8") as handle:
            best_summaries[name] = json.load(handle)["pooled_metrics"]
        best_model_result_dirs[name] = path

plot_model_comparison(best_summaries, best_config_out / "model_comparison.png")
write_comparison_table(best_summaries, best_config_out / "model_comparison.md")
plot_pr_curves_overlay(best_model_result_dirs, best_config_out / "pr_curves_overlay.png")
write_lift_table(best_summaries, best_config_out / "lift_over_random.md")
print("Built baseline-vs-best-config comparison at", best_config_out)
