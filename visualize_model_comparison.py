"""Compare pooled out-of-fold metrics across all trained models.

Reads each model's kfold_summary.json (written by kfold_cv_hybrid.py,
xgboost_baseline.py, kfold_cv_vanilla_lstm.py, and
logistic_regression_baseline.py), builds a grouped bar chart, and
writes a markdown comparison table.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import ensure_directory

LOGGER = logging.getLogger(__name__)

METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "average_precision", "roc_auc"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "average_precision": "Avg. Precision",
    "roc_auc": "ROC-AUC",
}


def load_model_summaries(model_result_dirs: dict[str, Path]) -> dict[str, dict]:
    """Load pooled_metrics from each model's kfold_summary.json.

    Args:
        model_result_dirs: Mapping of display name -> directory containing
            kfold_summary.json (the output_dir passed to each *_baseline
            or run_kfold_cv* function).

    Returns:
        Mapping of display name -> pooled_metrics dict. Models whose
        summary file is missing are skipped with a warning.
    """
    summaries: dict[str, dict] = {}
    for name, result_dir in model_result_dirs.items():
        summary_path = Path(result_dir) / "kfold_summary.json"
        if not summary_path.exists():
            LOGGER.warning("No kfold_summary.json found for %s at %s, skipping", name, summary_path)
            continue
        with open(summary_path, encoding="utf-8") as handle:
            data = json.load(handle)
        summaries[name] = data["pooled_metrics"]
    return summaries


def plot_model_comparison(summaries: dict[str, dict], output_path: Path) -> Path:
    """Save a grouped bar chart comparing models across all metrics."""

    model_names = list(summaries.keys())
    n_models = len(model_names)
    n_metrics = len(METRIC_KEYS)

    fig, axis = plt.subplots(figsize=(10, 6))
    bar_width = 0.8 / n_models
    x_positions = np.arange(n_metrics)

    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    for model_index, model_name in enumerate(model_names):
        values = [summaries[model_name].get(key, 0.0) for key in METRIC_KEYS]
        offset = (model_index - (n_models - 1) / 2) * bar_width
        bars = axis.bar(
            x_positions + offset,
            values,
            width=bar_width,
            label=model_name,
            color=colors[model_index],
        )
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    axis.set_xticks(x_positions)
    axis.set_xticklabels([METRIC_LABELS[key] for key in METRIC_KEYS])
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1.05)
    axis.set_title("Model Comparison - Pooled Out-of-Fold Metrics (5-Fold CV)")
    axis.legend(loc="upper right", fontsize=9)
    axis.grid(axis="y", alpha=0.3)

    fig.text(
        0.5, -0.02,
        "Note: accuracy is misleading here (~97.7% baseline achievable by predicting no default for everyone, given ~2.3% positive rate)",
        ha="center", fontsize=7, style="italic", color="dimgray",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved model comparison chart: %s", output_path)
    return output_path


def _dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a markdown table without external dependencies.

    Numeric columns are formatted to 4 decimal places; other columns are
    rendered as plain strings.
    """

    def format_cell(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    header = "| " + " | ".join(str(col) for col in frame.columns) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body_lines = [
        "| " + " | ".join(format_cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *body_lines])


def write_comparison_table(summaries: dict[str, dict], output_path: Path) -> Path:
    """Write a markdown table comparing all models, sorted by F1 descending."""

    rows = []
    for model_name, metrics in summaries.items():
        row = {"Model": model_name}
        row.update({METRIC_LABELS[key]: metrics.get(key, float("nan")) for key in METRIC_KEYS})
        row["Threshold"] = metrics.get("selected_threshold", float("nan"))
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)

    lines = ["# Model Comparison — Pooled Out-of-Fold Metrics (5-Fold CV)", ""]
    lines.append(_dataframe_to_markdown(frame))
    lines.append("")
    lines.append(
        "All metrics computed on pooled out-of-fold predictions (every positive "
        "borrower evaluated exactly once across folds), not a single held-out split."
    )
    lines.append("")
    lines.append(
        "Note: Accuracy is included for completeness but is not a reliable "
        "indicator of model quality here, given the ~2.3% positive rate — a "
        "model that predicts \"no default\" for every borrower would score "
        "~97.7% accuracy while catching zero actual defaulters. Precision, "
        "recall, F1, average precision, and ROC-AUC are more informative for "
        "this task."
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Saved comparison table: %s", output_path)
    return output_path


def compare_all_models(
    model_result_dirs: dict[str, Path],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Load all model summaries, plot comparison, and write markdown table.

    Args:
        model_result_dirs: Mapping of display name -> output_dir used for
            that model's CV run, e.g.
            {
                "Hybrid LSTM-FNN": OUTPUT_DIR / "kfold_hybrid",
                "XGBoost": OUTPUT_DIR / "xgboost_baseline",
                "Vanilla LSTM": OUTPUT_DIR / "kfold_vanilla_lstm",
                "Logistic Regression": OUTPUT_DIR / "logistic_regression_baseline",
            }
        output_dir: Directory to write the chart and table into.

    Returns:
        Tuple of (chart_path, table_path).
    """
    output_path = ensure_directory(Path(output_dir))
    summaries = load_model_summaries(model_result_dirs)

    if not summaries:
        raise RuntimeError(
            "No kfold_summary.json files found for any model. "
            "Run the CV scripts before calling compare_all_models()."
        )

    chart_path = plot_model_comparison(summaries, output_path / "model_comparison.png")
    table_path = write_comparison_table(summaries, output_path / "model_comparison.md")

    return chart_path, table_path


if __name__ == "__main__":
    raise RuntimeError(
        "Import compare_all_models(model_result_dirs, output_dir) and call it "
        "after running the CV scripts for each model, e.g. from a run_experiment script."
    )