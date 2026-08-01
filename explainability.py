"""Explainability utilities for the hybrid credit risk model."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".matplotlib"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf

from config import MODELS_DIR
from hybrid_model import F1Score
from utils import ensure_directory


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExplainabilityArtifacts:
    """Paths written by the explainability pipeline."""

    global_summary_plot: Path
    feature_importance_csv: Path
    local_explanations_json: Path
    waterfall_dir: Path
    force_plot_dir: Path


class HybridModelExplainer:
    """Generate SHAP explanations for hybrid temporal and static inputs.

    Uses ``shap.GradientExplainer`` rather than ``KernelExplainer``: the
    hybrid model is a differentiable Keras model, so gradient-based
    attribution is exact-er and orders of magnitude faster than
    perturbation-based KernelExplainer, and it returns a real per-input
    array (not a flattened, manually-reshaped one) so waterfall/force
    plots work natively instead of being skipped.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        temporal_feature_names: list[str],
        static_feature_names: list[str],
        output_dir: str | Path = MODELS_DIR / "explainability",
    ) -> None:
        self.model = model
        self.temporal_feature_names = temporal_feature_names
        self.static_feature_names = static_feature_names
        self.output_dir = ensure_directory(Path(output_dir))
        self.waterfall_dir = ensure_directory(self.output_dir / "waterfall")
        self.force_plot_dir = ensure_directory(self.output_dir / "force")

    def explain(
        self,
        X_temporal: np.ndarray,
        X_static: np.ndarray,
        borrower_ids: np.ndarray | None = None,
        background_size: int = 100,
        explanation_size: int = 50,
    ) -> ExplainabilityArtifacts:
        """Generate global and local SHAP explanations.

        Args:
            X_temporal: Temporal model input.
            X_static: Static model input.
            borrower_ids: Optional borrower identifiers for local explanations.
            background_size: Number of rows used to estimate the background.
            explanation_size: Number of rows explained and plotted.

        Returns:
            Paths to generated SHAP artifacts.
        """

        rng = np.random.default_rng(42)
        background_index = rng.choice(len(X_temporal), size=min(background_size, len(X_temporal)), replace=False)
        explain_index = np.arange(min(explanation_size, len(X_temporal)))

        background = [X_temporal[background_index], X_static[background_index]]
        explain_temporal = X_temporal[explain_index]
        explain_static = X_static[explain_index]

        explainer = shap.GradientExplainer(self.model, background)
        temporal_shap, static_shap = explainer.shap_values([explain_temporal, explain_static])
        # shape (n, seq_len, temporal_features, 1) / (n, static_features, 1) -> drop the
        # trailing single-output axis and flatten temporal to match feature_names order.
        temporal_shap = temporal_shap[..., 0].reshape(len(explain_index), -1)
        static_shap = static_shap[..., 0]
        shap_values_array = np.concatenate([temporal_shap, static_shap], axis=1)

        feature_names = self._combined_feature_names(X_temporal.shape[1])
        explain_frame = self._combine_inputs(explain_temporal, explain_static)

        predictions = self._predict_combined(explain_temporal, explain_static)
        base_value = float(np.mean(predictions) - shap_values_array.sum(axis=1).mean())
        shap_values = shap.Explanation(
            values=shap_values_array,
            base_values=np.full(len(explain_index), base_value),
            data=explain_frame,
            feature_names=feature_names,
        )

        summary_path = self.output_dir / "global_shap_summary.png"
        self._plot_summary(shap_values, summary_path)

        importance_path = self.output_dir / "feature_importance.csv"
        importance = self._feature_importance(shap_values, feature_names)
        importance.to_csv(importance_path, index=False)

        local_path = self.output_dir / "local_explanations.json"
        local_explanations = self._local_plain_language_explanations(
            shap_values,
            predictions,
            feature_names,
            borrower_ids=borrower_ids[explain_index] if borrower_ids is not None else None,
        )
        local_path.write_text(
            json.dumps(local_explanations, indent=2),
            encoding="utf-8",
        )

        self._plot_local_explanations(shap_values)

        LOGGER.info("Explainability artifacts written to %s", self.output_dir)
        return ExplainabilityArtifacts(
            global_summary_plot=summary_path,
            feature_importance_csv=importance_path,
            local_explanations_json=local_path,
            waterfall_dir=self.waterfall_dir,
            force_plot_dir=self.force_plot_dir,
        )

    def _predict_combined(self, X_temporal: np.ndarray, X_static: np.ndarray) -> np.ndarray:
        return self.model.predict([X_temporal, X_static], verbose=0).ravel()

    def _combine_inputs(
        self,
        X_temporal: np.ndarray,
        X_static: np.ndarray,
    ) -> np.ndarray:
        return np.concatenate(
            [X_temporal.reshape(X_temporal.shape[0], -1), X_static],
            axis=1,
        )

    def _combined_feature_names(self, sequence_length: int) -> list[str]:
        temporal_names = [
            f"month_{month_index + 1}_{feature_name}"
            for month_index in range(sequence_length)
            for feature_name in self.temporal_feature_names
        ]
        return [*temporal_names, *self.static_feature_names]

    @staticmethod
    def _plot_summary(
        shap_values: shap.Explanation,
        output_path: Path,
    ) -> None:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values.values,
            shap_values.data,
            feature_names=shap_values.feature_names,
            show=False,
            max_display=25,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    @staticmethod
    def _feature_importance(
        shap_values: shap.Explanation,
        feature_names: list[str],
    ) -> pd.DataFrame:
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        return pd.DataFrame(
            {
                "feature": feature_names,
                "mean_absolute_shap": mean_abs_shap,
            }
        ).sort_values("mean_absolute_shap", ascending=False)

    def _local_plain_language_explanations(
        self,
        shap_values: shap.Explanation,
        predictions: np.ndarray,
        feature_names: list[str],
        borrower_ids: np.ndarray | None = None,
        top_n: int = 5,
    ) -> list[dict[str, object]]:
        explanations: list[dict[str, object]] = []

        for row_index, row_values in enumerate(shap_values.values):
            top_indices = np.argsort(np.abs(row_values))[-top_n:][::-1]
            drivers = []
            for feature_index in top_indices:
                direction = "increased" if row_values[feature_index] > 0 else "reduced"
                drivers.append(
                    {
                        "feature": feature_names[feature_index],
                        "value": float(shap_values.data[row_index, feature_index]),
                        "shap_value": float(row_values[feature_index]),
                        "plain_language": (
                            f"{feature_names[feature_index]} {direction} the "
                            "predicted default risk for this borrower."
                        ),
                    }
                )

            borrower_id = (
                str(borrower_ids[row_index])
                if borrower_ids is not None and row_index < len(borrower_ids)
                else str(row_index)
            )
            explanations.append(
                {
                    "borrower_id": borrower_id,
                    "predicted_default_risk": float(predictions[row_index]),
                    "main_drivers": drivers,
                    "loan_officer_summary": self._loan_officer_summary(
                        predictions[row_index],
                        drivers,
                    ),
                }
            )

        return explanations

    @staticmethod
    def _loan_officer_summary(
        predicted_risk: float,
        drivers: list[dict[str, object]],
    ) -> str:
        risk_band = "high" if predicted_risk >= 0.60 else "moderate"
        if predicted_risk < 0.35:
            risk_band = "low"

        driver_text = "; ".join(
            str(driver["plain_language"]) for driver in drivers[:3]
        )
        return (
            f"The borrower has a {risk_band} predicted default risk "
            f"({predicted_risk:.1%}). Main model drivers: {driver_text}."
        )

    def _plot_local_explanations(
        self,
        shap_values: shap.Explanation,
        max_plots: int = 10,
    ) -> None:
        """Save native SHAP waterfall plots for the first ``max_plots`` rows.

        Unlike KernelExplainer's manually-wrapped output, ``shap_values``
        here is a real ``shap.Explanation``, so ``shap.plots.waterfall``
        works directly.
        """
        for row_index in range(min(max_plots, len(shap_values))):
            plt.figure()
            shap.plots.waterfall(shap_values[row_index], max_display=15, show=False)
            plt.tight_layout()
            plt.savefig(self.waterfall_dir / f"borrower_{row_index}.png", dpi=150, bbox_inches="tight")
            plt.close()
        LOGGER.info("Saved %d waterfall plots to %s", min(max_plots, len(shap_values)), self.waterfall_dir)


def load_trained_model(model_path: str | Path) -> tf.keras.Model:
    """Load a trained hybrid model with custom metrics."""

    return tf.keras.models.load_model(
        model_path,
        custom_objects={"F1Score": F1Score},
    )


def explain_model(
    model_path: str | Path,
    X_temporal: np.ndarray,
    X_static: np.ndarray,
    temporal_feature_names: list[str],
    static_feature_names: list[str],
    borrower_ids: np.ndarray | None = None,
    output_dir: str | Path = MODELS_DIR / "explainability",
) -> ExplainabilityArtifacts:
    """Load a trained model and generate SHAP explanations."""

    model = load_trained_model(model_path)
    explainer = HybridModelExplainer(
        model=model,
        temporal_feature_names=temporal_feature_names,
        static_feature_names=static_feature_names,
        output_dir=output_dir,
    )
    return explainer.explain(
        X_temporal=X_temporal,
        X_static=X_static,
        borrower_ids=borrower_ids,
    )


def explain() -> None:
    """CLI placeholder for notebook-driven explainability."""

    raise RuntimeError(
        "Use explain_model(model_path, X_temporal, X_static, feature names)."
    )


if __name__ == "__main__":
    explain()
