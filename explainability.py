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
    """Generate SHAP explanations for hybrid temporal and static inputs."""

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

        combined = self._combine_inputs(X_temporal, X_static)
        feature_names = self._combined_feature_names(X_temporal.shape[1])
        background = shap.sample(
            combined,
            min(background_size, len(combined)),
            random_state=42,
        )
        explain_frame = combined[: min(explanation_size, len(combined))]

        # Use KernelExplainer for high-dimensional feature spaces
        explainer = shap.KernelExplainer(
            self._predict_combined,
            background,
            feature_names=feature_names,
        )
        shap_values_array = explainer.shap_values(explain_frame)
        
        # Wrap in Explanation object for compatibility
        class ShapExplanation:
            def __init__(self, values, base_value=0.5):
                self.values = values
                self.base_values = np.array([base_value] * len(values))
                
            def __getitem__(self, idx):
                return self.values[idx]
        
        shap_values = ShapExplanation(shap_values_array)

        summary_path = self.output_dir / "global_shap_summary.png"
        self._plot_summary(shap_values, explain_frame, feature_names, summary_path)

        importance_path = self.output_dir / "feature_importance.csv"
        importance = self._feature_importance(shap_values, feature_names)
        importance.to_csv(importance_path, index=False)

        local_path = self.output_dir / "local_explanations.json"
        local_explanations = self._local_plain_language_explanations(
            shap_values,
            explain_frame,
            feature_names,
            borrower_ids=borrower_ids,
        )
        local_path.write_text(
            json.dumps(local_explanations, indent=2),
            encoding="utf-8",
        )

        self._plot_local_explanations(shap_values, feature_names)

        LOGGER.info("Explainability artifacts written to %s", self.output_dir)
        return ExplainabilityArtifacts(
            global_summary_plot=summary_path,
            feature_importance_csv=importance_path,
            local_explanations_json=local_path,
            waterfall_dir=self.waterfall_dir,
            force_plot_dir=self.force_plot_dir,
        )

    def _predict_combined(self, combined: np.ndarray) -> np.ndarray:
        sequence_length = len(self.temporal_feature_names)
        temporal_width = self._temporal_width_from_combined(combined)
        temporal_flat = combined[:, :temporal_width]
        static = combined[:, temporal_width:]
        temporal = temporal_flat.reshape(
            combined.shape[0],
            -1,
            sequence_length,
        )
        return self.model.predict([temporal, static], verbose=0).ravel()

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

    def _temporal_width_from_combined(self, combined: np.ndarray) -> int:
        static_width = len(self.static_feature_names)
        return combined.shape[1] - static_width

    @staticmethod
    def _plot_summary(
        shap_values: shap.Explanation,
        values: np.ndarray,
        feature_names: list[str],
        output_path: Path,
    ) -> None:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values.values,
            values,
            feature_names=feature_names,
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
        values: np.ndarray,
        feature_names: list[str],
        borrower_ids: np.ndarray | None = None,
        top_n: int = 5,
    ) -> list[dict[str, object]]:
        explanations: list[dict[str, object]] = []
        predictions = self._predict_combined(values)

        for row_index, row_values in enumerate(shap_values.values):
            top_indices = np.argsort(np.abs(row_values))[-top_n:][::-1]
            drivers = []
            for feature_index in top_indices:
                direction = "increased" if row_values[feature_index] > 0 else "reduced"
                drivers.append(
                    {
                        "feature": feature_names[feature_index],
                        "value": float(values[row_index, feature_index]),
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
        feature_names: list[str],
        max_plots: int = 10,
    ) -> None:
        """Generate local explanations. Waterfall and force plots skipped for KernelExplainer."""
        # Waterfall and force plots require native shap.Explanation objects
        # For KernelExplainer, we generate JSON explanations instead (handled in explain method)
        LOGGER.info("Local explanations stored in JSON format for compatibility")


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
