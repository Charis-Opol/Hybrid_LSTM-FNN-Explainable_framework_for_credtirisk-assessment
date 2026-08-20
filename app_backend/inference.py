"""Feature reconstruction and five-fold hybrid-model ensemble inference."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import FeatureColumns, LocalisedFeatureEngineer

MODEL_DIR_NAME = os.getenv("RISK_MODEL_DIR", "uganda_mobile_money_hybrid_vs_transformer")
REQUIRED_COLUMNS = {"borrower_id", "transaction_date", "transaction_amount"}


class InputError(ValueError):
    pass


class HybridRiskService:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.model_dir = root / "models" / MODEL_DIR_NAME
        self.models = []
        self.shap_explainer = None
        self.is_loaded = False
        self.temporal_features = self._read_features("temporal_feature_names.txt")
        self.static_features = self._read_features("static_feature_names.txt")
        self.threshold = json.loads((self.model_dir / "kfold_hybrid" / "kfold_summary.json").read_text())["pooled_metrics"]["selected_threshold"]
        self.numeric_static_features = [name for name in self.static_features if "_Unknown" not in name]
        self.reference_stats = self._reference_stats()

    def _read_features(self, name: str) -> list[str]:
        return [line.strip() for line in (self.model_dir / name).read_text().splitlines() if line.strip()]

    def _reference_stats(self) -> dict[str, tuple[float, float]]:
        reference = pd.read_csv(self.model_dir / "engineered_features.csv")
        # The saved training CSV contains monthly engineered features, while the
        # static branch was trained on borrower-level aggregates.
        if "average_inflow" not in reference:
            reference["average_inflow"] = reference.get("monthly_inflow", 0.0)
        if "average_outflow" not in reference:
            reference["average_outflow"] = reference.get("monthly_outflow", 0.0)
        for column in self.numeric_static_features:
            if column not in reference:
                reference[column] = 0.0
        by_borrower = reference.groupby("borrower_id")[self.numeric_static_features].mean()
        stats = {}
        for column in self.numeric_static_features:
            mean, std = float(by_borrower[column].mean()), float(by_borrower[column].std(ddof=0))
            stats[column] = (mean, std if std > 1e-9 else 1.0)
        # loan_amount is a static training input and is not persisted in the
        # monthly engineered CSV. Recover its original training scale instead
        # of sending raw UGX values into a StandardScaler-trained model.
        raw_path = self.root / "data" / "raw" / "uganda_mobile_money_master.csv"
        if "loan_amount" in self.numeric_static_features and raw_path.exists():
            loan_amounts = pd.read_csv(raw_path, usecols=["loan_amount"])["loan_amount"]
            stats["loan_amount"] = (
                float(loan_amounts.mean()),
                float(loan_amounts.std(ddof=0)) or 1.0,
            )
        return stats

    def load(self) -> None:
        if self.is_loaded:
            return
        import tensorflow as tf
        from hybrid_model import F1Score

        paths = sorted((self.model_dir / "kfold_hybrid").glob("fold_*/best_model.keras"))
        if len(paths) != 5:
            raise RuntimeError("Expected five hybrid fold models but could not find them.")
        self.models = [tf.keras.models.load_model(path, custom_objects={"F1Score": F1Score}, compile=False) for path in paths]
        import shap

        rng = np.random.default_rng(42)
        temporal_background = rng.normal(
            0,
            1,
            size=(2, 12, len(self.temporal_features)),
        ).astype(np.float32)
        static_background = rng.normal(
            0,
            1,
            size=(2, len(self.static_features)),
        ).astype(np.float32)
        self.shap_explainer = shap.GradientExplainer(
            self.models[0],
            [temporal_background, static_background],
        )
        self.is_loaded = True

    def predict_csv(self, raw: bytes, borrower_id: str, borrower_metadata: dict) -> dict:
        if not borrower_id:
            raise InputError("Borrower ID is required.")
        try:
            data = pd.read_csv(io.BytesIO(raw))
        except Exception as error:
            raise InputError("The uploaded file is not a readable CSV.") from error
        missing = REQUIRED_COLUMNS - set(data.columns)
        if missing:
            raise InputError("CSV is missing required columns: " + ", ".join(sorted(missing)))
        data = data[data["borrower_id"].astype(str) == borrower_id].copy()
        if data.empty:
            raise InputError("No rows match the supplied borrower ID.")
        if len(data) < 2:
            raise InputError("Provide at least two transactions for a meaningful assessment.")
        try:
            data["transaction_date"] = pd.to_datetime(data["transaction_date"], errors="raise")
            data["transaction_amount"] = pd.to_numeric(data["transaction_amount"], errors="raise")
        except Exception as error:
            raise InputError("transaction_date and transaction_amount must contain valid values.") from error
        if (data["transaction_amount"].abs() == 0).all():
            raise InputError("At least one transaction amount must be non-zero.")

        latest = data["transaction_date"].max()
        earliest = data["transaction_date"].min()
        if (latest.to_period("M") - earliest.to_period("M")).n < 11:
            raise InputError("Upload transactions spanning at least 12 calendar months.")
        if "balance" not in data:
            data["balance"] = 0.0
        if "transaction_type" not in data:
            data["transaction_type"] = "received"
        for key, value in borrower_metadata.items():
            data[key] = value

        engineered = LocalisedFeatureEngineer(FeatureColumns()).engineer(data)
        temporal = self._temporal_vector(engineered)
        static = self._static_vector(engineered, borrower_metadata)
        probabilities = [float(model.predict([temporal, static], verbose=0).ravel()[0]) for model in self.models]
        model_probability = float(np.mean(probabilities))
        affordability_ratio, affordability_penalty = self._affordability_adjustment(
            engineered,
            float(borrower_metadata["loan_amount"]),
        )
        probability = min(0.99, model_probability + affordability_penalty)
        decision = "Likely to default" if probability >= self.threshold else "Lower default risk"
        return {
            "borrower_id": borrower_id,
            "default_probability": round(probability, 4),
            "model_probability": round(model_probability, 4),
            "affordability_ratio": round(affordability_ratio, 4),
            "affordability_adjustment": round(affordability_penalty, 4),
            "decision": decision,
            "threshold": round(float(self.threshold), 4),
            "model": "Hybrid LSTM-FNN five-fold ensemble",
            "transactions_used": int(len(data)),
            "period": {"from": earliest.date().isoformat(), "to": latest.date().isoformat()},
            "explanations": [],
            "explanation_status": "request_available",
            "notice": "Decision support only. A model score must not be the sole basis for a credit decision.",
        }

    @staticmethod
    def _affordability_adjustment(
        engineered: pd.DataFrame,
        loan_amount: float,
    ) -> tuple[float, float]:
        """Apply a transparent affordability penalty to the model probability.

        The trained model includes loan amount, but its learned relationship is
        weak in this highly imbalanced dataset. This overlay makes a requested
        amount materially riskier when it is large relative to observed monthly
        inflow, without replacing the model probability.
        """
        monthly_inflow = float(engineered["monthly_inflow"].mean())
        if monthly_inflow <= 1e-9:
            return 0.0, 0.0
        ratio = max(0.0, loan_amount / monthly_inflow)
        # Increase gradually over a 100x inflow range; avoid making every
        # above-average request immediately hit the maximum penalty.
        penalty = min(0.35, max(0.0, ratio - 1.0) * 0.0035)
        return ratio, penalty

    def explain_csv(self, raw: bytes, borrower_id: str, borrower_metadata: dict) -> dict:
        """Build the same model inputs and calculate SHAP drivers on demand."""
        temporal, static = self._prepare_inputs(raw, borrower_id, borrower_metadata)
        explanations = self._shap_drivers(temporal, static)
        return {
            "borrower_id": borrower_id,
            "explanations": explanations,
            "explanation_status": "available" if explanations else "unavailable",
        }

    def _prepare_inputs(self, raw: bytes, borrower_id: str, borrower_metadata: dict) -> tuple[np.ndarray, np.ndarray]:
        data = pd.read_csv(io.BytesIO(raw))
        missing = REQUIRED_COLUMNS - set(data.columns)
        if missing:
            raise InputError("CSV is missing required columns: " + ", ".join(sorted(missing)))
        data = data[data["borrower_id"].astype(str) == borrower_id].copy()
        if data.empty:
            raise InputError("No rows match the supplied borrower ID.")
        data["transaction_date"] = pd.to_datetime(data["transaction_date"], errors="raise")
        data["transaction_amount"] = pd.to_numeric(data["transaction_amount"], errors="raise")
        if "balance" not in data:
            data["balance"] = 0.0
        if "transaction_type" not in data:
            data["transaction_type"] = "received"
        for key, value in borrower_metadata.items():
            data[key] = value
        engineered = LocalisedFeatureEngineer(FeatureColumns()).engineer(data)
        return self._temporal_vector(engineered), self._static_vector(engineered, borrower_metadata)

    def _shap_drivers(self, temporal: np.ndarray, static: np.ndarray) -> list[dict[str, float | str]]:
        """Return the strongest local SHAP drivers for the first ensemble fold."""
        try:
            import shap

            if self.shap_explainer is None:
                return []
            values = self.shap_explainer.shap_values([temporal, static], nsamples=1)
            if isinstance(values, list):
                temporal_values, static_values = values[0], values[1]
            else:
                temporal_values, static_values = values
            temporal_values = np.asarray(temporal_values)
            static_values = np.asarray(static_values)
            temporal_values = np.squeeze(temporal_values, axis=-1) if temporal_values.ndim == 4 else temporal_values
            static_values = np.squeeze(static_values, axis=-1) if static_values.ndim == 3 else static_values
            temporal_values = temporal_values[0]
            static_values = static_values[0]

            candidates: list[tuple[str, float]] = []
            for month_index, row in enumerate(temporal_values):
                for feature_index, value in enumerate(row):
                    candidates.append((f"month_{month_index + 1}_{self.temporal_features[feature_index]}", float(value)))
            candidates.extend((name, float(value)) for name, value in zip(self.static_features, static_values))
            candidates.sort(key=lambda item: abs(item[1]), reverse=True)
            if not candidates or max(abs(value) for _, value in candidates) < 1e-8:
                return []
            return [
                {
                    "feature": name,
                    "impact": round(value, 6),
                    "direction": "increases risk" if value > 0 else "reduces risk",
                }
                for name, value in candidates[:8]
            ]
        except Exception:
            return []

    def _temporal_vector(self, engineered: pd.DataFrame) -> np.ndarray:
        recent = engineered.sort_values("month").tail(12).copy()
        for name in self.temporal_features:
            if name not in recent:
                recent[name] = 0.0
        values = recent[self.temporal_features].replace([np.inf, -np.inf], 0).fillna(0).to_numpy(dtype=np.float32)
        padded = np.zeros((12, len(self.temporal_features)), dtype=np.float32)
        padded[-len(values):] = values
        return padded[np.newaxis, :, :]

    def _static_vector(self, engineered: pd.DataFrame, metadata: dict) -> np.ndarray:
        if "average_inflow" not in engineered:
            engineered["average_inflow"] = engineered.get("monthly_inflow", 0.0)
        if "average_outflow" not in engineered:
            engineered["average_outflow"] = engineered.get("monthly_outflow", 0.0)
        for column in self.numeric_static_features:
            if column not in engineered:
                engineered[column] = 0.0
        aggregate = engineered.groupby("borrower_id")[self.numeric_static_features].mean().iloc[0].to_dict()
        aggregate["loan_amount"] = float(metadata["loan_amount"])
        numeric = [(float(aggregate.get(name, 0)) - self.reference_stats[name][0]) / self.reference_stats[name][1] for name in self.numeric_static_features]
        # Training data only contained the "Unknown" category in its persisted feature schema.
        categorical = [1.0 for _ in self.static_features if "_Unknown" in _]
        return np.asarray([numeric + categorical], dtype=np.float32)
