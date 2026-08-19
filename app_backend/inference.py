"""Feature reconstruction and five-fold hybrid-model ensemble inference."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import FeatureColumns, LocalisedFeatureEngineer

MODEL_DIR_NAME = "uganda_mobile_money_evaluation_comparison"
REQUIRED_COLUMNS = {"borrower_id", "transaction_date", "transaction_amount"}


class InputError(ValueError):
    pass


class HybridRiskService:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.model_dir = root / "models" / MODEL_DIR_NAME
        self.models = []
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
        by_borrower = reference.groupby("borrower_id")[self.numeric_static_features].mean()
        stats = {}
        for column in self.numeric_static_features:
            mean, std = float(by_borrower[column].mean()), float(by_borrower[column].std(ddof=0))
            stats[column] = (mean, std if std > 1e-9 else 1.0)
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
        probability = float(np.mean(probabilities))
        decision = "Likely to default" if probability >= self.threshold else "Lower default risk"
        return {
            "borrower_id": borrower_id,
            "default_probability": round(probability, 4),
            "decision": decision,
            "threshold": round(float(self.threshold), 4),
            "model": "Hybrid LSTM-FNN five-fold ensemble",
            "transactions_used": int(len(data)),
            "period": {"from": earliest.date().isoformat(), "to": latest.date().isoformat()},
            "notice": "Decision support only. A model score must not be the sole basis for a credit decision.",
        }

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
        aggregate = engineered.groupby("borrower_id")[self.numeric_static_features].mean().iloc[0].to_dict()
        aggregate["loan_amount"] = float(metadata["loan_amount"])
        numeric = [(float(aggregate.get(name, 0)) - self.reference_stats[name][0]) / self.reference_stats[name][1] for name in self.numeric_static_features]
        # Training data only contained the "Unknown" category in its persisted feature schema.
        categorical = [1.0 for _ in self.static_features if "_Unknown" in _]
        return np.asarray([numeric + categorical], dtype=np.float32)
