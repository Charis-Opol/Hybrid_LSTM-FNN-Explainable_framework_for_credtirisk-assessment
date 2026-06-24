"""Temporal dataset construction for LSTM-based borrower modelling."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    BORROWER_ID_COLUMN,
    SEQUENCE_LENGTH_MONTHS,
    TARGET_COLUMN,
    TRANSACTION_DATE_COLUMN,
)
from feature_engineering import engineer_features


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TemporalDataset:
    """Container for temporal model inputs."""

    X_temporal: np.ndarray
    borrower_ids: np.ndarray
    labels: np.ndarray | None
    mask: np.ndarray
    feature_names: list[str]


class TemporalDatasetBuilder:
    """Build fixed-length monthly sequences for each borrower."""

    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH_MONTHS,
        borrower_id_column: str = BORROWER_ID_COLUMN,
        target_column: str = TARGET_COLUMN,
        mask_value: float = 0.0,
    ) -> None:
        self.sequence_length = sequence_length
        self.borrower_id_column = borrower_id_column
        self.target_column = target_column
        self.mask_value = mask_value

    def build(
        self,
        data: pd.DataFrame,
        feature_columns: list[str] | None = None,
    ) -> TemporalDataset:
        """Create one 12-month sequence for every borrower.

        Args:
            data: Raw transaction dataframe or engineered borrower-month frame.
            feature_columns: Optional list of monthly feature columns.

        Returns:
            TemporalDataset containing ``X_temporal`` with shape
            ``(number_of_borrowers, 12, number_of_features)`` by default,
            borrower IDs, optional labels, and a binary observed-month mask.
        """

        monthly = self._ensure_monthly_features(data)
        features = feature_columns or self._infer_feature_columns(monthly)
        borrower_ids = np.array(sorted(monthly[self.borrower_id_column].unique()))

        sequences: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        labels: list[float] = []

        for borrower_id in borrower_ids:
            borrower_frame = monthly[
                monthly[self.borrower_id_column] == borrower_id
            ].sort_values("month")
            sequence, mask = self._borrower_sequence(borrower_frame, features)
            sequences.append(sequence)
            masks.append(mask)

            if self.target_column in borrower_frame:
                labels.append(float(borrower_frame[self.target_column].max()))

        label_array = np.array(labels) if labels else None
        result = TemporalDataset(
            X_temporal=np.stack(sequences),
            borrower_ids=borrower_ids,
            labels=label_array,
            mask=np.stack(masks),
            feature_names=features,
        )
        LOGGER.info("Built temporal dataset with shape %s", result.X_temporal.shape)
        return result

    def _ensure_monthly_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if "month" in data.columns:
            monthly = data.copy()
            monthly["month"] = pd.PeriodIndex(monthly["month"], freq="M")
            return monthly

        if TRANSACTION_DATE_COLUMN not in data:
            raise ValueError(
                "Input must contain either an engineered 'month' column or "
                f"raw '{TRANSACTION_DATE_COLUMN}' transactions."
            )
        return engineer_features(data)

    def _infer_feature_columns(self, monthly: pd.DataFrame) -> list[str]:
        excluded = {self.borrower_id_column, "month", self.target_column}
        return [
            column
            for column in monthly.select_dtypes(include="number").columns
            if column not in excluded
        ]

    def _borrower_sequence(
        self,
        borrower_frame: pd.DataFrame,
        feature_columns: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        latest_month = borrower_frame["month"].max()
        month_index = pd.period_range(
            end=latest_month,
            periods=self.sequence_length,
            freq="M",
        )
        indexed = borrower_frame.set_index("month").reindex(month_index)
        mask = indexed[feature_columns].notna().any(axis=1).astype(float).to_numpy()
        sequence = indexed[feature_columns].fillna(self.mask_value).to_numpy(dtype=float)
        return sequence, mask


def build_temporal_dataset(
    data: pd.DataFrame,
    sequence_length: int = SEQUENCE_LENGTH_MONTHS,
    feature_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Build fixed-length monthly borrower sequences.

    Args:
        data: Raw transaction dataframe or engineered borrower-month frame.
        sequence_length: Number of monthly time steps per borrower.
        feature_columns: Optional explicit feature columns.

    Returns:
        ``X_temporal``, borrower IDs, and optional labels.
    """

    dataset = TemporalDatasetBuilder(sequence_length=sequence_length).build(
        data,
        feature_columns=feature_columns,
    )
    return dataset.X_temporal, dataset.borrower_ids, dataset.labels

