"""Static dataset construction for feed-forward borrower modelling."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import BORROWER_ID_COLUMN, TARGET_COLUMN, TRANSACTION_DATE_COLUMN
from feature_engineering import engineer_features


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StaticDataset:
    """Container for static model inputs."""

    X_static: np.ndarray
    borrower_ids: np.ndarray
    labels: np.ndarray | None
    feature_names: list[str]
    transformer: ColumnTransformer


class StaticDatasetBuilder:
    """Aggregate one model-ready feature vector per borrower."""

    DEFAULT_STATIC_COLUMNS = [
        "income_source",
        "loan_amount",
        "sacco_membership",
        "average_balance",
        "average_inflow",
        "average_outflow",
        "income_regularity",
        "financial_stability_score",
        "behaviour_score",
        "location",
        "preferred_network",
        "preferred_channel",
    ]

    def __init__(
        self,
        borrower_id_column: str = BORROWER_ID_COLUMN,
        target_column: str = TARGET_COLUMN,
    ) -> None:
        self.borrower_id_column = borrower_id_column
        self.target_column = target_column
        self.transformer_: ColumnTransformer | None = None

    def build(self, data: pd.DataFrame) -> StaticDataset:
        """Build scaled and encoded static borrower features.

        Args:
            data: Raw transaction dataframe or engineered borrower-month frame.

        Returns:
            StaticDataset with ``X_static``, borrower IDs, optional labels,
            transformed feature names, and the fitted transformer.
        """

        borrower_frame = self._aggregate_borrowers(data)
        borrower_ids = borrower_frame[self.borrower_id_column].to_numpy()
        labels = (
            borrower_frame[self.target_column].to_numpy()
            if self.target_column in borrower_frame
            else None
        )
        feature_frame = borrower_frame.drop(
            columns=[
                column
                for column in [self.borrower_id_column, self.target_column]
                if column in borrower_frame
            ]
        )

        numeric_columns = list(feature_frame.select_dtypes(include="number").columns)
        categorical_columns = list(
            feature_frame.select_dtypes(include=["object", "category", "bool"]).columns
        )

        self.transformer_ = ColumnTransformer(
            transformers=[
                ("numeric", StandardScaler(), numeric_columns),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_columns,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        X_static = self.transformer_.fit_transform(feature_frame)
        feature_names = list(self.transformer_.get_feature_names_out())

        LOGGER.info("Built static dataset with shape %s", X_static.shape)
        return StaticDataset(
            X_static=X_static,
            borrower_ids=borrower_ids,
            labels=labels,
            feature_names=feature_names,
            transformer=self.transformer_,
        )

    def _aggregate_borrowers(self, data: pd.DataFrame) -> pd.DataFrame:
        if "month" in data.columns:
            monthly = data.copy()
        elif TRANSACTION_DATE_COLUMN in data:
            monthly = engineer_features(data)
        else:
            raise ValueError(
                "Input must contain either an engineered 'month' column or "
                f"raw '{TRANSACTION_DATE_COLUMN}' transactions."
            )

        if "average_inflow" not in monthly:
            monthly["average_inflow"] = monthly.get("monthly_inflow", 0)
        if "average_outflow" not in monthly:
            monthly["average_outflow"] = monthly.get("monthly_outflow", 0)

        aggregations: dict[str, tuple[str, str]] = {}
        for column in monthly.columns:
            if column in {self.borrower_id_column, "month"}:
                continue
            if column == self.target_column:
                aggregations[column] = (column, "max")
            elif pd.api.types.is_numeric_dtype(monthly[column]):
                aggregations[column] = (column, "mean")
            else:
                aggregations[column] = (column, self._mode_or_unknown)

        borrower_frame = monthly.groupby(self.borrower_id_column).agg(**aggregations)
        borrower_frame = borrower_frame.reset_index()

        for column in self.DEFAULT_STATIC_COLUMNS:
            if column not in borrower_frame:
                borrower_frame[column] = (
                    0 if column not in self._categorical_defaults() else "Unknown"
                )

        selected_columns = [
            self.borrower_id_column,
            *self.DEFAULT_STATIC_COLUMNS,
        ]
        if self.target_column in borrower_frame:
            selected_columns.append(self.target_column)

        return borrower_frame[selected_columns]

    @staticmethod
    def _mode_or_unknown(values: pd.Series) -> object:
        mode = values.dropna().mode()
        return mode.iloc[0] if not mode.empty else "Unknown"

    @staticmethod
    def _categorical_defaults() -> set[str]:
        return {
            "income_source",
            "sacco_membership",
            "location",
            "preferred_network",
            "preferred_channel",
        }


def build_static_dataset(
    data: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Build one static feature vector per borrower.

    Args:
        data: Raw transaction dataframe or engineered borrower-month frame.

    Returns:
        ``X_static``, borrower IDs, and optional labels.
    """

    dataset = StaticDatasetBuilder().build(data)
    return dataset.X_static, dataset.borrower_ids, dataset.labels

