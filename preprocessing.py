"""Data preprocessing pipeline for credit risk assessment.

The pipeline loads raw borrower transaction datasets, validates schema
assumptions, cleans invalid records, detects outliers, encodes categorical
features, and scales numerical values.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    BORROWER_ID_COLUMN,
    PROCESSED_DATA_DIR,
    TARGET_ALIASES,
    TARGET_COLUMN,
    TRANSACTION_DATE_COLUMN,
)


LOGGER = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Reusable preprocessing pipeline for raw borrower transaction data.

    Args:
        input_path: Optional source CSV, XLS, or XLSX path.
        output_path: Optional destination path for processed data.
        required_columns: Columns that must exist before preprocessing.
        categorical_columns: Optional explicit categorical feature columns.
        numerical_columns: Optional explicit numerical feature columns.
        impossible_value_rules: Optional column rules, expressed as
            ``{"column": {"min": 0, "max": 100}}``. Rows violating these
            limits are removed during cleaning.
    """

    def __init__(
        self,
        input_path: str | Path | None = None,
        output_path: str | Path | None = None,
        required_columns: list[str] | None = None,
        categorical_columns: list[str] | None = None,
        numerical_columns: list[str] | None = None,
        impossible_value_rules: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.input_path = Path(input_path) if input_path else None
        self.output_path = Path(output_path) if output_path else None
        self.required_columns = required_columns or [
            BORROWER_ID_COLUMN,
            TRANSACTION_DATE_COLUMN,
            TARGET_COLUMN,
        ]
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.impossible_value_rules = impossible_value_rules or {}

        self.data: pd.DataFrame | None = None
        self.processed_data: pd.DataFrame | None = None
        self.outlier_report_: dict[str, dict[str, float | int]] = {}
        self.transformer_: ColumnTransformer | None = None

    def load_data(self, path: str | Path | None = None) -> pd.DataFrame:
        """Load a CSV or Excel dataset.

        Args:
            path: Optional path overriding the path supplied at initialization.

        Returns:
            Loaded raw dataframe.

        Raises:
            ValueError: If no input path is supplied or the extension is not
                supported.
            FileNotFoundError: If the dataset path does not exist.
        """

        dataset_path = Path(path) if path else self.input_path
        if dataset_path is None:
            raise ValueError("An input path is required to load data.")
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        suffix = dataset_path.suffix.lower()
        LOGGER.info("Loading raw dataset from %s", dataset_path)
        if suffix == ".csv":
            self.data = pd.read_csv(dataset_path)
        elif suffix in {".xls", ".xlsx"}:
            self.data = pd.read_excel(dataset_path)
        else:
            raise ValueError(
                "Unsupported dataset format. Use CSV, XLS, or XLSX files."
            )

        self.data = self._normalize_target_column(self.data)
        LOGGER.info("Loaded dataset with shape %s", self.data.shape)
        return self.data

    def validate(self, data: pd.DataFrame | None = None) -> pd.DataFrame:
        """Validate required columns and basic schema expectations.

        Args:
            data: Optional dataframe overriding the currently loaded data.

        Returns:
            Validated dataframe.

        Raises:
            ValueError: If there is no dataframe or required columns are
                missing.
        """

        frame = self._resolve_data(data)
        missing_columns = [
            column for column in self.required_columns if column not in frame
        ]
        if missing_columns:
            raise ValueError(
                "Dataset is missing required columns: "
                + ", ".join(missing_columns)
            )

        LOGGER.info("Validated required columns: %s", self.required_columns)
        return frame

    def clean(self, data: pd.DataFrame | None = None) -> pd.DataFrame:
        """Clean raw data before model-ready preprocessing.

        Cleaning removes duplicates, converts transaction dates, handles
        missing values, removes records that violate impossible-value rules,
        and adds IQR-based outlier flags for numeric columns.

        Args:
            data: Optional dataframe overriding the currently loaded data.

        Returns:
            Cleaned dataframe.
        """

        frame = self._resolve_data(data)
        frame = self._normalize_target_column(frame)
        frame = self.validate(frame).copy()
        initial_rows = len(frame)
        frame = frame.drop_duplicates()
        LOGGER.info("Removed %s duplicate rows", initial_rows - len(frame))

        if TRANSACTION_DATE_COLUMN in frame:
            frame[TRANSACTION_DATE_COLUMN] = pd.to_datetime(
                frame[TRANSACTION_DATE_COLUMN],
                errors="coerce",
            )

        frame = self._remove_invalid_required_values(frame)
        frame = self._remove_impossible_values(frame)
        frame = self._fill_missing_values(frame)
        frame = self._detect_outliers(frame)

        if {BORROWER_ID_COLUMN, TRANSACTION_DATE_COLUMN}.issubset(frame.columns):
            frame = frame.sort_values(
                [BORROWER_ID_COLUMN, TRANSACTION_DATE_COLUMN],
                kind="mergesort",
            ).reset_index(drop=True)

        self.data = frame
        LOGGER.info("Cleaned dataset shape is %s", frame.shape)
        return frame

    def preprocess(self, data: pd.DataFrame | None = None) -> pd.DataFrame:
        """Run the full preprocessing workflow.

        The returned dataframe preserves borrower ID, transaction date, and
        target columns while transforming feature columns.

        Args:
            data: Optional dataframe overriding the currently loaded data.

        Returns:
            Model-ready dataframe with encoded and scaled features.
        """

        cleaned = self.clean(data)
        passthrough_columns = [
            column
            for column in [
                BORROWER_ID_COLUMN,
                TRANSACTION_DATE_COLUMN,
                TARGET_COLUMN,
            ]
            if column in cleaned
        ]
        feature_data = cleaned.drop(columns=passthrough_columns)

        numerical_columns = self._select_numerical_columns(feature_data)
        categorical_columns = self._select_categorical_columns(feature_data)

        LOGGER.info("Scaling numerical columns: %s", numerical_columns)
        LOGGER.info("Encoding categorical columns: %s", categorical_columns)

        self.transformer_ = ColumnTransformer(
            transformers=[
                ("numeric", StandardScaler(), numerical_columns),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_columns,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        transformed = self.transformer_.fit_transform(feature_data)
        transformed_columns = self.transformer_.get_feature_names_out()
        transformed_frame = pd.DataFrame(
            transformed,
            columns=transformed_columns,
            index=cleaned.index,
        )

        self.processed_data = pd.concat(
            [
                cleaned[passthrough_columns].reset_index(drop=True),
                transformed_frame.reset_index(drop=True),
            ],
            axis=1,
        )
        LOGGER.info("Processed dataset shape is %s", self.processed_data.shape)
        return self.processed_data

    def save_processed(
        self,
        output_path: str | Path | None = None,
        data: pd.DataFrame | None = None,
    ) -> Path:
        """Save processed data to disk.

        Args:
            output_path: Optional destination path. Defaults to
                ``data/processed/processed_data.csv``.
            data: Optional dataframe to save. Defaults to ``processed_data``.

        Returns:
            Path where the processed dataset was saved.

        Raises:
            ValueError: If there is no processed dataframe to save.
        """

        frame = data if data is not None else self.processed_data
        if frame is None:
            raise ValueError("No processed data available to save.")

        destination = (
            Path(output_path)
            if output_path
            else self.output_path
            or PROCESSED_DATA_DIR / "processed_data.csv"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.suffix.lower() in {".xls", ".xlsx"}:
            frame.to_excel(destination, index=False)
        else:
            frame.to_csv(destination, index=False)

        LOGGER.info("Saved processed dataset to %s", destination)
        return destination

    def _resolve_data(self, data: pd.DataFrame | None) -> pd.DataFrame:
        frame = data if data is not None else self.data
        if frame is None:
            raise ValueError("No data available. Call load_data() first.")
        return frame

    def _normalize_target_column(self, frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame
        if TARGET_COLUMN not in normalized:
            for alias in TARGET_ALIASES:
                if alias in normalized.columns:
                    normalized = normalized.rename(columns={alias: TARGET_COLUMN})
                    LOGGER.info(
                        "Renamed target column %s to %s",
                        alias,
                        TARGET_COLUMN,
                    )
                    break
        return normalized

    def _remove_invalid_required_values(
        self,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        before = len(frame)
        required_subset = [
            column for column in self.required_columns if column in frame.columns
        ]
        frame = frame.dropna(subset=required_subset)
        LOGGER.info(
            "Removed %s rows with missing required values",
            before - len(frame),
        )
        return frame

    def _remove_impossible_values(self, frame: pd.DataFrame) -> pd.DataFrame:
        cleaned = frame
        for column, rule in self.impossible_value_rules.items():
            if column not in cleaned:
                LOGGER.warning(
                    "Skipping impossible-value rule for missing column %s",
                    column,
                )
                continue

            before = len(cleaned)
            if "min" in rule:
                cleaned = cleaned[cleaned[column] >= rule["min"]]
            if "max" in rule:
                cleaned = cleaned[cleaned[column] <= rule["max"]]
            LOGGER.info(
                "Removed %s rows violating impossible-value rule for %s",
                before - len(cleaned),
                column,
            )

        return cleaned

    def _fill_missing_values(self, frame: pd.DataFrame) -> pd.DataFrame:
        filled = frame.copy()
        numeric_columns = filled.select_dtypes(include="number").columns
        categorical_columns = filled.select_dtypes(
            include=["object", "category", "bool"]
        ).columns

        for column in numeric_columns:
            median_value = filled[column].median()
            if pd.isna(median_value):
                median_value = 0
            filled[column] = filled[column].fillna(median_value)

        for column in categorical_columns:
            mode_value = filled[column].mode(dropna=True)
            fill_value: Any = mode_value.iloc[0] if not mode_value.empty else "Unknown"
            filled[column] = filled[column].fillna(fill_value)

        return filled

    def _detect_outliers(self, frame: pd.DataFrame) -> pd.DataFrame:
        annotated = frame.copy()
        numeric_columns = annotated.select_dtypes(include="number").columns

        self.outlier_report_ = {}
        for column in numeric_columns:
            if column == TARGET_COLUMN:
                continue

            q1 = annotated[column].quantile(0.25)
            q3 = annotated[column].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                lower_bound = upper_bound = float(q1)
                outlier_mask = pd.Series(False, index=annotated.index)
            else:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outlier_mask = (annotated[column] < lower_bound) | (
                    annotated[column] > upper_bound
                )

            flag_column = f"{column}_is_outlier"
            annotated[flag_column] = outlier_mask.astype(int)
            self.outlier_report_[column] = {
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "count": int(outlier_mask.sum()),
            }

        LOGGER.info("Generated IQR outlier report for numeric columns")
        return annotated

    def _select_numerical_columns(self, frame: pd.DataFrame) -> list[str]:
        if self.numerical_columns is not None:
            return [column for column in self.numerical_columns if column in frame]
        return list(frame.select_dtypes(include="number").columns)

    def _select_categorical_columns(self, frame: pd.DataFrame) -> list[str]:
        if self.categorical_columns is not None:
            return [
                column for column in self.categorical_columns if column in frame
            ]
        return list(
            frame.select_dtypes(include=["object", "category", "bool"]).columns
        )

