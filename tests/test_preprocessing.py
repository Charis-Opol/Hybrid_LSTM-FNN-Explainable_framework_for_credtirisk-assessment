"""Tests for the preprocessing pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import PreprocessingPipeline  # noqa: E402


def test_preprocess_cleans_sorts_encodes_and_scales() -> None:
    """The pipeline should produce model-ready data from messy input."""

    raw = pd.DataFrame(
        {
            "borrower_id": ["B2", "B1", "B1", "B3", "B4"],
            "transaction_date": [
                "2024-02-01",
                "2024-01-03",
                "2024-01-03",
                "2024-01-01",
                "not-a-date",
            ],
            "defaulted": [0, 1, 1, 0, 1],
            "loan_amount": [1000.0, 1500.0, 1500.0, -50.0, 900.0],
            "channel": ["mobile_money", None, None, "agent", "bank"],
        }
    )
    pipeline = PreprocessingPipeline(
        impossible_value_rules={"loan_amount": {"min": 0}},
    )

    processed = pipeline.preprocess(raw)

    assert processed["borrower_id"].tolist() == ["B1", "B2"]
    assert "loan_amount" in processed.columns
    assert "loan_amount_is_outlier" in processed.columns
    assert any(column.startswith("channel_") for column in processed.columns)
    assert pipeline.outlier_report_["loan_amount"]["count"] == 0


def test_validate_raises_for_missing_required_columns() -> None:
    """Validation should report missing required columns clearly."""

    pipeline = PreprocessingPipeline()
    frame = pd.DataFrame({"borrower_id": ["B1"]})

    with pytest.raises(ValueError, match="missing required columns"):
        pipeline.validate(frame)


def test_load_data_supports_csv(tmp_path: Path) -> None:
    """CSV files should load into the pipeline data attribute."""

    csv_path = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "borrower_id": ["B1"],
            "transaction_date": ["2024-01-01"],
            "defaulted": [0],
        }
    ).to_csv(csv_path, index=False)

    pipeline = PreprocessingPipeline(csv_path)
    loaded = pipeline.load_data()

    assert loaded.shape == (1, 3)
    assert pipeline.data is loaded

