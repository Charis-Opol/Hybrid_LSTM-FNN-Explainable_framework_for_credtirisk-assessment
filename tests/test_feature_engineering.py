"""Tests for localised feature engineering."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feature_engineering import engineer_features  # noqa: E402


def test_engineer_features_creates_localised_columns() -> None:
    """Feature engineering should return borrower-month localised features."""

    raw = pd.DataFrame(
        {
            "borrower_id": ["B1", "B1", "B1"],
            "transaction_date": ["2024-01-01", "2024-01-15", "2024-06-01"],
            "transaction_amount": [1000, 300, 800],
            "transaction_type": ["deposit", "payment", "deposit"],
            "balance": [1000, 700, 1500],
            "defaulted": [0, 0, 0],
        }
    )

    engineered = engineer_features(raw, analysis_date="2024-06-30")

    assert {"monthly_inflow", "net_cashflow", "school_fees_season"}.issubset(
        engineered.columns
    )
    assert {"financial_stability_score", "credit_behaviour_score"}.issubset(
        engineered.columns
    )
    assert engineered["borrower_id"].nunique() == 1

