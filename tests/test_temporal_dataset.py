"""Tests for temporal dataset construction."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from temporal_dataset import TemporalDatasetBuilder  # noqa: E402


def test_temporal_builder_pads_to_twelve_months() -> None:
    """Each borrower sequence should have exactly 12 monthly steps."""

    monthly = pd.DataFrame(
        {
            "borrower_id": ["B1", "B1", "B2"],
            "month": ["2024-01", "2024-03", "2024-02"],
            "monthly_inflow": [100.0, 200.0, 300.0],
            "monthly_outflow": [40.0, 50.0, 60.0],
            "defaulted": [0, 0, 1],
        }
    )

    dataset = TemporalDatasetBuilder().build(monthly)

    assert dataset.X_temporal.shape == (2, 12, 2)
    assert dataset.labels.tolist() == [0.0, 1.0]
    assert dataset.mask.shape == (2, 12)

