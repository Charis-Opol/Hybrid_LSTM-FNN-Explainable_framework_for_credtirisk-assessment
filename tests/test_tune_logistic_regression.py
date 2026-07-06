"""Regression tests for the logistic regression tuning grid."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tune_logistic_regression import PARAM_GRID  # noqa: E402


def test_param_grid_explicitly_sets_penalty_and_l1_ratio_per_group() -> None:
    """Each grid block should carry its own penalty and l1_ratio settings."""

    assert len(PARAM_GRID) == 3

    assert PARAM_GRID[0]["classifier__penalty"] == ["l2"]
    assert PARAM_GRID[0]["classifier__l1_ratio"] == [0.0]
    assert PARAM_GRID[0]["classifier__solver"] == ["lbfgs"]

    assert PARAM_GRID[1]["classifier__penalty"] == ["l1"]
    assert PARAM_GRID[1]["classifier__l1_ratio"] == [1.0]
    assert PARAM_GRID[1]["classifier__solver"] == ["liblinear"]

    assert PARAM_GRID[2]["classifier__penalty"] == ["elasticnet"]
    assert PARAM_GRID[2]["classifier__l1_ratio"] == [0.25, 0.5, 0.75]
    assert PARAM_GRID[2]["classifier__solver"] == ["saga"]
