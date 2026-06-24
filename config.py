"""Project configuration for the credit risk framework.

This module centralizes paths, random seeds, and model defaults used across
the research pipeline.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

RANDOM_SEED = 42
SEQUENCE_LENGTH_MONTHS = 12
TARGET_COLUMN = "defaulted"
TARGET_ALIASES = ["default_label"]
BORROWER_ID_COLUMN = "borrower_id"
TRANSACTION_DATE_COLUMN = "transaction_date"

# TODO: Extend with experiment tracking, training hyperparameters, and
# dataset-specific schemas as implementation progresses.

