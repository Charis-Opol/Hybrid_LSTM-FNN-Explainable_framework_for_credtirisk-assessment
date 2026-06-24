"""Train the hybrid model on the Uganda mobile money sample dataset.

Usage:
    python experiments/train_uganda_mobile_money.py ^
        --data "C:\\Users\\chari\\Downloads\\Uganda_Mobile_Money_Logs_3000.xlsx" ^
        --epochs 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feature_engineering import FeatureColumns, engineer_features  # noqa: E402
from static_dataset import StaticDatasetBuilder  # noqa: E402
from temporal_dataset import TemporalDatasetBuilder  # noqa: E402
from train import train_model  # noqa: E402
from utils import configure_logging  # noqa: E402


DEFAULT_DATA_PATH = Path(
    r"C:\Users\chari\Downloads\Uganda_Mobile_Money_Logs_3000.xlsx"
)


def load_and_normalize_dataset(path: Path) -> pd.DataFrame:
    """Load the workbook and map its columns into framework names."""

    data = pd.read_excel(path, sheet_name="mobile_money_logs")
    return data.rename(
        columns={
            "amount_ugx": "transaction_amount",
            "balance_after_ugx": "balance",
            "default_label": "defaulted",
            "loan_amount_ugx": "loan_amount",
            "sacco_member": "sacco_membership",
            "location_type": "location",
            "network": "preferred_network",
            "channel": "preferred_channel",
        }
    )


def build_training_arrays(data: pd.DataFrame) -> tuple:
    """Engineer features and build aligned temporal/static arrays."""

    feature_columns = FeatureColumns(
        amount="transaction_amount",
        balance="balance",
        target="defaulted",
    )
    engineered = engineer_features(data, columns=feature_columns)

    temporal = TemporalDatasetBuilder().build(engineered)
    static = StaticDatasetBuilder().build(engineered)

    static_by_borrower = {
        borrower_id: row_index
        for row_index, borrower_id in enumerate(static.borrower_ids.tolist())
    }
    static_indices = [
        static_by_borrower[borrower_id]
        for borrower_id in temporal.borrower_ids.tolist()
    ]
    X_static_aligned = static.X_static[static_indices]

    if temporal.labels is None:
        raise ValueError("The dataset must include default labels for training.")

    return (
        temporal.X_temporal,
        X_static_aligned,
        temporal.labels,
        engineered,
        temporal.feature_names,
        static.feature_names,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "uganda_mobile_money",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    """Run the Uganda mobile money training experiment."""

    configure_logging()
    args = parse_args()
    logging.info("Loading dataset from %s", args.data)
    data = load_and_normalize_dataset(args.data)
    X_temporal, X_static, labels, engineered, temporal_names, static_names = (
        build_training_arrays(data)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    engineered.to_csv(args.output_dir / "engineered_features.csv", index=False)
    (args.output_dir / "temporal_feature_names.txt").write_text(
        "\n".join(temporal_names),
        encoding="utf-8",
    )
    (args.output_dir / "static_feature_names.txt").write_text(
        "\n".join(static_names),
        encoding="utf-8",
    )

    artifacts = train_model(
        X_temporal=X_temporal,
        X_static=X_static,
        labels=labels,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    logging.info("Training artifacts: %s", artifacts)


if __name__ == "__main__":
    main()

