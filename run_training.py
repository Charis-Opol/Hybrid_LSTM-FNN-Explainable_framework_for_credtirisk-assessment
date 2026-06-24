#!/usr/bin/env python
"""Train the hybrid credit risk model on the Uganda mobile money dataset."""

from __future__ import annotations

import logging
from pathlib import Path

from preprocessing import PreprocessingPipeline
from feature_engineering import engineer_features
from temporal_dataset import TemporalDatasetBuilder
from static_dataset import StaticDatasetBuilder
from train import train_model
from config import RAW_DATA_DIR, MODELS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Execute the full training pipeline."""

    # 1. Load and preprocess raw data
    LOGGER.info("=" * 80)
    LOGGER.info("STEP 1: Loading and preprocessing raw data")
    LOGGER.info("=" * 80)
    
    excel_path = RAW_DATA_DIR / "Uganda_Mobile_Money_Logs_3000.xlsx"
    pipeline = PreprocessingPipeline(input_path=excel_path)
    raw_data = pipeline.load_data()
    LOGGER.info(f"Raw data shape: {raw_data.shape}")
    LOGGER.info(f"Columns: {raw_data.columns.tolist()}")
    
    cleaned_data = pipeline.preprocess(raw_data)
    LOGGER.info(f"Cleaned data shape: {cleaned_data.shape}")
    LOGGER.info(f"Number of borrowers: {cleaned_data['borrower_id'].nunique()}")

    # 2. Engineer features
    LOGGER.info("=" * 80)
    LOGGER.info("STEP 2: Engineering localised features")
    LOGGER.info("=" * 80)
    
    engineered_data = engineer_features(cleaned_data)
    LOGGER.info(f"Engineered data shape: {engineered_data.shape}")
    LOGGER.info(f"Columns: {engineered_data.columns.tolist()}")

    # 3. Build temporal dataset
    LOGGER.info("=" * 80)
    LOGGER.info("STEP 3: Building temporal dataset")
    LOGGER.info("=" * 80)
    
    temporal_builder = TemporalDatasetBuilder()
    temporal_dataset = temporal_builder.build(engineered_data)
    LOGGER.info(f"Temporal input shape: {temporal_dataset.X_temporal.shape}")
    LOGGER.info(f"Temporal feature names: {temporal_dataset.feature_names}")
    
    # Save temporal feature names
    features_path = MODELS_DIR / "uganda_mobile_money" / "temporal_feature_names.txt"
    features_path.parent.mkdir(parents=True, exist_ok=True)
    with open(features_path, "w") as f:
        f.write("\n".join(temporal_dataset.feature_names))
    LOGGER.info(f"Saved temporal feature names to {features_path}")

    # 4. Build static dataset
    LOGGER.info("=" * 80)
    LOGGER.info("STEP 4: Building static dataset")
    LOGGER.info("=" * 80)
    
    static_builder = StaticDatasetBuilder()
    static_dataset = static_builder.build(engineered_data)
    LOGGER.info(f"Static input shape: {static_dataset.X_static.shape}")
    LOGGER.info(f"Static feature names: {static_dataset.feature_names}")
    
    # Save static feature names
    static_features_path = MODELS_DIR / "uganda_mobile_money" / "static_feature_names.txt"
    with open(static_features_path, "w") as f:
        f.write("\n".join(static_dataset.feature_names))
    LOGGER.info(f"Saved static feature names to {static_features_path}")

    # 5. Train the model
    LOGGER.info("=" * 80)
    LOGGER.info("STEP 5: Training the hybrid model")
    LOGGER.info("=" * 80)
    
    output_dir = MODELS_DIR / "uganda_mobile_money"
    artifacts = train_model(
        X_temporal=temporal_dataset.X_temporal,
        X_static=static_dataset.X_static,
        labels=temporal_dataset.labels,
        output_dir=output_dir,
        epochs=100,
        batch_size=32,
    )

    LOGGER.info("=" * 80)
    LOGGER.info("TRAINING COMPLETE")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Trained model: {artifacts.trained_model}")
    LOGGER.info(f"Best checkpoint: {artifacts.best_checkpoint}")
    LOGGER.info(f"Training history: {artifacts.training_history}")
    LOGGER.info(f"Evaluation results: {artifacts.evaluation}")
    LOGGER.info(f"History plot: {artifacts.history_plot}")
    LOGGER.info(f"TensorBoard logs: {artifacts.tensorboard_log_dir}")


if __name__ == "__main__":
    main()
