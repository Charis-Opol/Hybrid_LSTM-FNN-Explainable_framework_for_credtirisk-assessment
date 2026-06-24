"""Shared utilities for the credit risk framework."""

import logging
import random
from pathlib import Path

import numpy as np
import tensorflow as tf


def configure_logging(level: int = logging.INFO) -> None:
    """Configure project-wide logging.

    Args:
        level: Logging level to use for the root logger.
    """

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not already exist.

    Args:
        path: Directory path to create.

    Returns:
        The same path for fluent usage.
    """

    path.mkdir(parents=True, exist_ok=True)
    return path


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments.

    Args:
        seed: Integer seed used by Python, NumPy, and TensorFlow.
    """

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

