#!/usr/bin/env python
"""Visualize training history from the hybrid model training run."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    history_path = base_dir / "models" / "uganda_mobile_money" / "training_history.csv"
    output_path = base_dir / "models" / "uganda_mobile_money" / "training_history_plot.png"

    history = pd.read_csv(history_path)

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 1, 1)
    plt.plot(history["accuracy"], label="train accuracy")
    plt.plot(history["val_accuracy"], label="val accuracy")
    plt.plot(history["auc"], label="train auc")
    plt.plot(history["val_auc"], label="val auc")
    plt.title("Training History: Accuracy and AUC")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(history["loss"], label="train loss")
    plt.plot(history["val_loss"], label="val loss")
    plt.title("Training History: Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
