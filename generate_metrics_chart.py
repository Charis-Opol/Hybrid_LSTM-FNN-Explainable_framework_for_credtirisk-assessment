"""Quick script to generate performance metrics bar chart."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Load evaluation metrics
eval_json = Path("models/uganda_mobile_money/evaluation/validation/evaluation.json")
with open(eval_json) as f:
    metrics = json.load(f)

# Create bar chart
fig, ax = plt.subplots(figsize=(11, 7))

metric_names = list(metrics.keys())
metric_values = list(metrics.values())

# Use distinct colors
colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E", "#BC4749"]

bars = ax.bar(metric_names, metric_values, color=colors[:len(metric_names)], 
               alpha=0.85, edgecolor="black", linewidth=2)

# Add value labels on top of bars
for bar, value in zip(bars, metric_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
            f'{value:.4f}', ha='center', va='bottom', fontsize=11, fontweight="bold")

ax.set_ylim(0, 1.1)
ax.set_ylabel("Score", fontsize=13, fontweight="bold")
ax.set_xlabel("Performance Metric", fontsize=13, fontweight="bold")
ax.set_title("Validation Set - Performance Metrics Summary", fontsize=15, fontweight="bold", pad=20)
ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=1)
ax.set_axisbelow(True)

# Rotate x labels for better readability
plt.xticks(rotation=15, ha='right')

fig.tight_layout()
plt.savefig("models/uganda_mobile_money/plots/performance_metrics.png", 
            dpi=300, bbox_inches="tight")
plt.close()

print("Performance metrics bar chart saved!")
