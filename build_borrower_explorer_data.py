#!/usr/bin/env python
"""Assemble a single JSON payload for the interactive borrower-explorer
artifact: per-borrower risk scores, attention-by-month, top IG drivers
(with reliability flag), and PCA embedding coordinates, for both the
hybrid and transformer interpretability twins.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from config import RANDOM_SEED
from run_experiment_evaluation_comparison import OUTPUT_DIR

INTERP_DIR = OUTPUT_DIR / "interpretability"
IG_DIR = OUTPUT_DIR / "integrated_gradients"
OUT_PATH = (
    Path(__file__).resolve().parent
    / "models" / "uganda_mobile_money_evaluation_comparison" / "interpretability_report"
    / "data" / "borrower_explorer_data.json"
)

RELIABLE_DELTA_THRESHOLD = 1.0


def load_model_data(model_name: str, n_layers: int) -> dict:
    oof_true = np.load(INTERP_DIR / model_name / "oof_true.npy").astype(int)
    oof_risk = np.load(INTERP_DIR / model_name / "oof_risk.npy")
    attention_layers = [
        np.load(INTERP_DIR / model_name / f"oof_attention_layer{i + 1}.npy")
        for i in range(n_layers)
    ]
    fused_embedding = np.load(INTERP_DIR / model_name / "oof_fused_embedding.npy")

    deltas = np.load(IG_DIR / model_name / "convergence_deltas.npy")
    computed = ~np.isnan(deltas)
    reliable = computed & (np.abs(deltas) < RELIABLE_DELTA_THRESHOLD)

    local_explanations = json.loads((IG_DIR / model_name / f"ig_local_explanations_{model_name}.json").read_text())
    drivers_by_index = {entry["borrower_index"]: entry["main_drivers"] for entry in local_explanations}

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(fused_embedding)

    return {
        "true": oof_true,
        "risk": oof_risk,
        "attention_layers": attention_layers,
        "reliable": reliable,
        "drivers_by_index": drivers_by_index,
        "pca_coords": coords,
    }


def main() -> None:
    hybrid = load_model_data("hybrid", n_layers=1)
    transformer = load_model_data("transformer", n_layers=2)

    n = len(hybrid["true"])
    explained_index = sorted(set(hybrid["drivers_by_index"]) | set(transformer["drivers_by_index"]))

    payload = {
        "meta": {
            "n_total": n,
            "n_positives": int(hybrid["true"].sum()),
            "n_explained": len(explained_index),
        },
        "pca": {
            "hybrid": {
                "all_points": hybrid["pca_coords"].round(3).tolist(),
                "labels": hybrid["true"].tolist(),
            },
            "transformer": {
                "all_points": transformer["pca_coords"].round(3).tolist(),
                "labels": transformer["true"].tolist(),
            },
        },
        "borrowers": [],
    }

    for index in explained_index:
        entry = {
            "index": int(index),
            "true_label": int(hybrid["true"][index]),
            "hybrid": {
                "risk": round(float(hybrid["risk"][index]), 4),
                "attention_by_month": [round(float(v), 5) for v in hybrid["attention_layers"][0][index].mean(axis=0)],
                "pca": hybrid["pca_coords"][index].round(3).tolist(),
                "ig_reliable": bool(hybrid["reliable"][index]),
                "ig_drivers": hybrid["drivers_by_index"].get(index, []),
            },
            "transformer": {
                "risk": round(float(transformer["risk"][index]), 4),
                "attention_by_month_layer1": [round(float(v), 5) for v in transformer["attention_layers"][0][index].mean(axis=0)],
                "attention_by_month_layer2": [round(float(v), 5) for v in transformer["attention_layers"][1][index].mean(axis=0)],
                "pca": transformer["pca_coords"][index].round(3).tolist(),
                "ig_reliable": bool(transformer["reliable"][index]),
                "ig_drivers": transformer["drivers_by_index"].get(index, []),
            },
        }
        payload["borrowers"].append(entry)

    OUT_PATH.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KB), {len(explained_index)} borrowers")


if __name__ == "__main__":
    main()
