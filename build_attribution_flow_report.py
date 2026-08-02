#!/usr/bin/env python
"""Two views of model "reasoning" that are honest about what these models
actually are (no LLM-style chain-of-thought exists here):

1. Attribution-flow Sankey diagrams (hybrid vs. transformer) -- how much
   of each branch's information actually reaches the final prediction.
   Flow widths use the ablation-based branch shares (reliable for both
   models), not Integrated Gradients (unreliable for the transformer --
   see 02_integrated_gradients_study.md).
2. An XGBoost decision path -- for tree-based models, a literal reasoning
   chain genuinely exists (each split is an if/else on a real feature).
   Traces one borrower's actual path through one tree of the ensemble.
"""

from __future__ import annotations

import json
from pathlib import Path

import attention_visualization  # noqa: F401
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from XGBoost_baseline import flatten_temporal
from config import RANDOM_SEED
from run_experiment_evaluation_comparison import DATA_PATH, build_training_arrays, load_and_normalize_dataset

INTERP_DIR = Path("models/uganda_mobile_money_evaluation_comparison/interpretability")
OUT_DIR = Path("models/uganda_mobile_money_evaluation_comparison/interpretability_report")


def compute_ablation_shares(X_temporal, X_static, labels):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    train_index, holdout_index = next(iter(skf.split(X_static, labels)))
    baseline_t = X_temporal[train_index].mean(axis=0)
    baseline_s = X_static[train_index].mean(axis=0)
    Xt, Xs = X_temporal[holdout_index], X_static[holdout_index]

    shares = {}
    for model_name in ["hybrid", "transformer"]:
        model = tf.keras.models.load_model(
            INTERP_DIR / model_name / "fold_1" / "trained_model_with_attention.keras",
            compile=False, safe_mode=False,
        )
        out_real = np.array(model([Xt, Xs], training=False)[0]).ravel()
        Xt_ablated = np.tile(baseline_t, (len(Xt), 1, 1)).astype("float32")
        out_no_temporal = np.array(model([Xt_ablated, Xs], training=False)[0]).ravel()
        Xs_ablated = np.tile(baseline_s, (len(Xt), 1)).astype("float32")
        out_no_static = np.array(model([Xt, Xs_ablated], training=False)[0]).ravel()

        temporal_share = 1 - np.corrcoef(out_real, out_no_temporal)[0, 1]
        static_share = 1 - np.corrcoef(out_real, out_no_static)[0, 1]
        total = temporal_share + static_share
        shares[model_name] = {"temporal": temporal_share / total, "static": static_share / total}
    return shares


def train_xgboost_fold(X, labels):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    train_index, holdout_index = next(iter(skf.split(X, labels)))
    X_train, y_train = X[train_index], labels[train_index]
    scale_pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)
    return model, train_index, holdout_index


def trace_tree_path(tree: dict, sample: np.ndarray) -> list[int]:
    """Return the list of nodeids visited from root to the leaf ``sample`` falls into."""
    path = [tree["nodeid"]]
    node = tree
    while "leaf" not in node:
        feature_index = int(node["split"][1:])
        value = sample[feature_index]
        go_missing = np.isnan(value)
        go_yes = (not go_missing) and value < node["split_condition"]
        target_id = node["missing"] if go_missing else (node["yes"] if go_yes else node["no"])
        node = next(child for child in node["children"] if child["nodeid"] == target_id)
        path.append(node["nodeid"])
    return path


def flatten_tree(tree: dict, feature_names: list[str], parent=None, depth=0, out=None):
    if out is None:
        out = []
    entry = {
        "nodeid": tree["nodeid"],
        "depth": depth,
        "parent": parent,
        "is_leaf": "leaf" in tree,
    }
    if "leaf" in tree:
        entry["leaf_value"] = tree["leaf"]
    else:
        entry["feature"] = feature_names[int(tree["split"][1:])]
        entry["threshold"] = tree["split_condition"]
        entry["yes"] = tree["yes"]
        entry["no"] = tree["no"]
        entry["missing"] = tree["missing"]
    out.append(entry)
    if "children" in tree:
        for child in tree["children"]:
            flatten_tree(child, feature_names, parent=tree["nodeid"], depth=depth + 1, out=out)
    return out


def main() -> None:
    data = load_and_normalize_dataset(DATA_PATH)
    data = data.sort_values(["borrower_id", "transaction_date"]).reset_index(drop=True)
    X_temporal, X_static, labels, _, temporal_names, static_names, _ = build_training_arrays(data)
    labels = labels.astype(int)

    print("Computing ablation-based branch shares...")
    shares = compute_ablation_shares(X_temporal, X_static, labels)
    print(json.dumps(shares, indent=2))

    print("Training XGBoost fold model...")
    sequence_length = X_temporal.shape[1]
    feature_names = list(static_names) + [
        f"{name}_t{step}" for step in range(sequence_length) for name in temporal_names
    ]
    X_temporal_flat = flatten_temporal(X_temporal)
    X = np.concatenate([X_static, X_temporal_flat], axis=1)
    model, train_index, holdout_index = train_xgboost_fold(X, labels)

    proba = model.predict_proba(X[holdout_index])[:, 1]
    y_holdout = labels[holdout_index]
    positive_local = np.where(y_holdout == 1)[0]
    best_local = positive_local[np.argmax(proba[positive_local])]
    borrower_global_index = int(holdout_index[best_local])
    borrower_proba = float(proba[best_local])
    sample = X[borrower_global_index]

    tree_dump = model.get_booster().get_dump(dump_format="json", with_stats=True)
    tree0 = json.loads(tree_dump[0])
    path = trace_tree_path(tree0, sample)
    flat_nodes = flatten_tree(tree0, feature_names)

    payload = {
        "shares": shares,
        "xgboost": {
            "borrower_index": borrower_global_index,
            "predicted_probability": borrower_proba,
            "n_trees": len(tree_dump),
            "path": path,
            "nodes": flat_nodes,
            "sample_values": {
                node["feature"]: float(sample[feature_names.index(node["feature"])])
                for node in flat_nodes if not node["is_leaf"]
            },
        },
    }

    out_path = OUT_DIR / "data" / "attribution_flow_data.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
