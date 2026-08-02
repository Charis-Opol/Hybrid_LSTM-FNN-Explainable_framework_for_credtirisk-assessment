#!/usr/bin/env python
"""Render BertViz visualizations for the transformer and hybrid models:
per-borrower attention over the 12 monthly time steps ("M1".."M12" as
BertViz "tokens"), and a repurposed view of which raw features drove the
decision (see feature_attention_tokens_and_matrix -- not real attention,
there's no mechanism between individual features in either model).

BertViz's generated HTML depends on three CDN-hosted libraries at render
time (require.js, d3.js, jquery) -- fine inside a live notebook, but this
project's other interpretability artifacts (borrower_explorer.html) are
fully self-contained and work offline. To match that, this script:
  1. Uses locally-fetched copies of require.js/d3.js/jquery inlined as
     plain <script> tags.
  2. Rewrites each visualization's `requirejs(['jquery','d3'], function ($,
     d3) { ... });` wrapper (BertViz's AMD-style module loading, which
     needs require.js's async loader) into a plain IIFE `(function ($, d3)
     { ... })(window.jQuery, window.d3);` that runs against the globals
     the plain <script> tags already defined -- avoiding require.js (and
     its own CDN dependency) entirely.
"""

from __future__ import annotations

import re
from pathlib import Path

import attention_visualization  # noqa: F401  registers PaddingMaskLayer/ExpandMaskLayer/MaskedGlobalAveragePooling1D
import numpy as np
import tensorflow as tf
import torch
from bertviz import head_view, model_view
from sklearn.model_selection import StratifiedKFold

from config import RANDOM_SEED
from run_experiment_evaluation_comparison import DATA_PATH, build_training_arrays, load_and_normalize_dataset

INTERP_DIR = Path("models/uganda_mobile_money_evaluation_comparison/interpretability")
IG_DIR = Path("models/uganda_mobile_money_evaluation_comparison/integrated_gradients")
OUT_DIR = Path("models/uganda_mobile_money_evaluation_comparison/interpretability_report")
ASSETS_DIR = Path(
    r"C:\Users\chari\AppData\Local\Temp\claude\c--Users-chari-Desktop-Hybrid-LSTM-FNN-Explainable-framework-for-credtirisk-assessment"
    r"\1392b5e8-a49c-4df6-add1-25f1f4f2774c\scratchpad"
)
MONTHS = [f"M{i}" for i in range(1, 13)]


def load_model(name: str, n_layers: int):
    true = np.load(INTERP_DIR / name / "oof_true.npy").astype(int)
    risk = np.load(INTERP_DIR / name / "oof_risk.npy")
    # oof_attention_layer*.npy are already head-averaged (used for the static plots);
    # BertViz needs genuine per-head weights, so those are recomputed fresh below.
    return true, risk


def fold_holding_out(index: int, X_static: np.ndarray, labels: np.ndarray) -> int:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    for fold_number, (_, holdout_index) in enumerate(skf.split(X_static, labels), start=1):
        if index in holdout_index:
            return fold_number
    raise ValueError(f"borrower {index} not found in any holdout fold")


def per_head_attention(model_name: str, index: int, fold_number: int, X_temporal, X_static, n_layers: int) -> list[torch.Tensor]:
    """Run the borrower through the fold model that held it out to get the
    real per-head attention tensors (list of (1, heads, 12, 12) per layer).
    """
    model = tf.keras.models.load_model(
        INTERP_DIR / model_name / f"fold_{fold_number}" / "trained_model_with_attention.keras",
        compile=False, safe_mode=False,
    )
    outputs = model.predict([X_temporal[index][None, ...], X_static[index][None, ...]], verbose=0)
    # outputs = [risk_score, attention_layer_1, ..., attention_layer_n, fused_embedding]
    attention_layers = outputs[1:1 + n_layers]
    return [torch.from_numpy(np.asarray(layer)).float() for layer in attention_layers]


def reliable_ig_borrowers(model_name: str) -> np.ndarray:
    deltas = np.load(IG_DIR / model_name / "convergence_deltas.npy")
    return ~np.isnan(deltas) & (np.abs(deltas) < 1.0)


def feature_attention_tokens_and_matrix(model_name: str, index: int, temporal_names: list[str], static_names: list[str]):
    """BertViz visualizes pairwise attention between sequence positions --
    there's no such mechanism between individual raw features anywhere in
    these models (MultiHeadAttention only runs over the 12 monthly time
    steps). This repurposes BertViz's sentence-pair cross-attention view
    (normally "sentence A -> sentence B") as a "[DECISION] -> feature"
    view instead: a single virtual token attending to each of the 47 raw
    features, with the "attention weight" being that feature's Integrated
    Gradients attribution for this borrower (not real self-attention).
    Two heads carry direction, since IG attributions are signed and
    BertViz's colour scale assumes non-negative weights:
      head 0 = features that increased predicted risk
      head 1 = features that decreased predicted risk
    Each head is normalized by its own max magnitude for this borrower.
    """
    ig_temporal = np.load(IG_DIR / model_name / "ig_temporal.npy")[index]  # (12, 35)
    ig_static = np.load(IG_DIR / model_name / "ig_static.npy")[index]  # (12,)

    temporal_attribution = ig_temporal.sum(axis=0)  # collapse months -> one value per raw temporal feature
    combined = np.concatenate([temporal_attribution, ig_static])
    combined_names = list(temporal_names) + list(static_names)

    positive = np.clip(combined, 0, None)
    negative = np.clip(-combined, 0, None)
    positive = positive / max(positive.max(), 1e-9)
    negative = negative / max(negative.max(), 1e-9)

    n_features = len(combined_names)
    n_tokens = n_features + 1  # +1 for the [DECISION] token at position 0
    attention = np.zeros((1, 2, n_tokens, n_tokens), dtype=np.float32)
    attention[0, 0, 0, 1:] = positive
    attention[0, 1, 0, 1:] = negative

    tokens = ["[DECISION]"] + combined_names
    return tokens, [torch.from_numpy(attention)]


def rewrite_for_offline(html: str) -> str:
    """Strip the require.js <script src> tag and require.config call, and
    convert the requirejs(['jquery','d3'], function ($, d3) {...}); wrapper
    into a plain IIFE run against already-loaded globals.
    """
    html = re.sub(r'<script src="https://cdnjs\.cloudflare\.com/ajax/libs/require\.js/[^"]*"></script>', "", html)
    html = re.sub(r"require\.config\(\{[^}]*?\}\s*\}\s*\)\s*;", "", html, flags=re.S)
    html = re.sub(
        r"requirejs\(\['jquery',\s*'d3'\],\s*function\s*\(\s*\$,\s*d3\s*\)\s*\{",
        "(function ($, d3) {",
        html,
    )
    # The outermost requirejs(...) call is closed by the final "});" in the script.
    last_close = html.rfind("});")
    if last_close != -1:
        html = html[:last_close] + "})(window.jQuery, window.d3);" + html[last_close + 3:]
    return html


def section(title: str, subtitle: str, html: str) -> str:
    return f"""
    <section class="viz-section">
      <h2 class="serif">{title}</h2>
      <p class="subtitle">{subtitle}</p>
      <div class="viz-frame">{html}</div>
    </section>
    """


def main() -> None:
    hybrid_true, hybrid_risk = load_model("hybrid", n_layers=1)
    transformer_true, transformer_risk = load_model("transformer", n_layers=2)

    data = load_and_normalize_dataset(DATA_PATH)
    data = data.sort_values(["borrower_id", "transaction_date"]).reset_index(drop=True)
    X_temporal, X_static, labels, _, temporal_names, static_names, _ = build_training_arrays(data)

    pos = np.where(transformer_true == 1)[0]
    neg = np.where(transformer_true == 0)[0]
    defaulter = int(pos[np.argmax(transformer_risk[pos])])
    non_defaulter = int(neg[np.argmin(transformer_risk[neg])])

    # Feature-attention view needs a borrower with reliable IG convergence
    # for *both* models (see 02_integrated_gradients_study.md -- most
    # transformer borrowers don't converge), so it's picked separately.
    both_reliable = reliable_ig_borrowers("hybrid") & reliable_ig_borrowers("transformer")
    reliable_index = np.where(both_reliable)[0]
    reliable_pos = reliable_index[transformer_true[reliable_index] == 1]
    reliable_neg = reliable_index[transformer_true[reliable_index] == 0]
    ig_defaulter = int(reliable_pos[np.argmax(transformer_risk[reliable_pos])])
    ig_non_defaulter = int(reliable_neg[np.argmin(transformer_risk[reliable_neg])])

    sections = []
    for label, index in [("Defaulter", defaulter), ("Non-defaulter", non_defaulter)]:
        t_risk, h_risk = transformer_risk[index], hybrid_risk[index]
        fold_number = fold_holding_out(index, X_static, labels)

        transformer_attn = per_head_attention("transformer", index, fold_number, X_temporal, X_static, n_layers=2)
        hybrid_attn = per_head_attention("hybrid", index, fold_number, X_temporal, X_static, n_layers=1)

        hv = head_view(attention=transformer_attn, tokens=MONTHS, html_action="return")
        sections.append(section(
            f"Transformer — {label} (borrower #{index})",
            f"Transformer risk score {t_risk:.3f}. Toggle layers/heads above; the visualization "
            f"lines up with the attention-matrix study — column-dominant, not query-dependent.",
            rewrite_for_offline(hv.data),
        ))

        mv = model_view(attention=transformer_attn, tokens=MONTHS, html_action="return")
        sections.append(section(
            f"Transformer, all layers/heads at once — {label} (borrower #{index})",
            "Every layer x head grid cell for this borrower in one view.",
            rewrite_for_offline(mv.data),
        ))

        hv_hybrid = head_view(attention=hybrid_attn, tokens=MONTHS, html_action="return")
        sections.append(section(
            f"Hybrid GRU-Attention — {label} (borrower #{index})",
            f"Hybrid risk score {h_risk:.3f}. Only one attention layer (2 heads) -- and per the "
            f"ablation study, this branch barely reaches the hybrid model's final prediction.",
            rewrite_for_offline(hv_hybrid.data),
        ))

    feature_view_note = (
        "Not real attention -- there's no mechanism anywhere in these models that attends between "
        "individual raw features (MultiHeadAttention only runs over the 12 monthly time steps). This "
        "reuses BertViz's sentence-pair cross-attention view as a [DECISION] token → 47 raw-feature "
        "view instead, with “attention weight” = this borrower's Integrated Gradients attribution, "
        "normalized 0-1. After it loads, pick “Sentence A -> Sentence B” from the dropdown "
        "(the default “All” view includes an empty, meaningless feature→feature panel). "
        "Head 0 = features that increased predicted risk, head 1 = features that decreased it."
    )
    for label, index in [("Defaulter", ig_defaulter), ("Non-defaulter", ig_non_defaulter)]:
        for model_name, risk_array in [("hybrid", hybrid_risk), ("transformer", transformer_risk)]:
            tokens, attn = feature_attention_tokens_and_matrix(model_name, index, temporal_names, static_names)
            hv_feat = head_view(attention=attn, tokens=tokens, sentence_b_start=1, html_action="return")
            sections.append(section(
                f"{model_name.capitalize()} — features driving the decision — {label} (borrower #{index})",
                f"Risk score {risk_array[index]:.3f}. {feature_view_note}",
                rewrite_for_offline(hv_feat.data),
            ))

    jquery_js = (ASSETS_DIR / "jquery.min.js").read_text(encoding="utf-8")
    d3_js = (ASSETS_DIR / "d3.min.js").read_text(encoding="utf-8")

    page = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>BertViz Attention Views</title>
<style>
  :root {{
    --ink: #1b1f27; --paper: #f6f2ea; --paper-raised: #ffffff;
    --accent: #b9812f; --line: #d8d0c0; --text: #2a2620; --text-dim: #6b6355;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --ink: #eae4d8; --paper: #17181b; --paper-raised: #1f2024;
      --accent: #d9a352; --line: #34353a; --text: #e7e1d4; --text-dim: #9b9486; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--paper); color: var(--text);
    font-family: "Segoe UI", Helvetica, Arial, sans-serif; padding: 28px 32px 80px; }}
  .serif {{ font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif; }}
  h1 {{ font-size: 22px; margin: 0 0 6px; }}
  .intro {{ color: var(--text-dim); max-width: 70ch; font-size: 13px; margin-bottom: 28px; }}
  .viz-section {{ background: var(--paper-raised); border: 1px solid var(--line); border-radius: 10px;
    padding: 18px 22px; margin-bottom: 22px; }}
  .viz-section h2 {{ font-size: 16px; margin: 0 0 2px; }}
  .viz-section .subtitle {{ font-size: 12px; color: var(--text-dim); margin: 0 0 14px; }}
  .viz-frame {{ overflow-x: auto; }}
</style>
</head>
<body>
  <h1 class="serif">BertViz Attention Views</h1>
  <p class="intro">Two kinds of view. The month views are the same underlying attention weights as the
  static attention-matrix study, rendered with BertViz's interactive head/model view -- tokens
  relabeled as months (M1 = oldest in the 12-month window, M12 = most recent). The
  "features driving the decision" views are a different thing: there's no real attention between
  raw features anywhere in these models, so those repurpose BertViz's sentence-pair display to show
  a [DECISION] token "attending to" each feature's Integrated Gradients attribution instead --
  see the note on each section. Self-contained: jQuery and D3 are inlined below, no network access
  needed to view.</p>
  <script>{jquery_js}</script>
  <script>{d3_js}</script>
  {''.join(sections)}
</body>
</html>
"""

    out_path = OUT_DIR / "bertviz_attention_views.html"
    out_path.write_text(page, encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
