# Interpretability Study — Hybrid LSTM-GRU-Attention vs. Transformer-Encoder Hybrid

Dataset: `uganda_mobile_money_master.csv` (3,000 borrowers, 70 positives, 2.3% default rate).
Both models trained with stratified 5-fold CV; every borrower interpreted by the fold model
that held it out (out-of-fold, consistent with how the rest of this project is evaluated).

## Contents

1. [01_attention_study.md](01_attention_study.md) — what each model's `MultiHeadAttention`
   layer(s) actually attend to, and what the learned fused embedding looks like.
2. [02_integrated_gradients_study.md](02_integrated_gradients_study.md) — Integrated Gradients
   feature attribution, implemented natively against the Keras models (Captum is PyTorch-only
   and can't attach to them).
3. [03_cross_pollination_experiment.md](03_cross_pollination_experiment.md) — follow-up
   architecture experiment: pairing each model's *dominant* branch with the other's, and the
   clean 2×2 result showing GRU-based temporal encoding loses the branch-competition regardless
   of which static branch it's paired with.
4. [borrower_explorer.html](borrower_explorer.html) — interactive tool: pick any of the 260
   explained borrowers and compare the hybrid vs. transformer columns side by side (open
   directly in a browser, no server needed).
5. [bertviz_attention_views.html](bertviz_attention_views.html) — two kinds of BertViz view, for
   one representative defaulter and one non-defaulter, both models:
   (a) head-view/model-view of the real per-head attention weights over the 12 months (not the
   head-averaged versions used elsewhere), and
   (b) "features driving the decision" — since neither model has any attention mechanism between
   individual raw features (only between the 12 monthly time steps), this repurposes BertViz's
   sentence-pair display as a [DECISION] → feature view, with attention weight = that borrower's
   Integrated Gradients attribution (two heads: increased-risk / decreased-risk features).
   Self-contained (jQuery/D3 inlined) — BertViz's default output depends on three CDN scripts at
   render time, which this rewrites away so it works offline like everything else here.
6. [attribution_flow.html](attribution_flow.html) — two honest alternatives to a "chain-of-thought"
   view (neither model generates intermediate reasoning tokens the way an LLM does): a Sankey
   diagram of how much of each branch's information reaches the final prediction (ablation-based,
   reliable for both models — reuses the same shares from `03_cross_pollination_experiment.md`),
   and, for XGBoost specifically, one borrower's actual decision path through one tree of the
   ensemble, since a real step-by-step reasoning chain genuinely exists there.
7. `images/attention/` — 8 PNGs backing the attention study.
8. `images/integrated_gradients/` — 4 PNGs backing the IG study.
9. `data/` — raw per-feature IG attribution CSVs, plus the JSON payloads behind the interactive
   explorer and the attribution-flow diagrams.

## Headline finding

The two architectures rely on **almost opposite input branches**, confirmed by two independent
methods (Integrated Gradients and direct ablation):

| | Hybrid (GRU+Attention) | Transformer-Encoder Hybrid |
|---|---|---|
| Ablate temporal branch → correlation with real predictions | 0.985 (barely changes) | 0.664 (changes a lot) |
| Ablate static branch → correlation with real predictions | 0.737 (changes a lot) | 0.955 (barely changes) |

The hybrid model's GRU-Attention branch computes a real, non-trivial attention pattern over
the 12-month window (see the attention study), but that computation barely reaches the final
prediction — the fused-embedding and output layers have effectively learned to rely on the
static (FNN) branch instead. The transformer does the reverse: it actually depends on the
transaction sequence. See the two study documents for full detail, caveats, and evidence.
