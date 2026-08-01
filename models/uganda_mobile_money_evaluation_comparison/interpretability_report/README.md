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
3. `images/attention/` — 8 PNGs backing the attention study.
4. `images/integrated_gradients/` — 4 PNGs backing the IG study.
5. `data/` — raw per-feature IG attribution CSVs.

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
