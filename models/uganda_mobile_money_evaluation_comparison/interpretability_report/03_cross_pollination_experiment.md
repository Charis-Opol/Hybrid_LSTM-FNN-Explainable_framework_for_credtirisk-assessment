# Cross-Pollination Experiment

## Motivation

The attention and Integrated Gradients studies found that the hybrid model relies almost
entirely on its static branch and the transformer relies almost entirely on its temporal
branch. The natural follow-up: build one model using the branch each architecture actually
depends on — the transformer's self-attention temporal encoder, fused with the hybrid's FNN
static encoder (`cross_pollinated_model.py`) — trained fresh with the same stratified 5-fold
CV, class weighting, and callbacks as every other model in the comparison.

## Result 1: it doesn't beat either parent architecture

| Model | Precision | Recall | F1 | AP | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | 0.133 | 0.500 | 0.210 | 0.122 | 0.887 |
| Transformer-Encoder Hybrid | 0.121 | 0.386 | 0.184 | 0.103 | 0.875 |
| Logistic Regression | 0.117 | 0.386 | 0.179 | 0.103 | 0.859 |
| Cross-Pollinated Reverse (Hybrid-temporal + Transformer-static) | 0.102 | 0.586 | 0.174 | 0.112 | 0.862 |
| Hybrid LSTM-GRU-Attention-FNN | 0.115 | 0.343 | 0.173 | 0.101 | 0.863 |
| Cross-Pollinated (Transformer-temporal + Hybrid-static) | 0.104 | 0.414 | 0.167 | 0.086 | 0.856 |
| Cross-Attention Fusion Hybrid | 0.093 | 0.529 | 0.159 | 0.083 | 0.858 |

The cross-pollinated model lands 5th of 7 (worse than both parents on F1 and AP) — a genuine
negative result for the "assemble the winning parts" hypothesis as a benchmark-improvement
strategy, at least untuned.

## Result 2: why it doesn't work — a clean 2×2

Ablating each branch (replace with the training-fold mean, measure how much predictions move)
on the new model:

| | Ablate temporal → corr. with real predictions | Ablate static → corr. with real predictions |
|---|---|---|
| Cross-Pollinated (Transformer-temporal + Hybrid-static) | 0.512 (changes a lot) | 0.871 (barely changes) |

Temporal still dominates — even now paired with the *other* static branch. To check whether
this is about the self-attention temporal encoder specifically (rather than something about
being "the new pairing"), the reverse combination was also built and tested: the hybrid's
GRU-Attention temporal encoder + the transformer's plain Dense static branch
(`build_cross_pollinated_model_reverse`):

| | Ablate temporal → corr. with real predictions | Ablate static → corr. with real predictions |
|---|---|---|
| Cross-Pollinated Reverse (Hybrid-temporal + Transformer-static) | 0.947 (barely changes) | 0.667 (changes a lot) |

Now static dominates — with a static branch that was, unmodified, the *weaker* signal source in
the original transformer model. Put together, all four combinations tested:

| Temporal encoder | Static encoder | Dominant branch |
|---|---|---|
| GRU + Attention (hybrid) | FNN (hybrid) | **Static** |
| GRU + Attention (hybrid) | Dense (transformer) | **Static** |
| Self-attention (transformer) | Dense (transformer) | **Temporal** |
| Self-attention (transformer) | FNN (hybrid) | **Temporal** |

**GRU-based temporal encoding loses the branch-importance competition every time, regardless of
which static branch it's paired with; self-attention-based temporal encoding wins every time,**
also regardless of pairing. This reframes the original interpretability finding: it isn't that
static features are inherently more informative than the transaction sequence for this task —
it's that in this fusion setup, the self-attention branch consistently dominates joint gradient
training over the GRU branch, a modality/branch-competition effect documented in the
multi-modal-fusion literature (branches with larger or better-conditioned gradients tend to
crowd out weaker-conditioned ones during joint training, independent of which one actually
carries more real-world signal).

## Practical implication

If temporal sequence modeling is meant to be a real contribution of this architecture (not just
present in name), the GRU branch needs deliberate intervention to avoid being starved during
joint training — e.g. pretraining it separately before fusion, an auxiliary loss on the temporal
embedding alone, gradient-balancing between branches, or replacing the GRU with a self-attention
temporal encoder outright (which the benchmark table shows is otherwise a reasonable trade,
performance-wise, on this dataset).

## Interactive companion

[borrower_explorer.html](borrower_explorer.html) — pick any of the 260 explained borrowers and
see the hybrid and transformer columns side by side: attention-by-month, top Integrated
Gradients drivers, and embedding position. The contrast described above (flat/low temporal
attention-strip and static-heavy IG drivers for the hybrid column, vs. a real varying temporal
strip for the transformer column) is visible case-by-case, not just in the aggregate plots.
