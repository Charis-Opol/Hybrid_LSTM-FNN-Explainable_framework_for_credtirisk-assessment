# Model Comparison — Pooled Out-of-Fold Metrics (5-Fold CV)

| Model | Accuracy | Precision | Recall | F1 | Avg. Precision | ROC-AUC | Threshold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Hybrid LSTM-GRU-Attention-FNN | 0.4194 | 0.3505 | 0.9089 | 0.5059 | 0.3535 | 0.5572 | 0.4440 |
| XGBoost | 0.4106 | 0.3477 | 0.9156 | 0.5040 | 0.3544 | 0.5514 | 0.2585 |
| Transformer-Encoder Hybrid | 0.3740 | 0.3397 | 0.9688 | 0.5030 | 0.3588 | 0.5532 | 0.3356 |
| Cross-Attention Fusion Hybrid | 0.3804 | 0.3404 | 0.9541 | 0.5018 | 0.3540 | 0.5550 | 0.3680 |
| Logistic Regression | 0.3842 | 0.3406 | 0.9437 | 0.5006 | 0.3506 | 0.5463 | 0.2976 |
| Vanilla LSTM | 0.3640 | 0.3347 | 0.9566 | 0.4959 | 0.3650 | 0.5572 | 0.3773 |

All metrics computed on pooled out-of-fold predictions (every positive borrower evaluated exactly once across folds), not a single held-out split.

Note: Accuracy is included for completeness but is not a reliable indicator of model quality here, given the ~2.3% positive rate — a model that predicts "no default" for every borrower would score ~97.7% accuracy while catching zero actual defaulters. Precision, recall, F1, average precision, and ROC-AUC are more informative for this task.