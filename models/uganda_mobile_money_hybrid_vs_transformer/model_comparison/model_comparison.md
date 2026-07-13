# Model Comparison — Pooled Out-of-Fold Metrics (5-Fold CV)

| Model | Accuracy | Precision | Recall | F1 | Avg. Precision | ROC-AUC | Threshold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Transformer-Encoder Hybrid | 0.8983 | 0.1173 | 0.5143 | 0.1910 | 0.0933 | 0.8590 | 0.6357 |
| Hybrid LSTM-GRU-Attention-FNN | 0.9010 | 0.1153 | 0.4857 | 0.1863 | 0.1143 | 0.8672 | 0.6477 |

All metrics computed on pooled out-of-fold predictions (every positive borrower evaluated exactly once across folds), not a single held-out split.

Note: Accuracy is included for completeness but is not a reliable indicator of model quality here, given the ~2.3% positive rate — a model that predicts "no default" for every borrower would score ~97.7% accuracy while catching zero actual defaulters. Precision, recall, F1, average precision, and ROC-AUC are more informative for this task.