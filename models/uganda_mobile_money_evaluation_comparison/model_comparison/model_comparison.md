# Model Comparison — Pooled Out-of-Fold Metrics (5-Fold CV)

| Model | Accuracy | Precision | Recall | F1 | Avg. Precision | ROC-AUC | Threshold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0.9123 | 0.1331 | 0.5000 | 0.2102 | 0.1221 | 0.8866 | 0.0688 |
| Transformer-Encoder Hybrid | 0.9200 | 0.1205 | 0.3857 | 0.1837 | 0.1029 | 0.8746 | 0.7889 |
| Logistic Regression | 0.9177 | 0.1169 | 0.3857 | 0.1794 | 0.1034 | 0.8592 | 0.6545 |
| Hybrid LSTM-GRU-Attention-FNN | 0.9233 | 0.1154 | 0.3429 | 0.1727 | 0.1015 | 0.8626 | 0.6847 |
| Cross-Attention Fusion Hybrid | 0.8690 | 0.0932 | 0.5286 | 0.1585 | 0.0830 | 0.8580 | 0.6647 |

All metrics computed on pooled out-of-fold predictions (every positive borrower evaluated exactly once across folds), not a single held-out split.

Note: Accuracy is included for completeness but is not a reliable indicator of model quality here, given the ~2.3% positive rate — a model that predicts "no default" for every borrower would score ~97.7% accuracy while catching zero actual defaulters. Precision, recall, F1, average precision, and ROC-AUC are more informative for this task.