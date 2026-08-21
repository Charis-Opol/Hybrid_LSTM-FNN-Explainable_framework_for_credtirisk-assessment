# Model Comparison — Pooled Out-of-Fold Metrics (5-Fold CV)

| Model | Accuracy | Precision | Recall | F1 | Avg. Precision | ROC-AUC | Threshold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0.9123 | 0.1331 | 0.5000 | 0.2102 | 0.1221 | 0.8866 | 0.0688 |
| Cross-Attention Fusion Hybrid | 0.9497 | 0.1653 | 0.2857 | 0.2094 | 0.1009 | 0.8204 | 0.8514 |
| Hybrid LSTM-Attention-FNN | 0.8667 | 0.1145 | 0.7000 | 0.1968 | 0.1035 | 0.8816 | 0.6285 |
| Transformer-Encoder Hybrid | 0.8983 | 0.1173 | 0.5143 | 0.1910 | 0.0933 | 0.8590 | 0.6357 |
| Logistic Regression | 0.9177 | 0.1169 | 0.3857 | 0.1794 | 0.1034 | 0.8592 | 0.6545 |

All metrics computed on pooled out-of-fold predictions (every positive borrower evaluated exactly once across folds), not a single held-out split.

Note: Accuracy is included for completeness but is not a reliable indicator of model quality here, given the ~2.3% positive rate — a model that predicts "no default" for every borrower would score ~97.7% accuracy while catching zero actual defaulters. Precision, recall, F1, average precision, and ROC-AUC are more informative for this task.