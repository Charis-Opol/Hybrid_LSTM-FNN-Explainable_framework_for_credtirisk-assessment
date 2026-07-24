# Model Comparison — Pooled Out-of-Fold Metrics (5-Fold CV)

| Model | Accuracy | Precision | Recall | F1 | Avg. Precision | ROC-AUC | Threshold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost (no_weight) | 0.9613 | 0.2262 | 0.2714 | 0.2468 | 0.1527 | 0.8870 | 0.1235 |
| XGBoost (smote_0.5) | 0.9340 | 0.1559 | 0.4143 | 0.2266 | 0.1631 | 0.8891 | 0.1395 |
| XGBoost (baseline) | 0.9123 | 0.1331 | 0.5000 | 0.2102 | 0.1221 | 0.8866 | 0.0688 |
| Vanilla LSTM (weight_1.5) | 0.9390 | 0.1401 | 0.3143 | 0.1938 | 0.0978 | 0.8358 | 0.6835 |
| Logistic Regression (no_weight) | 0.9263 | 0.1281 | 0.3714 | 0.1905 | 0.1112 | 0.8684 | 0.1098 |
| Vanilla LSTM (baseline) | 0.9343 | 0.1287 | 0.3143 | 0.1826 | 0.1002 | 0.8534 | 0.6501 |
| Logistic Regression (baseline) | 0.9177 | 0.1169 | 0.3857 | 0.1794 | 0.1034 | 0.8592 | 0.6545 |
| Hybrid (weight_1.5) | 0.9050 | 0.1119 | 0.4429 | 0.1787 | 0.1019 | 0.8670 | 0.7739 |
| Hybrid (baseline) | 0.9233 | 0.1154 | 0.3429 | 0.1727 | 0.1015 | 0.8626 | 0.6847 |

All metrics computed on pooled out-of-fold predictions (every positive borrower evaluated exactly once across folds), not a single held-out split.

Note: Accuracy is included for completeness but is not a reliable indicator of model quality here, given the ~2.3% positive rate — a model that predicts "no default" for every borrower would score ~97.7% accuracy while catching zero actual defaulters. Precision, recall, F1, average precision, and ROC-AUC are more informative for this task.