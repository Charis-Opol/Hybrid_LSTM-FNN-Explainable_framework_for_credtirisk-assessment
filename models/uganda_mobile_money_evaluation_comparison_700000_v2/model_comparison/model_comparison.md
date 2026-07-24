# Model Comparison — Pooled Out-of-Fold Metrics (5-Fold CV)

| Model | Accuracy | Precision | Recall | F1 | Avg. Precision | ROC-AUC | Threshold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0.6008 | 0.1494 | 0.6564 | 0.2434 | 0.1515 | 0.6655 | 0.2402 |
| Logistic Regression | 0.5216 | 0.1411 | 0.7648 | 0.2382 | 0.1424 | 0.6579 | 0.4144 |
| Transformer-Encoder Hybrid | 0.6024 | 0.1426 | 0.6115 | 0.2312 | 0.1428 | 0.6508 | 0.5285 |
| Hybrid LSTM-GRU-Attention-FNN | 0.5260 | 0.1362 | 0.7198 | 0.2290 | 0.1406 | 0.6450 | 0.5304 |
| Cross-Attention Fusion Hybrid | 0.4394 | 0.1322 | 0.8507 | 0.2289 | 0.1400 | 0.6449 | 0.4989 |
| Vanilla LSTM | 0.5510 | 0.1311 | 0.6380 | 0.2175 | 0.1261 | 0.5950 | 0.4499 |

All metrics computed on pooled out-of-fold predictions (every positive borrower evaluated exactly once across folds), not a single held-out split.

Note: Accuracy is included for completeness but is not a reliable indicator of model quality here, given the ~2.3% positive rate — a model that predicts "no default" for every borrower would score ~97.7% accuracy while catching zero actual defaulters. Precision, recall, F1, average precision, and ROC-AUC are more informative for this task.