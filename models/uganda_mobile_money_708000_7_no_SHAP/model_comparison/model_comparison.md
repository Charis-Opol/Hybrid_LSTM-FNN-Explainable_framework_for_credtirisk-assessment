# Model Comparison — Pooled Out-of-Fold Metrics (5-Fold CV)

| Model                         |   Precision |   Recall |     F1 |   Avg. Precision |   ROC-AUC |   Threshold |
|:------------------------------|------------:|---------:|-------:|-----------------:|----------:|------------:|
| XGBoost                       |      0.1331 |   0.5000 | 0.2102 |           0.1221 |    0.8866 |      0.0688 |
| Hybrid LSTM-GRU-Attention-FNN |      0.1153 |   0.4857 | 0.1863 |           0.1143 |    0.8672 |      0.6477 |
| Vanilla LSTM                  |      0.1287 |   0.3143 | 0.1826 |           0.1002 |    0.8534 |      0.6501 |
| Logistic Regression           |      0.1169 |   0.3857 | 0.1794 |           0.1034 |    0.8592 |      0.6545 |

All metrics computed on pooled out-of-fold predictions (every positive borrower evaluated exactly once across folds), not a single held-out split.