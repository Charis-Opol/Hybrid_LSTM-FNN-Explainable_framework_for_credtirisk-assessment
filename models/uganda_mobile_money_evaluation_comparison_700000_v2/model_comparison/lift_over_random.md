# Lift Over Random Baseline

| Model | Base Rate | Avg. Precision | Lift over Random |
| --- | --- | --- | --- |
| XGBoost | 0.0978 | 0.1515 | 1.5495 |
| Transformer-Encoder Hybrid | 0.0978 | 0.1428 | 1.4599 |
| Logistic Regression | 0.0978 | 0.1424 | 1.4565 |
| Hybrid LSTM-GRU-Attention-FNN | 0.0978 | 0.1406 | 1.4376 |
| Cross-Attention Fusion Hybrid | 0.0978 | 0.1400 | 1.4320 |
| Vanilla LSTM | 0.0978 | 0.1261 | 1.2898 |

Lift = Average Precision / Base Rate. A model with zero skill (random guessing) has average precision equal to the base rate, i.e. lift = 1.0. Lift is a more honest measure of model quality than raw average precision under severe class imbalance, since raw AP looks small in absolute terms even for a genuinely strong model when the base rate itself is very low.