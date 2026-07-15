# Lift Over Random Baseline

| Model | Base Rate | Avg. Precision | Lift over Random |
| --- | --- | --- | --- |
| Vanilla LSTM | 0.3270 | 0.3650 | 1.1163 |
| Transformer-Encoder Hybrid | 0.3270 | 0.3588 | 1.0973 |
| XGBoost | 0.3270 | 0.3544 | 1.0837 |
| Cross-Attention Fusion Hybrid | 0.3270 | 0.3540 | 1.0827 |
| Hybrid LSTM-GRU-Attention-FNN | 0.3270 | 0.3535 | 1.0811 |
| Logistic Regression | 0.3270 | 0.3506 | 1.0722 |

Lift = Average Precision / Base Rate. A model with zero skill (random guessing) has average precision equal to the base rate, i.e. lift = 1.0. Lift is a more honest measure of model quality than raw average precision under severe class imbalance, since raw AP looks small in absolute terms even for a genuinely strong model when the base rate itself is very low.