# Lift Over Random Baseline

| Model | Base Rate | Avg. Precision | Lift over Random |
| --- | --- | --- | --- |
| XGBoost | 0.0233 | 0.1221 | 5.2327 |
| Logistic Regression | 0.0233 | 0.1034 | 4.4313 |
| Transformer-Encoder Hybrid | 0.0233 | 0.1029 | 4.4103 |
| Hybrid LSTM-GRU-Attention-FNN | 0.0233 | 0.1015 | 4.3485 |
| Cross-Attention Fusion Hybrid | 0.0233 | 0.0830 | 3.5565 |

Lift = Average Precision / Base Rate. A model with zero skill (random guessing) has average precision equal to the base rate, i.e. lift = 1.0. Lift is a more honest measure of model quality than raw average precision under severe class imbalance, since raw AP looks small in absolute terms even for a genuinely strong model when the base rate itself is very low.