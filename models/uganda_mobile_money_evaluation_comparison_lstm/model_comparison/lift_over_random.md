# Lift Over Random Baseline

| Model | Base Rate | Avg. Precision | Lift over Random |
| --- | --- | --- | --- |
| XGBoost | 0.0233 | 0.1221 | 5.2327 |
| Hybrid LSTM-Attention-FNN | 0.0233 | 0.1035 | 4.4345 |
| Logistic Regression | 0.0233 | 0.1034 | 4.4313 |
| Cross-Attention Fusion Hybrid | 0.0233 | 0.1009 | 4.3259 |
| Transformer-Encoder Hybrid | 0.0233 | 0.0933 | 3.9982 |

Lift = Average Precision / Base Rate. A model with zero skill (random guessing) has average precision equal to the base rate, i.e. lift = 1.0. Lift is a more honest measure of model quality than raw average precision under severe class imbalance, since raw AP looks small in absolute terms even for a genuinely strong model when the base rate itself is very low.