# Lift Over Random Baseline

| Model | Base Rate | Avg. Precision | Lift over Random |
| --- | --- | --- | --- |
| XGBoost (smote_0.5) | 0.0233 | 0.1631 | 6.9885 |
| XGBoost (no_weight) | 0.0233 | 0.1527 | 6.5434 |
| XGBoost (baseline) | 0.0233 | 0.1221 | 5.2327 |
| Logistic Regression (no_weight) | 0.0233 | 0.1112 | 4.7644 |
| Logistic Regression (baseline) | 0.0233 | 0.1034 | 4.4313 |
| Hybrid (weight_1.5) | 0.0233 | 0.1019 | 4.3668 |
| Hybrid (baseline) | 0.0233 | 0.1015 | 4.3485 |
| Vanilla LSTM (baseline) | 0.0233 | 0.1002 | 4.2959 |
| Vanilla LSTM (weight_1.5) | 0.0233 | 0.0978 | 4.1908 |

Lift = Average Precision / Base Rate. A model with zero skill (random guessing) has average precision equal to the base rate, i.e. lift = 1.0. Lift is a more honest measure of model quality than raw average precision under severe class imbalance, since raw AP looks small in absolute terms even for a genuinely strong model when the base rate itself is very low.