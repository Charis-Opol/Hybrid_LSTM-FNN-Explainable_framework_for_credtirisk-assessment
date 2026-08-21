# Stream Ablation -- Temporal (GRU/LSTM) vs. Static (FNN) Contribution

Classification head (Dense64 -> BN -> Dropout -> Dense32 -> BN -> Dropout -> sigmoid) held identical across all rows; only the input stream(s) change.

| Configuration | accuracy | precision | recall | f1 | average_precision | roc_auc | selected_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Static-only (FNN) | 0.9387 | 0.1437 | 0.3286 | 0.2000 | 0.1161 | 0.8672 | 0.7340 |
| Temporal-only (GRU) | 0.8970 | 0.1056 | 0.4571 | 0.1716 | 0.0838 | 0.8543 | 0.7527 |
| Temporal-only (LSTM) | 0.8843 | 0.0866 | 0.4143 | 0.1432 | 0.0771 | 0.8275 | 0.6683 |
| Full Hybrid (GRU + static) | 0.9010 | 0.1153 | 0.4857 | 0.1863 | 0.1143 | 0.8672 | 0.6477 |
| Full Hybrid (LSTM + static) | 0.8667 | 0.1145 | 0.7000 | 0.1968 | 0.1035 | 0.8816 | 0.6285 |