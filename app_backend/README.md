# Mobile-money risk API

Run this API from the repository root with the project environment activated:

```powershell
.venv\Scripts\python.exe -m pip install -r app_backend\requirements.txt
.venv\Scripts\python.exe -m uvicorn app_backend.main:app --reload --port 8000
```

The Flutter app uploads a CSV with `borrower_id`, `transaction_date`,
`transaction_amount`, and optionally `transaction_type` and `balance`. The CSV
must contain at least twelve calendar months for the specified borrower.

The API uses the five saved `kfold_hybrid` checkpoints in
`models/uganda_mobile_money_evaluation_comparison` and averages their scores.
The original experiment did not persist fitted preprocessing transformers, so
numeric static inputs are calibrated against the included `engineered_features.csv`.
This is an inference reconstruction, not a replacement for retraining and saving a
single production model plus its exact preprocessing pipeline.
