# Mobile-money risk API

Run this API from the repository root with the project environment activated:

```powershell
.venv\Scripts\python.exe -m pip install -r app_backend\requirements.txt
.venv\Scripts\python.exe -m uvicorn app_backend.main:app --reload --port 8000
```

The Flutter app uploads a CSV with `borrower_id`, `transaction_date`,
`transaction_amount`, and optionally `transaction_type` and `balance`. The CSV
must contain at least twelve calendar months for the specified borrower.

## Dataset format

Use one transaction per row, saved as UTF-8 CSV. Use these exact headers:

```csv
borrower_id,transaction_date,transaction_amount,transaction_type,balance
UG-001,2025-01-15,125000,received,125000
UG-001,2025-02-15,126000,received,126000
```

`borrower_id` is the value entered in the app. `transaction_date` must be a
parseable date in `YYYY-MM-DD` format. `transaction_amount` must be numeric and
at least one amount must be non-zero. Use positive amounts; `transaction_type`
determines direction. Accepted incoming values include `received`, `deposit`,
`credit`, `inflow`, and `income`; outgoing values include `sent`, `withdrawal`,
`debit`, `payment`, `outflow`, and `paid`. `balance` is numeric and may be `0`
when unavailable. Include transactions spanning twelve calendar months, not
merely twelve rows.

Download a correctly formatted twelve-month template from `GET /template`.

The assessment response returns the model score immediately. Use the app's
`Explain with SHAP` button, or `POST /explain` with the same CSV and borrower
fields, to calculate local SHAP drivers for that borrower.

The returned `default_probability` combines the raw hybrid-model probability
with a transparent affordability adjustment based on requested loan amount
divided by average monthly inflow. The response also includes
`model_probability`, `affordability_ratio`, and `affordability_adjustment` so
the two components remain visible.

The API uses the five saved `kfold_hybrid` checkpoints in
`models/uganda_mobile_money_hybrid_vs_transformer` and averages their scores.
Set `RISK_MODEL_DIR` to another model-directory name under `models/` when a
different hybrid experiment should be used.
The original experiment did not persist fitted preprocessing transformers, so
numeric static inputs are calibrated against the included `engineered_features.csv`.
This is an inference reconstruction, not a replacement for retraining and saving a
single production model plus its exact preprocessing pipeline.

The desktop Flutter app uses `http://127.0.0.1:8000`. For an Android emulator,
use `http://10.0.2.2:8000` in the app's API address field.
