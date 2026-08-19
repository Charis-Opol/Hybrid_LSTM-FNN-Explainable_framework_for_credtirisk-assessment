# Mobile Money Credit Risk Application

This repository now includes a Flutter client in `flutter_app` and a FastAPI
backend in `app_backend`. The backend connects to the trained five-fold hybrid
model ensemble at `models/uganda_mobile_money_evaluation_comparison/kfold_hybrid`.

1. Start the backend using the commands in `app_backend/README.md`.
2. Install Flutter and run the client using `flutter_app/README.md`.
3. Upload a single borrower’s CSV covering at least 12 calendar months.

Use the API endpoint `GET /template` for the CSV header and example rows.

Do not treat a predicted score as an automated loan decision. The model was
evaluated on a research dataset with a small default class and needs governance,
fairness testing, monitoring, security controls, and independent validation before
production lending use.
