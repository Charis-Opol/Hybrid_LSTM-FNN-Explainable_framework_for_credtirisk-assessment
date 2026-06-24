# Credit Risk Framework

Research-grade Python framework for the undergraduate dissertation:

**A Hybrid LSTM-FNN Framework with Localised Feature Engineering and Explainable AI for Credit Risk Assessment in Ugandan Microfinance Institutions.**

## Project Status

The framework is implemented incrementally through the core research workflow:

- Raw-data preprocessing.
- Localised Ugandan feature engineering.
- Temporal and static dataset construction.
- LSTM and feed-forward encoders.
- Hybrid LSTM-FNN model compilation.
- Training, evaluation, baseline comparison, and SHAP explainability utilities.

## Requirements

- Python 3.11
- pandas
- numpy
- scikit-learn
- tensorflow
- shap

## Structure

```text
credit-risk-framework/
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- external/
|-- models/
|-- notebooks/
|-- tests/
|-- preprocessing.py
|-- feature_engineering.py
|-- temporal_dataset.py
|-- static_dataset.py
|-- lstm_encoder.py
|-- fnn_encoder.py
|-- hybrid_model.py
|-- explainability.py
|-- train.py
|-- evaluate.py
|-- config.py
|-- utils.py
|-- requirements.txt
`-- README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

The intended workflow is:

1. Place raw microfinance transaction data in `data/raw/`.
2. Run preprocessing to produce cleaned datasets in `data/processed/`.
3. Engineer localised behavioural and financial features.
4. Build temporal and static model inputs.
5. Train and evaluate the hybrid LSTM-FNN model.
6. Generate SHAP explanations for model transparency.

```python
from preprocessing import PreprocessingPipeline
from feature_engineering import engineer_features
from temporal_dataset import TemporalDatasetBuilder
from static_dataset import StaticDatasetBuilder
from train import train_model

pipeline = PreprocessingPipeline(
    input_path="data/raw/transactions.csv",
    impossible_value_rules={"loan_amount": {"min": 0}},
)
raw = pipeline.load_data()
cleaned = pipeline.clean(raw)
features = engineer_features(cleaned)

temporal = TemporalDatasetBuilder().build(features)
static = StaticDatasetBuilder().build(features)

artifacts = train_model(
    temporal.X_temporal,
    static.X_static,
    temporal.labels,
)
```

## Expected Core Columns

The default pipeline expects:

- `borrower_id`
- `transaction_date`
- `transaction_amount`
- `transaction_type`
- `balance`
- `defaulted`

Optional static columns such as `income_source`, `loan_amount`, `sacco_membership`, `location`, `preferred_network`, and `preferred_channel` are used when present.

## Verification

Run tests after installing dependencies:

```bash
pip install -r requirements.txt
pytest tests
```
