# Customer Churn Prediction & Revenue Impact Optimization

End-to-end data science project that predicts likely churners and quantifies financial impact of targeted retention.

## Business Goal

Predict customers likely to churn and simulate a retention strategy to minimize revenue loss.

## Professional Project Structure

```text
ChurnPredictionProject/
├── app.py
├── requirements.txt
├── pyproject.toml
├── README.md
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── models/
├── notebooks/
│   ├── 01_eda_and_modeling_template.ipynb
│   └── 02_churn_end_to_end_starter.ipynb
├── reports/
│   ├── RESULTS_REPORT_TEMPLATE.md
│   └── figures/
├── scripts/
│   ├── train_model.py
│   └── run_business_simulation.py
├── src/
│   └── churn_prediction/
│       ├── __init__.py
│       ├── business.py
│       ├── config.py
│       ├── data.py
│       ├── evaluation.py
│       ├── features.py
│       ├── modeling.py
│       └── pipeline.py
└── tests/
    └── test_smoke.py
```

## Why This Structure Works

- `src/churn_prediction/` keeps reusable, production-style Python modules.
- `scripts/` contains executable entry points for training and simulation.
- `notebooks/` supports EDA and storytelling without mixing with core logic.
- `models/` and `reports/` separate artifacts from source code.
- `data/raw` and `data/processed` enforce clean data lifecycle boundaries.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train the Pipeline

```bash
python scripts/train_model.py --data customer_churn.csv
```

If `--data` is relative, it is resolved from `data/raw/`.

## Run Business Impact Simulation

```bash
python scripts/run_business_simulation.py --data customer_churn.csv
```

## Streamlit Placeholder (Phase 8)

```bash
streamlit run app.py
```

## Portfolio Assets

- Starter notebook: `notebooks/02_churn_end_to_end_starter.ipynb`
- Recruiter-ready report template: `reports/RESULTS_REPORT_TEMPLATE.md`

## Phase Coverage

- Problem framing and business objective
- EDA and exploratory insights
- Preprocessing pipeline for numerical/categorical features
- Baseline model benchmarking (Logistic Regression, Random Forest, optional XGBoost)
- Threshold optimization for business trade-off
- Business impact simulation (cost, revenue saved, net benefit)
- Optional deployment scaffold

## Suggested Next Actions

1. Place your dataset in `data/raw/`.
2. Ensure target column is named `churn` (or update `ModelConfig` in `src/churn_prediction/config.py`).
3. Execute `scripts/train_model.py` and capture results for your portfolio report.
4. Convert notebook template into `.ipynb` and add your visuals/insights.
