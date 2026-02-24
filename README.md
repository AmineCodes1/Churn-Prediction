# Customer Churn Prediction & Revenue Impact Optimization

End-to-end machine learning project that predicts customer churn and links model decisions to expected financial impact.

## Business Objective

Identify high-risk customers and simulate a targeted retention campaign that maximizes net benefit.

## Current Results (Telco dataset)

- Best model: **XGBoost**
- Holdout ROC-AUC: **0.842**
- 5-fold CV ROC-AUC: **0.843 ± 0.011**
- Tuned threshold (recall-focused): **0.10**
- Estimated campaign net benefit: **$136,350**

Detailed outputs are documented in `reports/RESULTS_REPORT.md`.

## Repository Structure

```text
ChurnPredictionProject/
├── app.py
├── pyproject.toml
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── models/
├── notebooks/
│   └── 01_churn_end_to_end_starter.ipynb
├── reports/
│   ├── RESULTS_REPORT.md
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
│       ├── explainability.py
│       ├── features.py
│       ├── modeling.py
│       └── pipeline.py
└── tests/
        └── test_smoke.py
```

## What Is Implemented

- Reusable training pipeline with preprocessing + model selection.
- Model benchmarking (Logistic Regression, Random Forest, XGBoost when available).
- Threshold optimization for business trade-offs (precision-constrained recall).
- Business impact simulation (cost, saved revenue, net benefit).
- Robust evaluation artifacts:
    - Stratified 5-fold CV summary (mean/std).
    - Probability quality metrics (PR-AUC, Brier score, log loss).
    - Calibration table (binned predicted vs observed risk).
- Explainability artifacts:
    - SHAP summary plot (generated in `reports/figures/`).
    - Top SHAP features with actionable direction insights.
- Streamlit app for single-customer and batch scoring.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train and Evaluate

```bash
python scripts/train_model.py --data customer_churn.csv
```

If `--data` is relative, it resolves from `data/raw/`.

## Run Business Simulation Only

```bash
python scripts/run_business_simulation.py --data customer_churn.csv
```

## Run Streamlit App

```bash
python -m streamlit run app.py
```

## Notebook Walkthrough

Open `notebooks/01_churn_end_to_end_starter.ipynb` to reproduce:

1. Data loading and quick EDA
2. Baseline training and benchmarking
3. Threshold + business impact summary
4. Robust diagnostics (CV + calibration)
5. Interview-ready interpretation
6. SHAP explainability artifact and actionable insights

## Notes

- SHAP summary images are generated at runtime into `reports/figures/`.
- `models/` and `reports/figures/` are artifact directories and may be gitignored except placeholders.
