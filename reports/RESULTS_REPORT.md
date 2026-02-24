# Churn Prediction Project — Results Report

## Executive Summary
- **Business Problem:** Predict customers likely to churn and optimize targeted retention actions.
- **Model Outcome:** Best model achieved **ROC-AUC = 0.842** on holdout data.
- **Business Outcome:** Estimated **net benefit = $136,350** for the evaluated holdout cohort (**1,409 customers**) under targeted retention policy.
- **Recommendation:** Target top **20%** high-risk customers using threshold **0.10** for high-recall capture.

## 1. Problem Framing
- Churn definition: customer cancellation/churn status in the observed telco period.
- Target variable: **`Churn`** (Yes = churn, No = retained; normalized to 1/0 for modeling).
- Cost assumptions:
  - Retention campaign cost/customer: **$50**
  - Average CLV: **$800**

## 2. Data Summary
- Dataset name/source: **Telco Customer Churn dataset**
- Number of rows: **7,043**
- Number of features: **20** (excluding target)
- Churn rate: **26.54%**
- Missing data approach: **Median imputation (numeric) and most-frequent imputation (categorical) in pipeline**

## 3. Modeling Approach
- Preprocessing:
  - Numeric: median imputation + scaling
  - Categorical: most-frequent imputation + one-hot encoding
- Models benchmarked:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Validation strategy: **Stratified train/validation split (80/20), random state = 42**

## 4. Performance Results
| Model | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.840 | 0.536 | 0.719 | 0.614 |
| Random Forest | 0.826 | 0.622 | 0.545 | 0.581 |
| XGBoost | 0.842 | 0.655 | 0.519 | 0.579 |

### Threshold Optimization
- Selected threshold: **0.10**
- Reason: maximize recall while keeping precision above **0.30** (business rule in pipeline).
- Tuned metrics at threshold **0.10**:
  - Precision: **0.418**
  - Recall: **0.939**
  - F1: **0.578**

## 5. Interpretability Insights
### Top Churn Drivers
1. **Month-to-month contracts** (highest churn rate: ~42.7%)
2. **Short tenure (0–12 months)** (highest tenure-group churn: ~47.4%)
3. **Electronic check payment method** (highest payment-method churn: ~45.3%)
4. **Higher monthly charges** (positive churn correlation; corr ≈ 0.193)
5. **Lower service stickiness** (customers without long-term contract commitment)

### Top Retention Indicators
1. **Two-year contracts** (lowest churn rate: ~2.83%)
2. **Long tenure (49+ months)** (lowest tenure-group churn: ~9.51%)
3. **One-year contracts** (churn rate: ~11.3%)
4. **Automatic payment methods** (credit card/bank transfer show lower churn)
5. **Lower monthly charges** (associated with improved retention)

## 6. Business Impact Simulation
Assumptions:
- Retention cost/customer = **$50**
- Average CLV = **$800**
- Targeted high-risk population = **top 20%**

Formulas:
- Revenue Saved = Correctly Predicted Churners × CLV
- Net Benefit = Revenue Saved − (Retention Cost × Number Targeted)

| Metric | Value |
|---|---:|
| Number Targeted | 281 |
| Correctly Predicted Churners | 188 |
| Retention Cost Total | $14,050 |
| Revenue Saved | $150,400 |
| Net Benefit | $136,350 |

