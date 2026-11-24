# Churn Prediction Project — Results Report

## Executive Summary
- **Business Problem:** Predict customers likely to churn and optimize targeted retention actions.
- **Model Outcome:** Best model achieved **ROC-AUC = 0.842** on holdout data.
- **Business Outcome:** Estimated **net benefit = $137,150** for the evaluated holdout cohort (**1,409 customers**) under targeted retention policy.
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
- Missing data approach: Numeric-looking text columns are converted to numeric during loading; the 11 blank `TotalCharges` values are median-imputed in the numeric preprocessing pipeline. Categorical missing values use most-frequent imputation.

## 3. Modeling Approach
- Preprocessing:
  - Numeric: median imputation + scaling
  - Categorical: most-frequent imputation + one-hot encoding
- Models benchmarked:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Validation strategy: **Stratified train/validation split (80/20), random state = 42, plus stratified 5-fold CV on training split**

## 3.1 Robustness Checks
- Cross-validation stability (best model):
  - CV ROC-AUC mean ± std: **0.844 ± 0.009**
  - CV PR-AUC mean ± std: **0.664 ± 0.018**
  - CV Precision mean ± std: **0.663 ± 0.038**
  - CV Recall mean ± std: **0.530 ± 0.032**
  - CV F1 mean ± std: **0.589 ± 0.032**
- Probability quality on holdout:
  - PR-AUC: **0.657**
  - Brier score: **0.137**
  - Log loss: **0.421**
- Calibration diagnostics:
  - The model is reasonably calibrated in low-risk bins (0.0–0.1 predicted: 0.032 vs 0.043 observed) and high-risk bins (0.7–0.8 predicted: 0.739 vs 0.763 observed), with some underestimation around the 0.1–0.2 bin (0.144 predicted vs 0.185 observed).

## 4. Performance Results
| Model | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.841 | 0.524 | 0.754 | 0.618 |
| Random Forest | 0.830 | 0.546 | 0.652 | 0.594 |
| XGBoost | 0.842 | 0.662 | 0.508 | 0.575 |

### Threshold Optimization
- Selected threshold: **0.10**
- Reason: maximize recall while keeping precision above **0.30** (business rule in pipeline).
- Tuned metrics at threshold **0.10**:
  - Precision: **0.410**
  - Recall: **0.936**
  - F1: **0.570**

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
| Correctly Predicted Churners | 189 |
| Retention Cost Total | $14,050 |
| Revenue Saved | $151,200 |
| Net Benefit | $137,150 |

