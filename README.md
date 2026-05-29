# Loan Default Prediction & Borrower Segmentation

## Project Overview

This project simulates a banking-style credit risk workflow applied to a loan dataset. The goal is to identify borrowers at risk of default and uncover hidden borrower profiles using both predictive modelling and clustering techniques.

**Dataset:** `Loan_Default.csv`
- ~148,000+ rows, 34 columns
- Target variable: `Status` — `0` = non-default, `1` = default
- Class distribution: ~75% non-default, ~25% default (moderate imbalance)

---

## Workflow

1. Data Cleaning & Preprocessing
2. Exploratory Data Analysis (EDA)
3. Supervised Learning — Logistic Regression, Random Forest, XGBoost
4. Unsupervised Learning — K-Means Clustering
5. Project Summary & Interpretation

---

## Data Preprocessing

- Removed ID column
- Numerical missing values: **median imputation**
- Categorical missing values: **mode imputation**
- One-hot encoding via `pd.get_dummies(drop_first=True)`
- Train/test split: `test_size=0.2`, `random_state=42`
- `StandardScaler` applied for Logistic Regression only (tree-based models were not scaled)

**Key features used:**

| Type | Variables |
|------|-----------|
| Numerical | `income`, `loan_amount`, `Credit_Score`, `rate_of_interest`, `LTV`, `dtir1`, `property_value`, `term`, `Interest_rate_spread`, `Upfront_charges` |
| Categorical | `Gender`, `loan_type`, `loan_purpose`, `occupancy_type`, `approv_in_adv`, `business_or_commercial`, `interest_only`, `Neg_ammortization`, `Region`, `credit_type` |

---

## Exploratory Data Analysis

- Target variable distribution (countplot)
- Categorical variables vs. default status
- Numerical distributions and boxplots
- Correlation heatmap

**Key EDA findings:** Loan structure and financing-related variables (interest spread, LTV, debt-to-income ratio) showed stronger links to default risk than income alone.

---

## Supervised Learning Results

### Logistic Regression

| Metric | Score |
|--------|-------|
| Accuracy | ~0.87 |
| ROC-AUC | ~0.86 |

**Confusion Matrix:**
```
[[22197   297]
 [ 3528  3712]]
```

**Interpretation:** Strong, interpretable baseline. Suitable for banking contexts due to explainability.

**Key findings:**
- Higher LTV → higher default risk
- Higher `dtir1` → higher default risk
- `credit_type_EQUI` → strong positive association with default
- Loan amount showed a positive relationship with default risk

---

### Random Forest

| Metric | Score |
|--------|-------|
| ROC-AUC | ~1.00 |

**Top features by importance (from feature importance analysis):**
- `Interest_rate_spread`, `Upfront_charges`, `rate_of_interest`, `credit_type_EQUI` — highest importance
- `property_value`, `LTV`, `dtir1` — moderately important

**Interpretation:** Loan pricing, financing conditions, leverage, and debt burden were highly predictive of default.

> Near-perfect ROC-AUC may indicate possible information leakage — pricing variables such as `Interest_rate_spread` may already encode lender risk assessments.

---

### XGBoost

| Metric | Score |
|--------|-------|
| ROC-AUC | ~1.00 |

**Top features by importance:**
1. `Interest_rate_spread`
2. `credit_type_EQUI`
3. `Upfront_charges`

**Interpretation:** Confirmed that loan pricing and credit-type variables are among the strongest signals for default risk.

> Same leakage caveat applies as with Random Forest.

---

### Overall Supervised Learning Takeaway

> Borrowers with higher financing costs, higher leverage, heavier debt burden, and weaker credit-type indicators were more likely to be classified as default-risk borrowers. Loan structure and financing conditions appeared more predictive of default than income alone.

---

## Unsupervised Learning — K-Means Clustering

**Purpose:** Segment borrowers into hidden financial profiles beyond the binary default label.

**Variables used:**
`income`, `loan_amount`, `Credit_Score`, `LTV`, `dtir1`, `property_value`, `Interest_rate_spread`, `Upfront_charges`

**Preprocessing:** Median imputation, `inf`/`-inf` handling, `StandardScaler`

**Number of clusters:** 3 (selected via Elbow Method)

| Cluster | Size | Profile |
|---------|------|---------|
| Cluster 0 | ~55,916 | **Strong Credit Borrowers** — avg. Credit Score ~803, moderate income and leverage |
| Cluster 1 | ~56,349 | **Higher Risk Borrowers** — avg. Credit Score ~597, similar income to Cluster 0 but higher leverage |
| Cluster 2 | ~36,405 | **Premium / High Capacity Borrowers** — highest income, loan amount, and property value; lowest LTV and dtir1 |

**Key insight:** Borrowers with similar income levels fell into different risk groups due to differences in credit score, leverage, and debt burden — income alone does not explain default risk.

> Silhouette Score ≈ 0.136 — clusters are not strongly separated. K-Means results should be treated as **exploratory borrower profiling**, not a final credit decision model.

---

## Final Conclusion

This project demonstrates that loan default risk is more strongly linked to **financing pressure and credit quality** than income alone. Variables such as `Interest_rate_spread`, `Upfront_charges`, `rate_of_interest`, `LTV`, `dtir1`, and `credit_type_EQUI` emerged as important signals across the supervised analysis, especially in the tree-based models.

The K-Means clustering further supported this by showing that borrowers with comparable incomes could belong to very different risk profiles depending on their credit score, leverage, and debt burden.

Taken together, this project illustrates a simplified but grounded banking credit risk analytics workflow — combining **default prediction**, **model interpretation**, and **borrower segmentation**.

---

## Tech Stack

- Python, Pandas, NumPy
- Scikit-learn (Logistic Regression, Random Forest, K-Means, StandardScaler, train_test_split)
- XGBoost
- Matplotlib, Seaborn
- Jupyter Notebook

---

## Disclaimer

This project is for educational and portfolio purposes only. It does not constitute financial or lending advice.
