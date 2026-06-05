# Loan Default Prediction & Borrower Segmentation

## Project Overview

This project simulates a banking-style credit risk workflow applied to a loan dataset. The goal is to identify borrowers at risk of default and uncover hidden borrower profiles using both predictive modelling and clustering techniques.

**Dataset:** `Loan_Default.csv`

- ~148,000+ rows, 34 columns
- Target variable: `Status`
  - `0` = non-default
  - `1` = default
- Class distribution: ~75% non-default, ~25% default (moderate imbalance)

---

## Workflow

1. Data Cleaning & Preprocessing
2. Exploratory Data Analysis
3. Supervised Learning
   - Logistic Regression
   - Random Forest
   - XGBoost
4. Unsupervised Learning
   - K-Means Clustering
5. Project Summary & Interpretation

---

## Data Preprocessing

The dataset was cleaned and prepared before modelling.

Steps completed:

- Removed ID column
- Imputed numerical missing values using the median
- Imputed categorical missing values using the mode
- Applied one-hot encoding using `pd.get_dummies(drop_first=True)`
- Used an 80/20 train-test split with `random_state=42`
- Applied `StandardScaler` for Logistic Regression only
- Tree-based models were not scaled because scaling is not required for Random Forest and XGBoost

### Key Features Used

| Type | Variables |
|---|---|
| Numerical | `income`, `loan_amount`, `Credit_Score`, `rate_of_interest`, `LTV`, `dtir1`, `property_value`, `term`, `Interest_rate_spread`, `Upfront_charges` |
| Categorical | `Gender`, `loan_type`, `loan_purpose`, `occupancy_type`, `approv_in_adv`, `business_or_commercial`, `interest_only`, `Neg_ammortization`, `Region`, `credit_type` |

---

## Exploratory Data Analysis

The exploratory data analysis focused on understanding the relationship between borrower and loan features with default status.

EDA steps included:

- Target variable distribution
- Categorical variables against default status
- Numerical feature distributions
- Boxplots for key numerical variables
- Correlation heatmap

### Key EDA Findings

Loan structure and financing-related variables showed stronger links to default risk than income alone.

Important signals included:

- Interest rate spread
- Loan-to-value ratio
- Debt-to-income ratio
- Credit type
- Upfront charges

---

## Supervised Learning Results

Three supervised learning models were trained and compared:

1. Logistic Regression
2. Random Forest
3. XGBoost

---

## Logistic Regression

| Metric | Score |
|---|---|
| Accuracy | ~0.87 |
| ROC-AUC | ~0.86 |

### Confusion Matrix

```text
[[22197   297]
 [ 3528  3712]]
```

### Interpretation

Logistic Regression provided a strong and interpretable baseline model. Its performance was realistic and useful for understanding general default risk patterns.

### Key Findings

- Higher `LTV` was linked to higher default risk
- Higher `dtir1` was linked to higher default risk
- `credit_type_EQUI` showed a strong positive association with default
- Higher loan amount showed a positive relationship with default risk

---

## Random Forest

| Metric | Score |
|---|---|
| ROC-AUC | ~1.00 |

### Top Features by Importance

The most important features identified by the Random Forest model included:

- `Interest_rate_spread`
- `Upfront_charges`
- `rate_of_interest`
- `credit_type_EQUI`

Moderately important features included:

- `property_value`
- `LTV`
- `dtir1`

### Interpretation

The Random Forest model showed that loan pricing, financing conditions, leverage, and debt burden were highly predictive of default.

However, the near-perfect ROC-AUC score may indicate possible data leakage, especially because pricing-related variables may already encode lender risk assessments.

---

## XGBoost

| Metric | Score |
|---|---|
| ROC-AUC | ~1.00 |

### Top Features by Importance

The most important features identified by the XGBoost model were:

1. `Interest_rate_spread`
2. `credit_type_EQUI`
3. `Upfront_charges`

### Interpretation

XGBoost confirmed that loan pricing and credit-type variables were among the strongest predictors of default risk.

The same data leakage concern applies here because the model relied heavily on variables that may already reflect lender risk assessment.

---

## Overall Supervised Learning Takeaway

Borrowers with higher financing costs, higher leverage, heavier debt burden, and weaker credit-type indicators were more likely to be classified as default-risk borrowers.

Loan structure and financing conditions appeared more predictive of default than income alone.

---

## Unsupervised Learning — K-Means Clustering

### Purpose

K-Means clustering was used to segment borrowers into hidden financial profiles beyond the binary default label.

### Variables Used

The clustering model used the following variables:

- `income`
- `loan_amount`
- `Credit_Score`
- `LTV`
- `dtir1`
- `property_value`
- `Interest_rate_spread`
- `Upfront_charges`

### Preprocessing

Before clustering, the following steps were applied:

- Median imputation
- Handling of `inf` and `-inf` values
- Feature scaling using `StandardScaler`

### Number of Clusters

The number of clusters was selected using the Elbow Method.

Final number of clusters used:

```text
k = 3
```

### Cluster Profiles

| Cluster | Size | Profile |
|---|---:|---|
| Cluster 0 | ~55,916 | **Strong Credit Borrowers** — average Credit Score around 803, with moderate income and leverage |
| Cluster 1 | ~56,349 | **Higher Risk Borrowers** — average Credit Score around 597, similar income to Cluster 0 but higher leverage |
| Cluster 2 | ~36,405 | **Premium / High Capacity Borrowers** — highest income, loan amount, and property value, with lowest LTV and `dtir1` |

### Key Insight

Borrowers with similar income levels could fall into different risk groups due to differences in credit score, leverage, and debt burden.

This supports the idea that income alone does not fully explain borrower risk.

### Clustering Limitation

The silhouette score was approximately:

```text
0.136
```

This suggests that the clusters were not strongly separated.

Therefore, the K-Means results should be treated as exploratory borrower profiling rather than final customer segmentation or credit decision groups.

---

## Project Limitations and Data Leakage Considerations

Although the Random Forest and XGBoost models achieved near-perfect ROC-AUC scores, these results should be interpreted with caution.

Several of the most important features identified by the tree-based models were loan pricing and lender-assessment variables, including:

- `Interest_rate_spread`
- `Upfront_charges`
- `rate_of_interest`
- `credit_type_EQUI`

These variables may already contain information about the lender's assessment of borrower risk.

For example, borrowers who were considered riskier may have been assigned higher interest rates, wider interest rate spreads, or higher upfront charges.

As a result, these features may introduce **data leakage**, where the model learns from variables that indirectly encode the lender's prior risk assessment rather than learning only from information available before the loan decision.

Because of this, the near-perfect performance of Random Forest and XGBoost may not reflect real-world predictive performance.

In an actual banking or credit risk setting, further validation would be required, such as:

- Removing or testing pricing-related variables separately
- Comparing model performance before and after removing potential leakage features
- Validating the model on out-of-time or unseen loan data
- Using only features available before loan approval or pricing decisions
- Prioritising explainability, auditability, and regulatory interpretability

Therefore, while Random Forest and XGBoost were useful for learning feature importance and model behaviour, they should not be treated as final production-ready credit risk models.

The Logistic Regression model, with around **0.87 accuracy** and **0.86 ROC-AUC**, may represent a more realistic and interpretable baseline for this educational project because its performance was strong but not suspiciously perfect.

The unsupervised learning section also has limitations.

K-Means clustering was useful for exploring borrower profiles, but the silhouette score was only around **0.136**, which suggests that the clusters were not strongly separated.

This means the three borrower groups should be treated as **exploratory segments** rather than final customer classifications.

K-Means is also distance-based and assumes that clusters are separated in feature space. Real borrower behaviour may be more complex, overlapping, and influenced by factors not captured in the dataset.

In a real banking setting, further segmentation methods such as hierarchical clustering, Gaussian Mixture Models, or business-rule-based segmentation could be tested and compared.

Overall, the supervised learning models are useful for understanding predictive signals, while the unsupervised learning results are useful for early-stage borrower profiling. However, both should be treated as educational and exploratory rather than final credit decision systems.

---

## Final Conclusion

This project demonstrates that loan default risk is more strongly linked to **financing pressure and credit quality** than income alone.

Variables such as `Interest_rate_spread`, `Upfront_charges`, `rate_of_interest`, `LTV`, `dtir1`, and `credit_type_EQUI` emerged as important signals across the supervised analysis.

However, the near-perfect performance of Random Forest and XGBoost suggests possible **data leakage**, especially because several highly predictive variables are related to loan pricing and lender risk assessment.

This means the tree-based models should be treated as useful for learning and feature exploration, but not as final production-ready credit risk models.

The K-Means clustering further supported the idea that borrowers with comparable incomes could belong to very different financial profiles depending on credit score, leverage, and debt burden.

However, because the silhouette score was low, the clustering results should be treated as exploratory borrower profiling rather than a final segmentation model.

Taken together, this project illustrates a simplified but grounded banking credit risk analytics workflow combining:

- Default prediction
- Model interpretation
- Borrower segmentation
- Critical review of model limitations

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
  - Logistic Regression
  - Random Forest
  - K-Means
  - StandardScaler
  - train_test_split
- XGBoost
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Disclaimer

This project is for educational and portfolio purposes only. It does not constitute financial or lending advice.
