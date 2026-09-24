# Telco Customer Churn Prediction

Predicting which telecom customers are likely to churn (cancel their subscription), using the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle.

**Goal:** identify customers at risk of churning so the business can proactively target them with retention offers.

---

## Dataset

- **Source:** [IBM Sample Data / Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Rows:** 7,043 customers
- **Target:** `Churn` (Yes/No)
- **Features:** demographic info (gender, senior citizen, partner, dependents), account info (tenure, contract type, payment method, monthly/total charges), and subscribed services (internet, phone, streaming, security add-ons)

---

## Project Structure

```
.
├── churn_analysis.ipynb   # Main notebook: EDA, preprocessing, modeling, evaluation
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook churn_analysis.ipynb
```

**requirements.txt**
```
pandas
numpy
matplotlib
scikit-learn
```

---

## Methodology

1. **Data cleaning** — handled missing/blank values in `TotalCharges`, checked for duplicates
2. **Exploratory Data Analysis (EDA)** — churn rate by contract type, payment method, internet service, tenure, and monthly charges
3. **Preprocessing** — `StandardScaler` for numeric features, `OneHotEncoder` for categorical features, combined via `ColumnTransformer`
4. **Modeling** — baseline Logistic Regression, Random Forest, class-weighted Logistic Regression
5. **Threshold tuning** — 5-fold cross-validated out-of-fold predictions to find the F1-optimal decision threshold
6. **Evaluation** — accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, ROC/PR curves

---

## Key EDA Findings

- Customers on **month-to-month contracts** churn at 42.71%, vs. 11.27% for one-year and just 2.83% for two-year contracts.
- Customers paying by **electronic check** churn the most (45.29%), while automatic credit card payments churn the least (15.24%).
- Churn is strongly associated with **shorter tenure** and **higher monthly charges**.

---

## Model Evaluation & Selection

Four approaches were compared on the held-out test set:

| Metric | Baseline LR | Random Forest | Weighted LR | Weighted LR + Tuned Threshold (0.60) |
|---|---|---|---|---|
| Accuracy | 0.8055 | 0.7821 | 0.7381 | 0.7622 |
| Precision | 0.6572 | 0.6143 | 0.5043 | 0.5399 |
| Recall | 0.5588 | 0.4813 | 0.7834 | 0.7059 |
| F1-score | 0.6040 | 0.5397 | 0.6136 | **0.6118** |
| ROC-AUC | 0.8421 | 0.8195 | 0.8416 | 0.8416 |

**Analysis:**

- **Random Forest** underperformed the baseline across nearly every metric (F1 = 0.54, ROC-AUC = 0.82), likely due to using default hyperparameters without tuning.
- **Baseline Logistic Regression** achieved the highest accuracy and precision, but a relatively low recall (0.56) — meaning it misses a large share of customers who actually churn, which is costly from a business standpoint.
- **Class-weighted Logistic Regression** substantially increased recall (0.78) by making the model more sensitive to the minority (churn) class, at the cost of precision and accuracy.
- **Weighted LR with a tuned threshold (0.60)**, selected via cross-validated out-of-fold F1 optimization, gives the best overall trade-off: it keeps most of the recall gain (0.71 vs. 0.78) while recovering some precision (0.54 vs. 0.50), resulting in the best (or near-best) F1-score among all approaches.

**Selected model:** class-weighted Logistic Regression with a tuned decision threshold of 0.60. This choice reflects the business assumption that **the cost of missing a customer who will churn outweighs the cost of an unnecessary retention offer to a loyal customer**. The threshold can be adjusted further depending on the actual cost ratio between false negatives and false positives.

---

## Conclusion

- **Best model:** class-weighted Logistic Regression, outperforming both the unweighted baseline and Random Forest on recall/F1 for the churn class.
- **Best decision threshold:** ~0.60, selected via 5-fold cross-validation, giving a cross-validated F1-score of ~0.64.
- **Key churn drivers:** month-to-month contracts, electronic check payments, and fiber-optic internet service are all associated with higher churn rates.
- **Business takeaway:** the model surfaces a meaningfully larger share of at-risk customers than a naive baseline, at the cost of more false positives — the production threshold should ultimately be chosen based on the real cost of a missed churner vs. an unnecessary retention offer.

## Next Steps

- Try gradient boosting models (XGBoost / LightGBM) with hyperparameter tuning
- Add feature importance / SHAP analysis for interpretability
- Deploy the final pipeline as a simple API or Streamlit app

---

## Tech Stack

`Python` · `pandas` · `scikit-learn` · `matplotlib`
