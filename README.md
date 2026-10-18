# Customer Churn Prediction

A machine learning project that predicts whether a bank customer will **leave (churn)** or **stay**, using demographic, financial, and account-activity data. The notebook covers the full workflow: EDA, cleaning, feature engineering, model comparison, evaluation, feature importance, and saving the final model.

## Project Structure

```
.
├── customer_churn_Prediction.ipynb   # Full analysis and modeling notebook
└── README.md
```

## Dataset

- **Size:** 10,000 customers x 12 columns
- **Target:** `churn` (1 = left, 0 = stayed), about 20% positive (imbalanced)

| Column | Description |
|---|---|
| `customer_id` | Unique customer ID (dropped before modeling) |
| `credit_score` | Credit score |
| `country` | France / Spain / Germany |
| `gender` | Male / Female |
| `age` | Age in years |
| `tenure` | Years with the company |
| `balance` | Account balance |
| `products_number` | Number of products held |
| `credit_card` | Has credit card (1/0) |
| `active_member` | Active member (1/0) |
| `estimated_salary` | Estimated salary |
| `churn` | Target |

**Data quality issues handled in the notebook:** `credit_score` and `estimated_salary` are stored as text and contain missing values (164 and 91), and one row has a missing `churn` label.

> The dataset file is not included in this repository. Place `customer_data.csv` in the project root (or in a `data/` folder) before running.

## Workflow

1. **EDA:** distributions by churn, pairplot, violin and count plots, correlation heatmap, churn rate by number of products
2. **Cleaning:** fix data types, mean-impute missing numeric values, drop the row with a missing target, drop `customer_id`
3. **Feature engineering:** `balance_per_product`, `salary_balance_ratio`, `age_group`, `tenure_bucket`, `high_balance`
4. **Preprocessing pipeline:** median impute + standard scaling (numeric); most-frequent impute + one-hot encoding (categorical)
5. **Model comparison:** 5-fold stratified cross-validation on ROC AUC
6. **Evaluation:** hold-out test set (80/20 stratified split), confusion matrix, feature importance
7. **Inference example and model saving** with `joblib`

## Results

Cross-validated ROC AUC (training set):

| Model | Mean AUC |
|---|---|
| Logistic Regression | 0.7862 |
| Random Forest | 0.8503 |
| **Gradient Boosting** | **0.8610** |
| AdaBoost | 0.8451 |
| SVC | 0.8357 |

Gradient Boosting on the held-out test set:

| Metric | Score |
|---|---|
| Accuracy | 0.8675 |
| Precision | 0.8034 |
| Recall | 0.4619 |
| F1-score | 0.5866 |
| ROC AUC | 0.8693 |

**Note:** Accuracy looks strong, but recall on churners is only about 46%, so the model misses over half of the customers who actually leave. Likely next steps: class weighting or resampling, threshold tuning, and hyperparameter search (`GridSearchCV` is already imported).

## Getting Started

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook customer_churn_Prediction.ipynb
```

Run all cells top to bottom. The last cell saves the trained pipeline to `models/churn_model.joblib`.

## Using the Saved Model

```python
import joblib
import pandas as pd

model = joblib.load("models/churn_model.joblib")

# Input must contain the engineered features used in training:
# credit_score, country, gender, age, tenure, balance, products_number,
# credit_card, active_member, estimated_salary,
# balance_per_product, salary_balance_ratio, age_group, tenure_bucket, high_balance
prob = model.predict_proba(new_customer_df)[:, 1]
```

See the "Predict churn for a new customer" section of the notebook for how to build these features from raw inputs.

## Tech Stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, joblib, Jupyter
