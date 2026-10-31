# Customer Churn Prediction

This project uses machine learning to predict whether a telecommunications customer will leave the service. It includes a Jupyter notebook with data exploration, preprocessing, model tuning, and evaluation.

## Data

The included `customer_churn_data.csv` contains 7,043 customer records and 21 columns. It describes customer demographics, subscribed services, contract and payment details, monthly and total charges, and the `Churn` target. Eleven `TotalCharges` values are missing and are handled by median imputation in the notebook. No dataset download link or formal dataset citation was included with
the supplied files.

## Results

The notebook uses a stratified 80/20 train/test split. On a full run in the Python 3.14 environment, the held-out test metrics were:

| Model | Accuracy | Balanced accuracy | Precision (churn) | Recall (churn) | F1 (churn) | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.804 | 0.725 | 0.654 | 0.556 | 0.601 | 0.841 |
| Random Forest | 0.803 | 0.712 | 0.668 | 0.516 | 0.582 | 0.841 |
| K-Nearest Neighbors | 0.790 | 0.661 | 0.686 | 0.385 | 0.493 | 0.801 |

At the default classification threshold, Logistic Regression found the most churners (recall 0.556); its confusion matrix had 166 false negatives and 110 false positives. The notebook also displays specificity, balanced accuracy, and confusion matrices so the models can be compared beyond accuracy alone. Metrics are rounded to three decimal places and may vary across library versions.

## Project source

The supplied project was nested in a folder named `Customer-Churn-Prediction-main`, which suggests it may have come from a downloaded GitHub repository. The supplied files do not identify the original repository, author, or a tutorial, so the exact source cannot be verified or attributed here.

## Run the notebook

Install the notebook dependencies and run it from the repository root so it can find the CSV:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn feature-engine jupyter
jupyter notebook Customer_Churn_Prediction.ipynb
```
