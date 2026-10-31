# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer will leave the service (churn). The notebook covers data exploration, preprocessing, model training, tuning and evaluation.

## Dataset

`customer_churn_data.csv` has 7,043 customer records and 21 columns:

- Customer demographics
- Subscribed services
- Contract and payment details
- Monthly and total charges
- `Churn` (the target I predict)

Eleven `TotalCharges` values were missing, so I filled them with the median in the notebook.

## What I did

1. Explored the data (distributions, churn rate, relationships between features)
2. Cleaned and preprocessed it (missing values, encoding categorical columns)
3. Split the data 80/20 (stratified) into train and test sets
4. Trained and tuned three models: Logistic Regression, Random Forest and K-Nearest Neighbors
5. Compared them using accuracy, balanced accuracy, precision, recall, F1, AUROC and confusion matrices

## Results

Held-out test set results:

| Model | Accuracy | Balanced accuracy | Precision (churn) | Recall (churn) | F1 (churn) | AUROC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.804 | 0.725 | 0.654 | 0.556 | 0.601 | 0.841 |
| Random Forest | 0.803 | 0.712 | 0.668 | 0.516 | 0.582 | 0.841 |
| K-Nearest Neighbors | 0.790 | 0.661 | 0.686 | 0.385 | 0.493 | 0.801 |

**Takeaway:** Logistic Regression found the most churners (recall 0.556), with 166 false negatives and 110 false positives. Accuracy alone can be misleading here because most customers do not churn, so I also looked at balanced accuracy, recall and the confusion matrices. Exact numbers may vary slightly with library versions.

## Tech stack

- Python
- pandas, NumPy
- scikit-learn
- feature-engine
- matplotlib, seaborn
- Jupyter Notebook

## How to run

1. Clone the repo:
   ```bash
   git clone https://github.com/vivekreddy-25/customer-churn-prediction.git
   cd customer-churn-prediction
   ```
2. Install the dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn feature-engine jupyter
   ```
3. Start Jupyter from the repo folder (so the notebook can find the CSV) and open the notebook:
   ```bash
   jupyter notebook Customer_Churn_Prediction.ipynb
   ```

## Project files

- `Customer_Churn_Prediction.ipynb` - the full analysis and modelling notebook
- `customer_churn_data.csv` - the dataset

## What I could improve

- Try other models such as XGBoost
- Tune the classification threshold to catch more churners
- Handle the class imbalance (for example with class weights)

## Author

Vivek Macharla - M.Sc. AI & Big Data, SRH Berlin
[GitHub](https://github.com/vivekreddy-25) | [LinkedIn](https://www.linkedin.com/in/vivek-reddy-macharla-93a5b5282)