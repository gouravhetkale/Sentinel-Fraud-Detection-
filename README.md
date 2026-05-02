# 🛡️ Sentinel — Credit Card Fraud Detection

Real-time fraud detection dashboard built with **Random Forest + SMOTE** on 284,807 transactions.

---

## Results

| Metric | Score |
|---|---|
| Accuracy | 99.8% |
| Precision | 0.87 |
| Recall | 0.83 |
| F1-Score | 0.85 |
| AUPRC | 0.8725 |

---

## How It Works

The dataset has a severe class imbalance — only 0.17% of transactions are fraud. SMOTE synthesises new fraud examples to create a balanced 50/50 training set, then a Random Forest (150 trees) is trained on top.

---

## Run It

```bash
pip install -r requirements.txt

# Add creditcard.csv from Kaggle to the root folder
python train.py       # trains and saves the model
streamlit run app.py  # launches the dashboard
```

> Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## Stack
`Python` · `scikit-learn` · `imbalanced-learn` · `Streamlit` · `Pandas` · `Matplotlib`
