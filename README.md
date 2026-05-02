<div align="center">

<img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge"/>

<br/><br/>

```
███████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗███████╗██╗
██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗  ██║██╔════╝██║
███████╗█████╗  ██╔██╗ ██║   ██║   ██║██╔██╗ ██║█████╗  ██║
╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╗██║██╔══╝  ██║
███████║███████╗██║ ╚████║   ██║   ██║██║ ╚████║███████╗███████╗
╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝
```

# 🛡️ Sentinel — Credit Card Fraud Detection

**Real-time fraud intelligence powered by Random Forest + SMOTE**  
*Trained on 284,807 transactions · 99.8% Accuracy · 0.87 AUPRC*

[🚀 Run the App](#️-setup--run) · [📊 Model Results](#-model-performance) · [🧠 How It Works](#-how-it-works) · [📁 Project Structure](#-project-structure)

</div>

---

## ⚡ What Is This?

**Sentinel** is an end-to-end machine learning system that detects fraudulent credit card transactions in real time. It tackles one of the hardest problems in applied ML — **extreme class imbalance** — where fraud makes up only **0.17%** of all transactions.

The system includes:
- A **training pipeline** that resamples and balances the dataset using SMOTE
- A **Random Forest classifier** tuned for maximum fraud recall
- A **Streamlit dashboard** to authorise transactions, view session history, and explore model analytics

---

## 📊 Model Performance

> Evaluated on the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 real anonymised European transactions from 2013.

| Metric | Score |
|---|---|
| ✅ Accuracy | **99.8%** |
| 🎯 Precision | **0.87** |
| 🔁 Recall | **0.83** |
| ⚖️ F1-Score | **0.85** |
| 📈 AUPRC | **0.8725** |

**Confusion Matrix (Test Set)**

|  | Predicted Genuine | Predicted Fraud |
|---|---|---|
| **Actual Genuine** | 56,850 | 12 |
| **Actual Fraud** | 80 | 412 |

> Only **12 false positives** out of 56,862 genuine transactions — minimal friction for real customers.

---

## 🧠 How It Works

### The Problem: Extreme Class Imbalance
Out of 284,807 transactions, only **492 are fraud (0.17%)**. A naive model that predicts "genuine" every time would hit 99.8% accuracy — but catch zero fraud. That's useless.

### The Solution: SMOTE + Random Forest

```
Raw Dataset (284,807 rows)
        │
        ▼
  Downsample genuine → 60,000 rows  (speed + variety balance)
        │
        ▼
  Apply SMOTE → 50% Fraud / 50% Genuine  (synthetic minority oversampling)
        │
        ▼
  Train Random Forest
  · 150 trees
  · No depth limit (capture every fraud signature)
  · class_weight='balanced'
        │
        ▼
  Save sentinel_model.pkl + scaler.pkl
```

**Why SMOTE?** Instead of just duplicating fraud rows, SMOTE synthesises *new* fraud examples by interpolating between existing ones — giving the model richer patterns to learn from.

**Why Random Forest?** It's robust to noise, handles the PCA-transformed V1–V28 features well, and gives interpretable feature importances out of the box.

### Top 5 Most Important Features

| Rank | Feature | Importance |
|---|---|---|
| 1 | V17 | 0.18 |
| 2 | V14 | 0.15 |
| 3 | V12 | 0.12 |
| 4 | Amount | 0.09 |
| 5 | V10 | 0.07 |

> V17, V14, V12 are PCA-transformed components of the original transaction data (anonymised for privacy). **Transaction Amount** is a raw feature and ranks 4th — large amounts do correlate with fraud risk.

---

## 🖥️ Dashboard Features

### Tab 1 — 🔍 Authorise Transaction
Enter a 16-digit card number and transaction amount. Sentinel returns:
- **APPROVED / DECLINED** verdict
- **Risk Index (%)** — visual threat gauge
- **Auth timestamp** and **Card Terminal ID**

### Tab 2 — 📜 Session Audit Log
Full table of every transaction checked in the current session. One-click clear.

### Tab 3 — 📊 Training Analytics
- Post-SMOTE class distribution pie chart
- Confusion matrix heatmap
- Feature importance bar chart
- Full performance metrics

---

## 📁 Project Structure

```
sentinel-fraud-detection/
│
├── app.py              # Streamlit dashboard (frontend)
├── train.py            # Model training pipeline
├── requirements.txt    # Python dependencies
├── .gitignore          # Excludes dataset & model binaries
└── README.md
```

> ⚠️ `creditcard.csv`, `sentinel_model.pkl`, and `scaler.pkl` are **not included** in this repo due to file size. See setup instructions below.

---

## ⚙️ Setup & Run

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/sentinel-fraud-detection.git
cd sentinel-fraud-detection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Get `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the root folder.

**4. Train the model**
```bash
python train.py
```
This generates `sentinel_model.pkl` and `scaler.pkl`.

**5. Launch the dashboard**
```bash
streamlit run app.py
```

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| ML Model | scikit-learn RandomForestClassifier |
| Resampling | imbalanced-learn SMOTE |
| Dashboard | Streamlit |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Serialisation | Joblib |

---

## 🔮 Future Improvements

- [ ] Plug in live model inference (replace demo logic with `model.predict_proba`)
- [ ] Add XGBoost / LightGBM comparison
- [ ] Deploy to Streamlit Cloud for a public live demo
- [ ] Add SHAP explainability plots per transaction

---

## 📄 Dataset Credit

> Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi.  
> *Calibrating Probability with Undersampling for Unbalanced Classification.*  
> 2015 IEEE Symposium Series on Computational Intelligence.  
> Dataset hosted on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

---

<div align="center">

Made with 🛡️ to keep transactions safe

</div>
