
# 🛡️ DriftGuard — Real-Time ML Data Drift Detection System

> A production-ready system that monitors whether the data feeding a fraud detection model shifts over time — and automatically alerts when the model needs retraining.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=flat&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange?style=flat&logo=xgboost)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

---

## 📌 What is Data Drift?

When a machine learning model is deployed in production, the real-world data it receives can change over time. For fraud detection, this means **fraudsters adapt their behavior** — making the model less accurate without anyone realizing it.

**DriftGuard solves this problem** by continuously monitoring incoming transaction data and detecting when its statistical distribution shifts away from the training data baseline.

---

## 🚀 What It Does

- Detects **data drift** in production using the **Kolmogorov-Smirnov statistical test**
- Monitors **8 key features** identified by XGBoost feature importance
- Raises an **automatic alert** when retraining is needed
- Built on a fraud detection baseline model with **AUC = 0.977**

---

## 🏗️ Project Structure

```
DriftGuard/
│── fraud_detection.ipynb   # EDA + 5 model comparison
│── model_training.ipynb    # Train + save XGBoost model
│── data_simulator.ipynb    # Generate drift test data
├── api/
│   ├── main.py                    # FastAPI — 3 endpoints
│   └── schemas.py                 # Pydantic request/response models
├── Client/
│   └── app.py                     # Streamlit dashboard
├── drift_detection.py             # Core KS test drift engine
├── models/
│   ├── xgb_model.json             # Saved model
│   └── reference_data.csv         # Training baseline
├── data/
│   ├── drifted_sample.csv         # Demo — triggers drift alert
│   └── normal_sample.csv          # Demo — no drift
└── requirements.txt
```

## 🚀 How It Works

### Step 1 — Train and Save Baseline
XGBoost is trained on the first 70% of transactions ordered by time. This represents the reference period — what normal data looks like.

### Step 2 — New Data Comes In
A batch of new transactions is uploaded via the Streamlit dashboard.

### Step 3 — KS Test Runs
For each of the 8 monitored features, the KS test compares the new data distribution against the reference baseline.

```python
stat, pvalue = ks_2samp(reference_data[feature], new_data[feature])
drifted = pvalue < 0.05  # True = distributions are significantly different
```

### Step 4 — Alert and Recommend
- **p-value < 0.05** → Feature has drifted → 🔴 Alert raised
- **p-value ≥ 0.05** → Feature is stable → 🟢 Model OK

---

---

## ⚙️ Setup

```bash
git clone https://github.com/yourusername/driftguard.git
cd driftguard
pip install -r requirements.txt

# Terminal 1 — FastAPI
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Streamlit
streamlit run Client/app.py
```

Then open `http://localhost:8501`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict` | Fraud probability for one transaction |
| POST | `/check-drift` | Upload CSV → full drift report |

---

## 📊 Model Comparison

| Model | AUC |
|---|---|
| **XGBoost** ⭐ | **0.977** |
| LightGBM | 0.968 |
| CatBoost | 0.858 |
| Random Forest | 0.853 |
| AdaBoost | 0.814 |

---

## 🧪 Test Drift Detection

| File | Expected Result |
|---|---|
| `data/drifted_sample.csv` | 🔴 DRIFT DETECTED |
| `data/normal_sample.csv` | 🟢 NO DRIFT |

---

## 💡 Tech Stack

`Python` `XGBoost` `FastAPI` `Streamlit` `Scikit-learn` `SciPy` `Pandas` `Plotly`

---

## 💡 Key Concepts Demonstrated

- **MLOps** — monitoring models in production, not just training them
- **Statistical Hypothesis Testing** — KS test with p-value threshold of 0.05
- **Time-based train/test split** — realistic simulation of production drift
- **Imbalanced data handling** — only 0.17% of transactions are fraudulent
- **Feature importance driven monitoring** — tracking only the 8 most impactful features
- **REST API design** — clean endpoints with Pydantic validation
- **Full-stack ML app** — from research notebook to deployed dashboard

---
## Demo Video : https://drive.google.com/file/d/1VhX3EN7OApRmGq02zNrG84nAk32wgmWJ/view?usp=drive_link

## 📁 Dataset

[Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
284,807 transactions · 492 fraud cases · 0.17% fraud rate

---

## 👩‍💻 Author

**Shruti** · [GitHub](https://github.com/ShrutiPatel263) · [LinkedIn](https://linkedin.com/in/ShrutiPatel26)

