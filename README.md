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
