# 🔧 Predictive Maintenance using Machine Learning

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost-orange)
![Status](https://img.shields.io/badge/Project-Completed-green)

---

## 📌 Problem Statement
In industrial environments, unexpected machine failures lead to high maintenance costs and production downtime.

This project aims to build a **Predictive Maintenance system** that detects machine failures before they occur using sensor data.

---

## 🎯 Objective
- Predict machine failure (0 = No Failure, 1 = Failure)
- Reduce downtime and maintenance cost
- Improve operational efficiency

---

## 📊 Dataset Features
- Air Temperature [K]
- Process Temperature [K]
- Rotational Speed [rpm]
- Torque [Nm]
- Tool Wear [min]
- Machine Type (L, M, H)

Target:
- Machine failure

---

## 🧠 Machine Learning Pipeline

### 1️⃣ Data Preprocessing
- Handling missing values
- Encoding categorical features
- Feature scaling (StandardScaler)

### 2️⃣ Handling Imbalance
- SMOTE applied only on training data

### 3️⃣ Models Used
- XGBoost Classifier ⭐ (Best Model)
- Random Forest (Baseline)

---

## 🏆 Best Model Performance
- High Recall for failure detection
- Balanced precision/recall trade-off
- Optimized threshold for better failure detection

---

## 🖥️ Web Application
A Streamlit-based web app allows real-time prediction of machine failure.
