import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    layout="wide",
    page_icon="🛠️"
)

st.title("🛠️ Predictive Maintenance Dashboard (XGBoost)")
st.markdown("---")

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("ai4i2020.csv")

# Clean column names for XGBoost
df.columns = (
    df.columns.str.replace("[", "", regex=False)
              .str.replace("]", "", regex=False)
              .str.replace(" ", "_")
)

# Drop unnecessary failure breakdown columns
df = df.drop(columns=["UDI", "Product_ID", "TWF", "HDF", "PWF", "OSF", "RNF"])

# Encode Type
df = pd.get_dummies(df, columns=["Type"], drop_first=True)

# Define target
target = "Machine_failure"
X = df.drop(target, axis=1)
y = df[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = XGBClassifier(
    eval_metric="logloss",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.9,
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("⚙️ Dashboard Controls")

show_plots = st.sidebar.checkbox("Show Visualizations", True)
show_importance = st.sidebar.checkbox("Show Feature Importance", True)
show_prediction = st.sidebar.checkbox("Enable Manual Prediction", True)

# -----------------------------
# KPIs
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset Rows", df.shape[0])

with col2:
    st.metric("Dataset Columns", df.shape[1])

with col3:
    st.metric("Model Accuracy", f"{acc*100:.2f}%")

st.markdown("---")

# -----------------------------
# Tabs Layout
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📊 Exploratory Analysis", "🔥 Model Insights", "🧪 Manual Prediction"])

# ============================
# TAB 1 — EDA
# ============================
with tab1:
    st.subheader("📊 Data Overview")
    st.write(df.head())

    st.subheader("Failure Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x=df[target], ax=ax1)
    ax1.set_title("Machine Failure Count")
    st.pyplot(fig1)

    if show_plots:
        st.subheader("Correlation Heatmap")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), cmap="coolwarm", annot=False, ax=ax2)
        st.pyplot(fig2)

# ============================
# TAB 2 — Model Insights
# ============================
with tab2:
    st.subheader("Confusion Matrix")
    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3)
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    st.pyplot(fig3)

    if show_importance:
        st.subheader("Feature Importance")
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)

        fig4, ax4 = plt.subplots(figsize=(6, 8))
        ax4.barh(np.array(X.columns)[sorted_idx], importances[sorted_idx])
        ax4.set_title("XGBoost Feature Importance")
        st.pyplot(fig4)

# ============================
# TAB 3 — Manual Prediction
# ============================
with tab3:
    if show_prediction:
        st.subheader("🧪 Predict Machine Failure")

        colA, colB = st.columns(2)

        with colA:
            air_temp = st.number_input("Air Temperature (K)", value=298.0)
            rpm = st.number_input("Rotational Speed (rpm)", value=1500)
            torque = st.number_input("Torque (Nm)", value=40.0)

        with colB:
            process_temp = st.number_input("Process Temperature (K)", value=308.0)
            tool_wear = st.number_input("Tool Wear (min)", value=10)
            type_val = st.selectbox("Type", ["L", "M", "H"])

        type_L = 1 if type_val == "L" else 0
        type_M = 1 if type_val == "M" else 0

        input_data = pd.DataFrame([{
            "Air_temperature_K": air_temp,
            "Process_temperature_K": process_temp,
            "Rotational_speed_rpm": rpm,
            "Torque_Nm": torque,
            "Tool_wear_min": tool_wear,
            "Type_L": type_L,
            "Type_M": type_M,
        }])

        if st.button("Predict Failure"):
            pred = model.predict(input_data)[0]
            if pred == 1:
                st.error("⚠️ Machine Failure Predicted")
            else:
                st.success("✅ Machine is Safe")
