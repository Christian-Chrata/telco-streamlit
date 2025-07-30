import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Judul Utama
st.title("Survival Rate Predictor (Telco Churn)")
st.text("This web can be used to predict customer survival rate (non-churn)")

# Load model
with open(r"/mount/src/deploy_model/model.sav", "rb") as f:
    model = pickle.load(f)

# Load struktur data (tanpa label target)
sample_data = pd.read_csv("df_clean.csv").drop(columns=["customerID", "Churn"])
required_columns = sample_data.columns.tolist()

# Sidebar untuk input user
st.sidebar.header("Please input your features")

def create_user_input():
    user_data = {}

    for col in required_columns:
        if sample_data[col].dtype == "object":
            options = sample_data[col].dropna().unique().tolist()
            user_data[col] = st.sidebar.selectbox(f"{col}:", options)
        else:
            min_val = float(sample_data[col].min())
            max_val = float(sample_data[col].max())
            mean_val = float(sample_data[col].mean())
            user_data[col] = st.sidebar.slider(f"{col}:", min_val, max_val, mean_val)

    user_data_df = pd.DataFrame([user_data])
    return user_data_df

# Get input dari user
data_customer = create_user_input()

# Split tampilan horizontal
col1, col2 = st.columns(2)

# Tampilkan fitur yang dimasukkan
with col1:
    st.subheader("Customer's Input Features")
    st.write(data_customer.transpose())

# Prediksi menggunakan model
def predict(df_input):
    pred_class = model.predict(df_input)[0]
    pred_proba = model.predict_proba(df_input)[0][1]
    return pred_class, pred_proba

# Tampilkan hasil prediksi
with col2:
    st.subheader("Prediction Result")

    if st.button("Predict"):
        kelas, probability = predict(data_customer)

        if kelas == 1:
            st.success("✅ Class 1: This customer is **likely to stay** (survive).")
        else:
            st.error("⚠️ Class 0: This customer is **likely to churn** (not survive).")

        st.metric("Probability of Survival", f"{probability:.2f}")
