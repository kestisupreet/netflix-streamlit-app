import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import requests

# ===============================
# DOWNLOAD MODEL FILES (RELIABLE)
# ===============================
def download_file(url, filename):
    if not os.path.exists(filename):
        with requests.get(url, stream=True) as r:
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

os.makedirs("models", exist_ok=True)

# 🔥 REPLACE THESE LINKS WITH YOUR REAL ONES
MODEL_URL = "https://github.com/kestisupreet/netflix-streamlit-app/releases/download/v1/netflix_best_model.pkl"
SCALER_URL = "https://github.com/kestisupreet/netflix-streamlit-app/releases/download/v1/netflix_scaler.pkl"
ENCODER_URL = "https://github.com/kestisupreet/netflix-streamlit-app/releases/download/v1/netflix_encoders.pkl"

download_file(MODEL_URL, "models/model.pkl")
download_file(SCALER_URL, "models/scaler.pkl")
download_file(ENCODER_URL, "models/encoders.pkl")

# ===============================
# LOAD FILES
# ===============================
model = pickle.load(open("models/model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
encoders = pickle.load(open("models/encoders.pkl", "rb"))

df = pd.read_csv("netflix_users.csv")

st.title("🎬 Netflix User Prediction Dashboard")

# ===============================
# SIDEBAR INPUT
# ===============================
st.sidebar.header("Enter User Details")

age = st.sidebar.slider("Age", 10, 70, 25)
country = st.sidebar.selectbox("Country", encoders['Country'].classes_)
genre = st.sidebar.selectbox("Favorite Genre", encoders['Favorite_Genre'].classes_)
watch_time = st.sidebar.slider("Watch Time", 0, 100, 10)
month = st.sidebar.slider("Login Month", 1, 12, 5)
year = st.sidebar.slider("Login Year", 2020, 2025, 2024)

def get_age_group(age):
    if age <= 18:
        return 'Teen'
    elif age <= 25:
        return 'Young Adult'
    elif age <= 35:
        return 'Adult'
    elif age <= 50:
        return 'Mid Age'
    else:
        return 'Senior'

age_group = get_age_group(age)

# ===============================
# PREPROCESS INPUT
# ===============================
input_data = pd.DataFrame({
    'Age': [age],
    'Country': [country],
    'Watch_Time_Hours': [watch_time],
    'Favorite_Genre': [genre],
    'Login_Year': [year],
    'Login_Month': [month],
    'Age_Group': [age_group]
})

for col in input_data.columns:
    if col in encoders:
        input_data[col] = encoders[col].transform(input_data[col])

input_scaled = scaler.transform(input_data)

# ===============================
# PREDICTION
# ===============================
if st.button("Predict"):
    pred = model.predict(input_scaled)
    result = encoders['Subscription_Type'].inverse_transform(pred)
    st.success(f"Predicted: {result[0]}")

# ===============================
# DASHBOARD
# ===============================
import plotly.express as px

st.header("📊 Analytics")

col1, col2, col3 = st.columns(3)
col1.metric("Users", len(df))
col2.metric("Avg Watch Time", round(df['Watch_Time_Hours'].mean(), 2))
col3.metric("Top Country", df['Country'].mode()[0])

# Charts
fig1 = px.bar(df['Subscription_Type'].value_counts().reset_index(),
              x='Subscription_Type', y='count')
st.plotly_chart(fig1)

fig2 = px.pie(df, names='Favorite_Genre')
st.plotly_chart(fig2)
