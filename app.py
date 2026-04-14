import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ===============================
# LOAD MODELS
# ===============================
model = pickle.load(open("models/netflix_best_model.pkl", "rb"))
scaler = pickle.load(open("models/netflix_scaler.pkl", "rb"))
encoders = pickle.load(open("models/netflix_encoders.pkl", "rb"))

st.title("🎬 Netflix User Prediction Dashboard")

# ===============================
# SIDEBAR INPUT
# ===============================
st.sidebar.header("Enter User Details")

age = st.sidebar.slider("Age", 10, 70, 25)
country = st.sidebar.selectbox("Country", encoders['Country'].classes_)
genre = st.sidebar.selectbox("Favorite Genre", encoders['Favorite_Genre'].classes_)
watch_time = st.sidebar.slider("Watch Time (Hours)", 0, 100, 10)
month = st.sidebar.slider("Login Month", 1, 12, 5)
year = st.sidebar.slider("Login Year", 2020, 2025, 2024)

# Age Group logic
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
# ENCODE INPUT
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

# Encode
for col in input_data.columns:
    if col in encoders:
        input_data[col] = encoders[col].transform(input_data[col])
# Scale
input_scaled = scaler.transform(input_data)

# ===============================
# PREDICTION
# ===============================
if st.button("Predict Subscription Type"):
    prediction = model.predict(input_scaled)
    
    # Decode output
    sub_encoder = encoders['Subscription_Type']
    result = sub_encoder.inverse_transform(prediction)
    
    st.success(f"Predicted Subscription: {result[0]}")

# ===============================
# DATA VISUALIZATION
# ===============================
st.subheader("📊 Dataset Overview")

df = pd.read_csv("netflix_users.csv")
st.write(df.head())

st.subheader("📈 Users by Country")
st.bar_chart(df['Country'].value_counts())

st.subheader("🎭 Favorite Genres")
st.bar_chart(df['Favorite_Genre'].value_counts())