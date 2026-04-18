import streamlit as st
import pandas as pd
import numpy as np
import pickle

import gdown
import os

MODEL_PATH = "models/netflix_best_model.pkl"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1AbCXYZ123"
    gdown.download(url, MODEL_PATH, quiet=False)
import requests

url = "https://drive.google.com/file/d/1bHgNdX2Ha_71SvZuMorCsJhluc9BOX9w/view?usp=drive_link"
with open("model.pkl", "wb") as f:
    f.write(requests.get(url).content)

# ===============================
# LOAD MODELS
# ===============================
import os

BASE_DIR = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(BASE_DIR, "models/netflix_best_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "models/netflix_scaler.pkl"), "rb"))
encoders = pickle.load(open(os.path.join(BASE_DIR, "models/netflix_encoders.pkl"), "rb"))

df = pd.read_csv(os.path.join(BASE_DIR, "netflix_users.csv"))

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
# DASHBOARD VISUALIZATION 
# ===============================
import plotly.express as px

st.markdown("---")
st.header("📊 Netflix Analytics Dashboard")

# ===============================
# KPI CARDS
# ===============================
col1, col2, col3 = st.columns(3)

col1.metric("Total Users", len(df))
col2.metric("Avg Watch Time", round(df['Watch_Time_Hours'].mean(), 2))
col3.metric("Top Country", df['Country'].mode()[0])

st.markdown("---")

# ===============================
# ROW 1
# ===============================
col1, col2 = st.columns(2)

with col1:
    sub_counts = df['Subscription_Type'].value_counts().reset_index()
    fig1 = px.bar(sub_counts, x='Subscription_Type', y='count',
                  title="Subscription Distribution")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    genre_counts = df['Favorite_Genre'].value_counts().reset_index()
    fig2 = px.pie(genre_counts, names='Favorite_Genre', values='count',
                  title="Genre Popularity")
    st.plotly_chart(fig2, use_container_width=True)

# ===============================
# ROW 2
# ===============================
col3, col4 = st.columns(2)

with col3:
    fig3 = px.box(df, x='Subscription_Type', y='Watch_Time_Hours',
                  title="Watch Time by Subscription")
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    # Create Age Group if not exists
    if 'Age_Group' not in df.columns:
        bins = [0, 18, 25, 35, 50, 100]
        labels = ['Teen', 'Young Adult', 'Adult', 'Mid Age', 'Senior']
        df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)

    age_grp = df.groupby('Age_Group')['Watch_Time_Hours'].mean().reset_index()
    fig4 = px.bar(age_grp, x='Age_Group', y='Watch_Time_Hours',
                  title="Avg Watch Time by Age Group")
    st.plotly_chart(fig4, use_container_width=True)

# ===============================
# ROW 3
# ===============================
top_countries = df['Country'].value_counts().head(10).reset_index()

fig5 = px.bar(top_countries, x='Country', y='count',
              title="Top 10 Countries")
st.plotly_chart(fig5, use_container_width=True)