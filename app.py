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
# 🎨 POWER BI STYLE UI
# ===============================
import plotly.express as px

st.set_page_config(layout="wide")

# ===============================
# HEADER
# ===============================
st.markdown("""
    <h1 style='text-align: center; color: #E50914;'>
        🎬 Netflix Analytics Dashboard
    </h1>
""", unsafe_allow_html=True)

st.markdown("---")

# ===============================
# SIDEBAR FILTERS
# ===============================
st.sidebar.header("🔍 Filters")

selected_country = st.sidebar.multiselect(
    "Select Country",
    options=df['Country'].unique(),
    default=df['Country'].unique()
)

selected_subscription = st.sidebar.multiselect(
    "Subscription Type",
    options=df['Subscription_Type'].unique(),
    default=df['Subscription_Type'].unique()
)

# Filter data
filtered_df = df[
    (df['Country'].isin(selected_country)) &
    (df['Subscription_Type'].isin(selected_subscription))
]

# ===============================
# KPI CARDS
# ===============================
col1, col2, col3, col4 = st.columns(4)

col1.metric("👥 Total Users", len(filtered_df))
col2.metric("⏱ Avg Watch Time", round(filtered_df['Watch_Time_Hours'].mean(), 2))
col3.metric("🌍 Countries", filtered_df['Country'].nunique())
col4.metric("🎭 Genres", filtered_df['Favorite_Genre'].nunique())

st.markdown("---")

# ===============================
# ROW 1
# ===============================
col1, col2 = st.columns(2)

with col1:
    sub_counts = filtered_df['Subscription_Type'].value_counts().reset_index()
    sub_counts.columns = ['Subscription_Type', 'count']

    fig1 = px.bar(
        sub_counts,
        x='Subscription_Type',
        y='count',
        color='Subscription_Type',
        title="Subscription Distribution"
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    genre_counts = filtered_df['Favorite_Genre'].value_counts().reset_index()
    genre_counts.columns = ['Genre', 'count']

    fig2 = px.pie(
        genre_counts,
        names='Genre',
        values='count',
        title="Genre Popularity"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ===============================
# ROW 2
# ===============================
col3, col4 = st.columns(2)

with col3:
    fig3 = px.box(
        filtered_df,
        x='Subscription_Type',
        y='Watch_Time_Hours',
        color='Subscription_Type',
        title="Watch Time by Subscription"
    )
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    if 'Age_Group' not in filtered_df.columns:
        bins = [0, 18, 25, 35, 50, 100]
        labels = ['Teen', 'Young Adult', 'Adult', 'Mid Age', 'Senior']
        filtered_df['Age_Group'] = pd.cut(filtered_df['Age'], bins=bins, labels=labels)

    age_grp = filtered_df.groupby('Age_Group')['Watch_Time_Hours'].mean().reset_index()

    fig4 = px.bar(
        age_grp,
        x='Age_Group',
        y='Watch_Time_Hours',
        color='Age_Group',
        title="Avg Watch Time by Age Group"
    )
    st.plotly_chart(fig4, use_container_width=True)

# ===============================
# ROW 3
# ===============================
top_countries = filtered_df['Country'].value_counts().head(10).reset_index()
top_countries.columns = ['Country', 'count']

fig5 = px.bar(
    top_countries,
    x='Country',
    y='count',
    color='Country',
    title="Top 10 Countries"
)

st.plotly_chart(fig5, use_container_width=True)
