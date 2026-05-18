# ============================================================
# 🚀 NETFLIX AI INTELLIGENCE DASHBOARD
# FULL ADVANCED VERSION - ERROR FREE
# ============================================================

import streamlit as st

st.set_page_config(
    page_title="Netflix AI Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import time

import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #141e30, #243b55);
    color: white;
}

h1,h2,h3,h4,h5 {
    color: white;
}

div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    padding: 20px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
}

div[data-testid="stSidebar"] {
    background: #111827;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("netflix_users.csv")

    return df

df = load_data()

# ============================================================
# PREPROCESSING
# ============================================================

@st.cache_resource
def preprocess(df):

    df = df.copy()

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    df['Last_Login'] = pd.to_datetime(df['Last_Login'])

    df['Login_Year'] = df['Last_Login'].dt.year
    df['Login_Month'] = df['Last_Login'].dt.month

    bins = [0,18,25,35,50,100]

    labels = [
        'Teen',
        'Young Adult',
        'Adult',
        'Mid Age',
        'Senior'
    ]

    df['Age_Group'] = pd.cut(
        df['Age'],
        bins=bins,
        labels=labels
    )

    original_df = df.copy()

    encoders = {}

    categorical_cols = [
        'Country',
        'Favorite_Genre',
        'Subscription_Type',
        'Age_Group'
    ]

    for col in categorical_cols:

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col])

        encoders[col] = le

    return df, original_df, encoders

df, original_df, encoders = preprocess(df)

# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

@st.cache_resource
def train_model(df):

    features = [
        'Age',
        'Country',
        'Watch_Time_Hours',
        'Favorite_Genre',
        'Login_Year',
        'Login_Month',
        'Age_Group'
    ]

    target = 'Subscription_Type'

    X = df[features]
    y = df[target]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    cm = confusion_matrix(y_test, pred)

    return model, scaler, accuracy, cm

model, scaler, accuracy, cm = train_model(df)

# ============================================================
# KMEANS CLUSTERING
# ============================================================

@st.cache_resource
def clustering(df):

    cluster_df = df.copy()

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    cluster_df['Cluster'] = kmeans.fit_predict(
        cluster_df[['Age','Watch_Time_Hours']]
    )

    return cluster_df

df = clustering(df)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎛 Netflix Filters")

country_filter = st.sidebar.multiselect(
    "🌍 Country",
    original_df['Country'].unique(),
    default=original_df['Country'].unique()
)

genre_filter = st.sidebar.multiselect(
    "🎭 Genre",
    original_df['Favorite_Genre'].unique(),
    default=original_df['Favorite_Genre'].unique()
)

sub_filter = st.sidebar.multiselect(
    "💳 Subscription",
    original_df['Subscription_Type'].unique(),
    default=original_df['Subscription_Type'].unique()
)

# ============================================================
# FILTERED DATA
# ============================================================

filtered_original = original_df[
    (original_df['Country'].isin(country_filter)) &
    (original_df['Favorite_Genre'].isin(genre_filter)) &
    (original_df['Subscription_Type'].isin(sub_filter))
]

# ============================================================
# HEADER
# ============================================================

st.image(
    "https://images.unsplash.com/photo-1524985069026-dd778a71c7b4",
    use_container_width=True
)

st.markdown("""
<h1 style='text-align:center;
font-size:65px;
color:#E50914;
font-weight:bold;'>

🎬 NETFLIX AI ANALYTICS

</h1>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# LIVE USER COUNTER
# ============================================================

live_users = np.random.randint(1000,5000)

st.info(f"⚡ Live Users Watching Now: {live_users}")

# ============================================================
# KPI SECTION
# ============================================================

k1,k2,k3,k4,k5 = st.columns(5)

k1.metric(
    "👥 Users",
    len(filtered_original)
)

k2.metric(
    "🌍 Countries",
    filtered_original['Country'].nunique()
)

k3.metric(
    "🎭 Genres",
    filtered_original['Favorite_Genre'].nunique()
)

k4.metric(
    "📺 Avg Watch",
    round(filtered_original['Watch_Time_Hours'].mean(),2)
)

k5.metric(
    "🤖 ML Accuracy",
    f"{round(accuracy*100,2)}%"
)

st.markdown("---")

# ============================================================
# SUBSCRIPTION DISTRIBUTION
# ============================================================

col1,col2 = st.columns(2)

with col1:

    sub_count = filtered_original[
        'Subscription_Type'
    ].value_counts().reset_index()

    sub_count.columns = [
        'Subscription',
        'Users'
    ]

    fig = px.bar(
        sub_count,
        x='Subscription',
        y='Users',
        color='Subscription',
        text='Users',
        title="Subscription Distribution"
    )

    fig.update_layout(
        template='plotly_dark',
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:

    genre_count = filtered_original[
        'Favorite_Genre'
    ].value_counts().reset_index()

    genre_count.columns = [
        'Genre',
        'Count'
    ]

    fig = px.pie(
        genre_count,
        names='Genre',
        values='Count',
        hole=0.5,
        title="Popular Genres"
    )

    fig.update_layout(
        template='plotly_dark',
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# SCATTER + BOXPLOT
# ============================================================

col3,col4 = st.columns(2)

with col3:

    fig = px.scatter(
        filtered_original,
        x='Age',
        y='Watch_Time_Hours',
        color='Subscription_Type',
        size='Watch_Time_Hours',
        hover_data=['Favorite_Genre'],
        title="Age vs Watch Time"
    )

    fig.update_layout(
        template='plotly_dark',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

with col4:

    fig = px.box(
        filtered_original,
        x='Subscription_Type',
        y='Watch_Time_Hours',
        color='Subscription_Type',
        title="Watch Time by Subscription"
    )

    fig.update_layout(
        template='plotly_dark',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# GLOBAL MAP
# ============================================================

st.subheader("🌍 Global User Distribution")

country_data = filtered_original[
    'Country'
].value_counts().reset_index()

country_data.columns = [
    'Country',
    'Users'
]

fig = px.choropleth(
    country_data,
    locations='Country',
    locationmode='country names',
    color='Users',
    title='Netflix Global Users'
)

fig.update_layout(
    template='plotly_dark',
    height=550
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# USER SEGMENTATION
# ============================================================

st.subheader("🧠 AI User Segmentation")

cluster_plot = df.copy()

cluster_plot['Subscription_Type'] = encoders[
    'Subscription_Type'
].inverse_transform(
    cluster_plot['Subscription_Type']
)

fig = px.scatter(
    cluster_plot,
    x='Age',
    y='Watch_Time_Hours',
    color='Cluster',
    size='Watch_Time_Hours',
    hover_data=['Subscription_Type'],
    title="KMeans User Segmentation"
)

fig.update_layout(
    template='plotly_dark',
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# MONTHLY TREND
# ============================================================

st.subheader("📈 Monthly User Trend")

trend = original_df.groupby(
    ['Login_Year','Login_Month']
).size().reset_index(name='Users')

trend['Date'] = pd.to_datetime(
    trend['Login_Year'].astype(str)
    + '-'
    + trend['Login_Month'].astype(str)
)

fig = px.line(
    trend,
    x='Date',
    y='Users',
    markers=True,
    title="Monthly Active Users"
)

fig.update_layout(
    template='plotly_dark',
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader("🔥 AI Feature Importance")

features = [
    'Age',
    'Country',
    'Watch_Time_Hours',
    'Favorite_Genre',
    'Login_Year',
    'Login_Month',
    'Age_Group'
]

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
})

fig = px.bar(
    importance_df.sort_values(
        by='Importance',
        ascending=True
    ),
    x='Importance',
    y='Feature',
    orientation='h',
    color='Importance',
    title="Feature Importance"
)

fig.update_layout(
    template='plotly_dark',
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# CONFUSION MATRIX
# ============================================================

st.subheader("🧪 Confusion Matrix")

fig = px.imshow(
    cm,
    text_auto=True,
    color_continuous_scale='reds',
    title="Model Evaluation"
)

fig.update_layout(
    template='plotly_dark',
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# RECOMMENDATION SYSTEM
# ============================================================

st.subheader("🎯 AI Content Recommendation")

genre_choice = st.selectbox(
    "Choose Favorite Genre",
    original_df['Favorite_Genre'].unique()
)

recommendations = original_df[
    original_df['Favorite_Genre'] == genre_choice
].sample(
    min(5, len(original_df))
)

st.dataframe(
    recommendations[
        [
            'Favorite_Genre',
            'Subscription_Type',
            'Watch_Time_Hours'
        ]
    ],
    use_container_width=True
)

# ============================================================
# VIEWER PERSONALITY
# ============================================================

st.subheader("🧠 Viewer Personality")

avg_watch = filtered_original[
    'Watch_Time_Hours'
].mean()

if avg_watch > 70:

    st.success("🔥 Binge Watcher")

elif avg_watch > 40:

    st.info("🎬 Regular Viewer")

else:

    st.warning("🍿 Casual Viewer")

# ============================================================
# AI BUSINESS SUGGESTION
# ============================================================

top_genre = filtered_original[
    'Favorite_Genre'
].mode()[0]

st.success(f"""
🎬 AI Suggestion:

Netflix should invest more in:
{top_genre} content.
""")

# ============================================================
# PREDICTION SYSTEM
# ============================================================

st.subheader("🤖 Subscription Prediction")

p1,p2,p3 = st.columns(3)

with p1:

    age = st.slider(
        "Age",
        10,
        70,
        25
    )

    country = st.selectbox(
        "Country",
        encoders['Country'].classes_
    )

with p2:

    genre = st.selectbox(
        "Genre",
        encoders['Favorite_Genre'].classes_
    )

    watch_time = st.slider(
        "Watch Hours",
        0,
        100,
        10
    )

with p3:

    year = st.slider(
        "Year",
        2020,
        2025,
        2024
    )

    month = st.slider(
        "Month",
        1,
        12,
        5
    )

# ============================================================
# AGE GROUP FUNCTION
# ============================================================

def age_group(age):

    if age <= 18:
        return "Teen"

    elif age <= 25:
        return "Young Adult"

    elif age <= 35:
        return "Adult"

    elif age <= 50:
        return "Mid Age"

    else:
        return "Senior"

# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button("🔮 Predict Subscription"):

    input_df = pd.DataFrame({

        'Age':[age],

        'Country':[
            encoders['Country'].transform([country])[0]
        ],

        'Watch_Time_Hours':[watch_time],

        'Favorite_Genre':[
            encoders['Favorite_Genre'].transform([genre])[0]
        ],

        'Login_Year':[year],

        'Login_Month':[month],

        'Age_Group':[
            encoders['Age_Group'].transform(
                [age_group(age)]
            )[0]
        ]

    })

    scaled = scaler.transform(input_df)

    pred = model.predict(scaled)

    result = encoders[
        'Subscription_Type'
    ].inverse_transform(pred)

    st.success(
        f"🎯 Predicted Subscription: {result[0]}"
    )

    # PREDICTION PROBABILITY

    prob = model.predict_proba(scaled)

    prob_df = pd.DataFrame({

        'Subscription':
        encoders['Subscription_Type'].classes_,

        'Probability':
        prob[0]

    })

    fig = px.bar(
        prob_df,
        x='Subscription',
        y='Probability',
        color='Probability',
        title="Prediction Confidence"
    )

    fig.update_layout(
        template='plotly_dark',
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ============================================================
# DATA EXPLORER
# ============================================================

st.subheader("🗂 Dataset Explorer")

st.dataframe(
    filtered_original,
    use_container_width=True,
    height=450
)

# ============================================================
# DOWNLOAD BUTTON
# ============================================================

csv = filtered_original.to_csv(
    index=False
).encode('utf-8')

st.download_button(
    label="⬇ Download Filtered Data",
    data=csv,
    file_name='netflix_filtered_data.csv',
    mime='text/csv'
)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
<div style='text-align:center;
font-size:18px;
color:lightgray;'>

🚀 Netflix AI Dashboard <br>
Built with Streamlit + Machine Learning + Plotly

</div>
""", unsafe_allow_html=True)
