import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

# === Load assets ===
model = xgb.XGBClassifier()
model.load_model("model_xgb.json")
expected_features = joblib.load("model_features.pkl")
df = pd.read_csv("preprocessed_train.csv")

# === Streamlit UI ===
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("Customer Churn Analysis & Prediction")

# === EDA Section ===
with st.expander("Show Exploratory Data Analysis"):
    st.subheader("Data Preview")
    st.dataframe(df.head())

    st.subheader("Churn Distribution")
    churn_counts = df["churned"].value_counts()
    st.bar_chart(churn_counts)

    st.subheader("Feature Comparison")
    numeric_features = [f for f in df.columns if f != "churned" and pd.api.types.is_numeric_dtype(df[f])]
    selected_features = st.multiselect("Select features to compare with churn", numeric_features[:10])
    if selected_features:
        for feature in selected_features:
            st.write(f"Feature: {feature}")
            means = df.groupby('churned')[feature].mean()
            means.plot(kind='bar', color=['skyblue', 'salmon'])
            plt.ylabel(f"Average {feature}")
            plt.xlabel('Churned')
            plt.xticks(rotation=0)
            st.pyplot(plt.gcf())
            plt.clf()

    st.subheader("Correlation Heatmap")
    corr = df.select_dtypes(include=["number"]).corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.subheader("Insights & Recommendations")
    st.markdown("""
    - High **skip rate** and low **engagement score** increase churn risk.
    - Users with lower **session length** and **weekly listening hours** are more likely to churn.
    - Focus marketing efforts on users with low activity metrics and target them with notifications.
    """)

# === Sidebar: Single Prediction UI ===
st.sidebar.markdown("## ✍️ Predict Single Customer")
def get_user_input():
    age = st.sidebar.slider("Age", 18, 70, 35)
    weekly_hours = st.sidebar.slider("Weekly Listening Hours", 0, 40, 10)
    session_length = st.sidebar.slider("Avg. Session Length", 5, 60, 20)
    skip_rate = st.sidebar.slider("Skip Rate (%)", 0, 100, 20)
    notifications_clicked = st.sidebar.slider("Notifications Clicked", 0, 50, 5)
    account_months = st.sidebar.slider("Account Age (Months)", 1, 60, 12)
    engagement_score = st.sidebar.slider("Engagement Score", 0.0, 1.0, 0.5)
    return pd.DataFrame({
        'age': [age],
        'weekly_hours': [weekly_hours],
        'average_session_length': [session_length],
        'skip_rate': [skip_rate],
        'notifications_clicked': [notifications_clicked],
        'account_months': [account_months],
        'engagement_score': [engagement_score]
    })

user_input_df = get_user_input()
user_input_df = user_input_df.reindex(columns=expected_features, fill_value=0)

if st.sidebar.button("Predict Churn"):
    proba = model.predict_proba(user_input_df)[0]
    pred=np.argmax(proba)
    if pred == 1:
        st.error(f"Prediction: Customer likely to churn")
    else:
        st.success(f"Prediction: Customer likely to stay")
