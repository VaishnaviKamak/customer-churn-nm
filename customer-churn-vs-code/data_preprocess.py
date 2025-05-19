# 📦 1. Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

pd.set_option('display.max_columns', None)

# 📥 2. Load the dataset
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 🕵️ 3. First glance at the dataset
print(train_df.shape)
train_df.head()

for df in [train_df, test_df]:
    df['account_months'] = (-df['signup_date']) / 30
    df['account_months'] = df['account_months'].round(1)

# Optional: Print to check shape and column list
print("Train Columns:", train_df.columns.tolist())
print("Test Columns:", test_df.columns.tolist())

def calculate_engagement_score(df):
    return (
        df['weekly_hours'] * 2 +
        df['average_session_length'] * 1.5 +
        df['notifications_clicked'] * 1.2 +
        df['num_favorite_artists'] * 1 -
        df['num_subscription_pauses'] * 3
    )

# Apply to both train and test
train_df['engagement_score'] = calculate_engagement_score(train_df)
test_df['engagement_score'] = calculate_engagement_score(test_df)

# Export preprocessed training dataset
train_df.to_csv('preprocessed_train.csv', index=False)

# Export preprocessed test dataset
test_df.to_csv('preprocessed_test.csv', index=False)