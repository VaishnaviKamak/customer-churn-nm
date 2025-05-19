import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load and preprocess
df = pd.read_csv("preprocessed_train.csv")

#Drop unwanted columns
columns_to_drop = [
    'customer_id',
    'signup_date',
    'location'
]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

#One-hot encoding
df = pd.get_dummies(df, drop_first=True)

#Split into features and labels
X = df.drop(columns=["churned"])
y = df["churned"]

# Save the list of features for use in app.py
joblib.dump(X.columns.tolist(), "model_features.pkl")

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === Train XGBoost ===
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_model.save_model("model_xgb.json")  # ✅ Save as JSON

# Evaluation
print("\nXGBoost Results:")
print("Accuracy:", accuracy_score(y_val, xgb_model.predict(X_val)))
print("Confusion Matrix:\n", confusion_matrix(y_val, xgb_model.predict(X_val)))
print("Classification Report:\n", classification_report(y_val, xgb_model.predict(X_val)))
print("XGBoost model and feature list saved successfully.")