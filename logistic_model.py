import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load data
df = pd.read_csv("insurance_claims(1).csv")

# Drop target column from features
X = df.drop("fraud_reported", axis=1)
y = df["fraud_reported"].map({'Y': 1, 'N': 0})

# Replace missing values with a placeholder string
X = X.fillna("Missing")

# Separate categorical and numerical columns
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(exclude='object').columns.tolist()

# One-hot encode categorical columns
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_data = encoder.fit_transform(X[categorical_cols].astype(str))
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))

# Combine numerical and encoded categorical data
X_final = pd.concat([X[numerical_cols].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, "logistic_model.pkl")
joblib.dump(encoder, "encoders.pkl")
joblib.dump(X_final.columns.tolist(), "all_feature_columns.pkl")
joblib.dump(categorical_cols + numerical_cols[:3], "feature_columns.pkl")  # Just saving 3 numericals for demo

print("✅ Model, encoders, and feature columns saved successfully!")
