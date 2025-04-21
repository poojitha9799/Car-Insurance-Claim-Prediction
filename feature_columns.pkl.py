
import pandas as pd
import joblib

df = pd.read_csv("insurance_claims(1).csv")
df = df.fillna("Missing")

# Drop target
X = df.drop("fraud_reported", axis=1)
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(exclude='object').columns.tolist()

# Example: take all categorical + top 5 numerical columns
important_features = categorical_cols + numerical_cols[:5]
joblib.dump(important_features, "feature_columns.pkl")

print("✅ Saved corrected important features list.")
