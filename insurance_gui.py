import streamlit as st
import pandas as pd
import joblib

# Load trained model and supporting files
model = joblib.load("logistic_model.pkl")
encoder = joblib.load("encoders.pkl")
all_features = joblib.load("all_feature_columns.pkl")
important_features = joblib.load("feature_columns.pkl")

# Streamlit page settings
st.set_page_config(page_title="Insurance Claim Eligibility Checker", layout="centered")
st.title("🚗 Insurance Claim Eligibility Checker")
st.markdown("Enter the following details to check claim eligibility.")

# Input from user
user_data = {}

for feature in important_features:
    if feature in encoder.feature_names_in_:
        options = encoder.categories_[encoder.feature_names_in_.tolist().index(feature)]
        user_data[feature] = st.selectbox(f"{feature.replace('_', ' ').capitalize()}", options)
    else:
        user_data[feature] = st.number_input(f"{feature.replace('_', ' ').capitalize()}", step=1.0)

if st.button("Check Eligibility"):
    try:
        input_df = pd.DataFrame([user_data])

        # Replace missing with string for consistency
        input_df = input_df.fillna("Missing")

        # Encode categorical
        for col in input_df.columns:
            if col in encoder.feature_names_in_:
                input_df[col] = input_df[col].astype(str)

        encoded = encoder.transform(input_df[encoder.feature_names_in_])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

        # Combine with numerical (if any)
        numeric_cols = [col for col in input_df.columns if col not in encoder.feature_names_in_]
        final_input = pd.concat([input_df[numeric_cols].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

        # Add any missing features
        for col in all_features:
            if col not in final_input.columns:
                final_input[col] = 0

        final_input = final_input[all_features]  # Reorder

        # Prediction
        prediction = model.predict(final_input)[0]
        prob = model.predict_proba(final_input)[0][1]

        if prediction == 1:
            st.success(f"✅ Eligible for claim (Fraud suspected with {prob:.2%} probability)")
        else:
            st.info(f"❌ Not eligible for claim (Fraud unlikely with {prob:.2%} probability)")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
