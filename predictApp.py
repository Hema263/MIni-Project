import streamlit as st
import pandas as pd
import joblib

st.title("🔥 Student Burnout Prediction App")

# Load trained model, scaler, and features
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    selected_features = joblib.load("selected_features.pkl")
except Exception as e:
    st.error(f"❌ Could not load model/scaler/features: {e}")
    st.stop()

# Collect user input dynamically based on selected features
user_input = {}
st.write("Please enter your details below:")

for feature in selected_features:
    user_input[feature] = st.text_input(f"{feature}")

if st.button("Predict Burnout Level"):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([user_input])

        # Convert all values to numeric
        input_df = input_df.apply(pd.to_numeric, errors='coerce')

        # Fill missing values
        input_df = input_df.fillna(0)

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        st.success(f"✅ Predicted Burnout Level: {prediction}")

    except Exception as e:
        st.error(f"⚠️ Prediction could not be made: {e}")
