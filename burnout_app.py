import pandas as pd
import numpy as np
import joblib
import streamlit as st

# -------------------------------
# Load trained model and scaler
# -------------------------------
rf_model = joblib.load("burnout_model.joblib")   # trained Random Forest
scaler = joblib.load("scaler.joblib")            # StandardScaler

# LabelEncoder for target classes
from sklearn.preprocessing import LabelEncoder
le_y = LabelEncoder()
le_y.classes_ = np.array(["Low", "Medium", "High"])  # same order as training

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Student Burnout Predictor", layout="centered")
st.title("🎓 Student Burnout Level Predictor")
st.write("Predict your burnout level based on study, sleep, stress, CGPA, and attendance.")

# -------------------------------
# User Inputs
# -------------------------------
study_hours = st.selectbox("📚 Study Hours", ["<2 hours", "2-4 hours", "4-6 hours", "6+ hours"])
sleep_hours = st.selectbox("💤 Sleep Hours", ["<4 hours", "4-6 hours", "6-8 hours", "8+ hours"])
stress = st.selectbox("😰 Stress Level", ["Low", "Medium", "High"])
cgpa = st.number_input("🎯 CGPA (Out of 10)", min_value=0.0, max_value=10.0, step=0.1)
attendance = st.slider("📅 Attendance Percentage", min_value=0, max_value=100, step=1)

# -------------------------------
# Preprocess Input
# -------------------------------
if st.button("🔮 Predict Burnout Level"):
    # Put inputs into DataFrame
    input_df = pd.DataFrame({
        "CGPA (Out of 10)": [cgpa],
        "Attendance Percentage": [attendance],
        "Study Hours": [study_hours],
        "Sleep Hours": [sleep_hours],
        "Stress": [stress]
    })

    # One-hot encode categorical columns
    categorical_cols = ["Study Hours", "Sleep Hours", "Stress"]
    input_df = pd.get_dummies(input_df, columns=categorical_cols)

    # Match training columns (fix missing columns after encoding)
    # Load original training feature names (saved from training script)
    expected_cols = rf_model.feature_names_in_

    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0  # add missing column

    # Reorder columns to match model
    input_df = input_df[expected_cols]

    # Scale numeric features
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = rf_model.predict(input_scaled)
    burnout_level = le_y.inverse_transform(prediction)[0]

    # Show result
    st.success(f"🔥 Predicted Burnout Level: **{burnout_level}**")
