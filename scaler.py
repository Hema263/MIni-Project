# scaler.py
import pandas as pd
import joblib
import os

# 1) Load artifacts saved by train_select5.py
for fname in ("model.pkl","scaler.pkl","feature_names.pkl","selected_original_features.pkl","numeric_cols.pkl","label_encoder.pkl"):
    if not os.path.exists(fname):
        raise SystemExit(f"Missing required file: {fname}. Run train_select5.py first.")

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")                    # encoded column names expected by model
selected_original_features = joblib.load("selected_original_features.pkl")  # list of 5 original fields
numeric_cols = joblib.load("numeric_cols.pkl")
label_encoder = joblib.load("label_encoder.pkl")

print("Model expects these ORIGINAL input fields (5):", selected_original_features)
print("Model expects these ENCODED columns:", feature_names)

def predict_from_raw(input_dict):
    """
    input_dict should provide the 5 original feature keys and their values, for example:
    { 'study_hours': '0-2 hrs', 'sleep_hours': '5-6 hrs', 'activity_level': 'low', ... }
    """
    # ensure all keys exist (fill missing with "MISSING")
    row = {}
    for k in selected_original_features:
        row[k] = input_dict.get(k, "MISSING")   # missing values become the string "MISSING"
    input_df = pd.DataFrame([row])

    # encode the 5 original features the same way we trained (get_dummies)
    input_enc = pd.get_dummies(input_df, drop_first=True)

    # align columns to training
    input_enc = input_enc.reindex(columns=feature_names, fill_value=0)

    # scale numeric columns if any
    if numeric_cols:
        input_enc[numeric_cols] = scaler.transform(input_enc[numeric_cols])

    pred_enc = model.predict(input_enc)
    pred_label = label_encoder.inverse_transform(pred_enc)
    return pred_label[0]

# -------------------------
# Example: predict using first dataset row (helpful to verify)
# -------------------------
CSV_PATH = "Share STUDENT BURNOUT LEVEL PREDICTION (Responses) - Form responses .csv"
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    example_row = df[selected_original_features].iloc[0].to_dict()
    print("\nExample raw input (first dataset row):", example_row)
    print("Prediction for that row ->", predict_from_raw(example_row))
else:
    print(f"\nCSV file {CSV_PATH} not found here. To test, call predict_from_raw() with a dict for the 5 fields.")
