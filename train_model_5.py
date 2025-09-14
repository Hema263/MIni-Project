import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("Share STUDENT BURNOUT LEVEL PREDICTION (Responses) - Form responses .csv")

print("Columns in dataset:", df.columns.tolist())

# Your desired 5 features (update these to match your dataset exactly)
selected_features = [
    "Study Hours",
    "Sleep Hours",
    "Stress",
    "CGPA(Out of 10)",
    "Attendance Percentage"
]

# Keep only columns that exist
existing_features = [col for col in selected_features if col in df.columns]
if not existing_features:
    raise SystemExit("❌ None of the selected features exist in dataset. Fix the column names.")

# Check target column exists
target_column = "burnout_level"  # change to your dataset's actual target
if target_column not in df.columns:
    raise SystemExit(f"❌ Target column '{target_column}' not found in dataset.")

# Select features and target
X = df[existing_features]
y = df[target_column]

# Convert all to numeric (errors coerced to NaN)
X = X.apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values
X = X.fillna(0)
y = y.fillna(0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric columns
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Save artifacts
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(existing_features, "selected_features.pkl")

print("✅ Model trained and saved successfully with 5 features.")
