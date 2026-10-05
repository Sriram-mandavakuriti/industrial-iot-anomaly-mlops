import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix


# =========================================
# 1. Load processed dataset
# =========================================

DATA_PATH = "../data/processed/processed_data.csv"

df = pd.read_csv(DATA_PATH)

print("Processed dataset loaded.")
print("Dataset shape:", df.shape)


# =========================================
# 2. Select sensor features
# =========================================

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features]
y = df["Machine failure"]


# =========================================
# 3. Train-Test Split
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =========================================
# 4. Feature Scaling
# =========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================================
# 5. Create Isolation Forest Model
# =========================================

model = IsolationForest(
    n_estimators=200,
    contamination=0.034,
    random_state=42,
    n_jobs=-1
)


# =========================================
# 6. Train Model
# =========================================

print("\nTraining Isolation Forest...")

model.fit(X_train_scaled)

print("Model training completed.")


# =========================================
# 7. Generate Predictions
# =========================================

predictions = model.predict(X_test_scaled)


# Isolation Forest:
#  1  = Normal
# -1  = Anomaly

# Convert predictions:
# Normal  -> 0
# Anomaly -> 1

anomaly_predictions = (predictions == -1).astype(int)


# =========================================
# 8. Evaluate Model
# =========================================

print("\n========================================")
print("MODEL EVALUATION")
print("========================================")

print("\nConfusion Matrix:")

cm = confusion_matrix(y_test, anomaly_predictions)

print(cm)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        anomaly_predictions,
        target_names=["Normal", "Failure"]
    )
)


# =========================================
# 9. Count Detected Anomalies
# =========================================

number_of_anomalies = anomaly_predictions.sum()

print("\nDetected anomalies:", number_of_anomalies)

print("Actual machine failures:", y_test.sum())


# =========================================
# 10. Save Model and Scaler
# =========================================

os.makedirs("../models", exist_ok=True)

joblib.dump(model, "../models/isolation_forest.pkl")

joblib.dump(scaler, "../models/scaler.pkl")

print("\nModel saved:")
print("../models/isolation_forest.pkl")

print("\nScaler saved:")
print("../models/scaler.pkl")


print("\n========================================")
print("MODEL TRAINING PIPELINE COMPLETED")
print("========================================")