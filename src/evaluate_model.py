import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# =========================================
# 1. Load processed dataset
# =========================================

DATA_PATH = "../data/processed/processed_data.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")


# =========================================
# 2. Select features and target
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
# 4. Load scaler
# =========================================

scaler = joblib.load("../models/scaler.pkl")

X_test_scaled = scaler.transform(X_test)


# =========================================
# 5. Load trained model
# =========================================

model = joblib.load("../models/isolation_forest.pkl")

print("Model loaded successfully.")


# =========================================
# 6. Generate predictions
# =========================================

predictions = model.predict(X_test_scaled)

# Isolation Forest:
#  1  = Normal
# -1  = Anomaly

anomaly_predictions = (predictions == -1).astype(int)


# =========================================
# 7. Create results dataframe
# =========================================

results = X_test.copy()

results["Actual Failure"] = y_test.values
results["Predicted Anomaly"] = anomaly_predictions

results["Correct Prediction"] = (
    results["Actual Failure"] ==
    results["Predicted Anomaly"]
)


# =========================================
# 8. Create reports directory
# =========================================

os.makedirs("../reports", exist_ok=True)


# =========================================
# 9. Save prediction results
# =========================================

results.to_csv(
    "../reports/model_predictions.csv",
    index=False
)


# =========================================
# 10. Classification Report
# =========================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

report = classification_report(
    y_test,
    anomaly_predictions,
    target_names=["Normal", "Failure"]
)

print(report)


# =========================================
# 11. Confusion Matrix
# =========================================

cm = confusion_matrix(
    y_test,
    anomaly_predictions
)

print("========================================")
print("CONFUSION MATRIX")
print("========================================")

print(cm)


# =========================================
# 12. Save confusion matrix image
# =========================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Failure"]
)

display.plot()

plt.title("Isolation Forest - Confusion Matrix")

plt.savefig(
    "../reports/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =========================================
# 13. Summary
# =========================================

print("\n========================================")
print("EVALUATION COMPLETED")
print("========================================")

print("Total test samples:", len(y_test))

print("Actual failures:", y_test.sum())

print("Detected anomalies:", anomaly_predictions.sum())

print("\nSaved files:")

print("../reports/model_predictions.csv")

print("../reports/confusion_matrix.png")