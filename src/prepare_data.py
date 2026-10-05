import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# -----------------------------------------
# 1. Load processed dataset
# -----------------------------------------

DATA_PATH = "../data/processed/processed_data.csv"

df = pd.read_csv(DATA_PATH)

print("Processed dataset loaded.")
print("Dataset shape:", df.shape)


# -----------------------------------------
# 2. Select features and target
# -----------------------------------------

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features]
y = df["Machine failure"]


# -----------------------------------------
# 3. Train-Test Split
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------------------
# 4. Feature Scaling
# -----------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# -----------------------------------------
# 5. Display information
# -----------------------------------------

print("\n========================================")
print("DATA PREPARATION COMPLETED")
print("========================================")

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining failure distribution:")
print(y_train.value_counts())

print("\nTesting failure distribution:")
print(y_test.value_counts())

print("\nScaled training data shape:")
print(X_train_scaled.shape)

print("\nScaled testing data shape:")
print(X_test_scaled.shape)

print("\nFeature scaling completed successfully.")