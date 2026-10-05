import pandas as pd
import os

# -----------------------------------------
# 1. Load raw dataset
# -----------------------------------------

RAW_DATA_PATH = "../data/raw/ai4i2020.csv"
PROCESSED_DATA_PATH = "../data/processed/processed_data.csv"

df = pd.read_csv(RAW_DATA_PATH)

print("Raw dataset loaded successfully.")
print("Original shape:", df.shape)


# -----------------------------------------
# 2. Select machine sensor features
# -----------------------------------------

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# Create feature dataset
X = df[features].copy()

# Keep machine failure separately for evaluation
y = df["Machine failure"].copy()


# -----------------------------------------
# 3. Create processed dataset
# -----------------------------------------

processed_df = X.copy()

processed_df["Machine failure"] = y


# -----------------------------------------
# 4. Create processed directory
# -----------------------------------------

os.makedirs("../data/processed", exist_ok=True)


# -----------------------------------------
# 5. Save processed dataset
# -----------------------------------------

processed_df.to_csv(PROCESSED_DATA_PATH, index=False)


# -----------------------------------------
# 6. Display results
# -----------------------------------------

print("\n========================================")
print("DATA PREPROCESSING COMPLETED")
print("========================================")

print("\nSelected Features:")
for feature in features:
    print("-", feature)

print("\nProcessed Dataset Shape:")
print(processed_df.shape)

print("\nProcessed Dataset:")
print(processed_df.head())

print("\nSaved file:")
print(PROCESSED_DATA_PATH)