import pandas as pd

# Path to our dataset
DATA_PATH = "../data/raw/ai4i2020.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Display basic information
print("========================================")
print("AI4I 2020 DATASET LOADED SUCCESSFULLY")
print("========================================")

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())