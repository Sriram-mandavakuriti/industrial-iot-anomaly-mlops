import pandas as pd

# Load dataset
DATA_PATH = "../data/raw/ai4i2020.csv"
df = pd.read_csv(DATA_PATH)

print("========================================")
print("DATASET OVERVIEW")
print("========================================")

# 1. Dataset shape
print("\n1. Dataset Shape:")
print(df.shape)

# 2. Column names
print("\n2. Columns:")
for column in df.columns:
    print(column)

# 3. Data types
print("\n3. Data Types:")
print(df.dtypes)

# 4. Missing values
print("\n4. Missing Values:")
print(df.isnull().sum())

# 5. Statistical summary
print("\n5. Statistical Summary:")
print(df.describe())

# 6. Machine failure distribution
print("\n6. Machine Failure Distribution:")
print(df["Machine failure"].value_counts())

# 7. Machine failure percentage
print("\n7. Machine Failure Percentage:")
print(df["Machine failure"].value_counts(normalize=True) * 100)