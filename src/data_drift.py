import os
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Get project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REFERENCE_DATA = os.path.join(
    BASE_DIR, "data", "raw", "ai4i2020.csv"
)

REPORT_PATH = os.path.join(
    BASE_DIR, "reports", "data_drift_report.html"
)

# Load dataset
reference = pd.read_csv(REFERENCE_DATA)

# Simulate current/production data
current = reference.sample(frac=0.3, random_state=42).copy()

# Features used by our model
features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

reference = reference[features]
current = current[features]

# Evidently Data Drift Report
report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

result = report.run(
    reference_data=reference,
    current_data=current
)

# Save report
result.save_html(REPORT_PATH)

print("Data Drift analysis completed!")
print("Report saved to:", REPORT_PATH)