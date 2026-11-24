import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Load saved model and dataset
model = joblib.load("src/rf_5g_model.joblib")
df = pd.read_csv("data/Combined.csv")

# 2. Prepare features the same way as training
drop_cols = [c for c in ["Label", "Attack Type", "Attack Tool"] if c in df.columns]
X = df.drop(columns=drop_cols, errors="ignore")

for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

X = X.replace([float("inf"), float("-inf")], pd.NA).fillna(0)

# 3. Predict on a sample of 10 rows
sample = X.sample(10, random_state=42)
preds = model.predict(sample)

for idx, pred in zip(sample.index, preds):
    print(f"Row {idx} -> Predicted Traffic: {pred}")