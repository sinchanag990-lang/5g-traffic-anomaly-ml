import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="5G Traffic Anomaly Detector", layout="wide")
st.title("5G Traffic Anomaly Detector")


@st.cache_resource
def load_model():
  return joblib.load("src/rf_5g_model.joblib")


@st.cache_data
def load_data():
  df = pd.read_csv("data/Combined.csv")
  drop_cols = [c for c in ["Label", "Attack Type", "Attack Tool"] if c in df.columns]
  X = df.drop(columns=drop_cols, errors="ignore")

  for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

  return X.replace([float("inf"), float("-inf")], pd.NA).fillna(0)


model = load_model()
X = load_data()

n_rows = st.slider("Number of sample traffic rows to inspect", 5, 50, 15)
sample = X.sample(n_rows, random_state=42)
preds = model.predict(sample)

results = pd.DataFrame({
    "Row_Index": sample.index,
    "Predicted_Traffic": preds,
    "Status": ["Normal" if p == "Benign" else "Anomaly" for p in preds],
})

col1, col2 = st.columns(2)
col1.metric("Total Sampled Flows", len(results))
col2.metric("Detected Anomalies", int((results["Status"] == "Anomaly").sum()))

st.bar_chart(results["Predicted_Traffic"].value_counts())

traffic_types = ["All"] + sorted(results["Predicted_Traffic"].unique().tolist())
selected_type = st.selectbox("Filter by traffic type", traffic_types)

filtered_results = (
    results
    if selected_type == "All"
    else results[results["Predicted_Traffic"] == selected_type]
)

st.dataframe(filtered_results, use_container_width=True)

st.download_button(
    label="Download results as CSV",
    data=filtered_results.to_csv(index=False).encode("utf-8"),
    file_name="5g_predictions.csv",
    mime="text/csv",
)