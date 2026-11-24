import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load dataset
df = pd.read_csv("data/Combined.csv")

# 2. Drop non-feature or ID/label-leaking text columns if present
drop_cols = [c for c in ["Label", "Attack Type", "Attack Tool"] if c in df.columns]
target_col = "Attack Tool" if "Attack Tool" in df.columns else df.columns[-1]

X = df.drop(columns=drop_cols, errors="ignore")
y = df[target_col].astype(str)

# 3. Convert any remaining text columns to numbers
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# 4. Replace infinite or missing values
X = X.replace([float("inf"), float("-inf")], pd.NA).fillna(0)

# 5. Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Train a Random Forest model (n_jobs=-1 uses available CPU cores)
model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 7. Evaluate on unseen test data
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Save trained model
joblib.dump(model, "src/rf_5g_model.joblib")
print("Saved model to src/rf_5g_model.joblib")