import pandas as pd
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "dataset/winequality-white.csv"
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.pkl")
RESULT_PATH = os.path.join(OUTPUT_DIR, "results.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(DATA_PATH, sep=";")

X = df.drop("quality", axis=1)
y = df["quality"]

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train Best Model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# -----------------------------
# Save Outputs
# -----------------------------
joblib.dump(model, MODEL_PATH)

results = {
    "best_model": "RandomForestRegressor",
    "n_estimators": 200,
    "test_split": 0.2,
    "MSE": mse,
    "R2": r2
}

with open(RESULT_PATH, "w") as f:
    json.dump(results, f, indent=4)

# -----------------------------
# Print Metrics (REQUIRED)
# -----------------------------
print("Best Model: RandomForestRegressor")
print(f"n_estimators: 200")
print(f"MSE: {mse}")
print(f"R2: {r2}")
