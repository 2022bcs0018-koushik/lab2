import pandas as pd
import json, os, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = "dataset/winequality-white.csv"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(DATA_PATH, sep=";")
X = df.drop("quality", axis=1)
y = df["quality"]

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

joblib.dump(model, f"{OUT}/model.pkl")

json.dump(
    {"model": "Ridge", "alpha": 1.0, "MSE": mse, "R2": r2},
    open(f"{OUT}/results.json", "w"),
    indent=4
)

print(f"MSE: {mse}")
print(f"R2: {r2}")
