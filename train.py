"""Train and compare regression models using a chronological holdout split."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "campus_crowd.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA, parse_dates=["date"])
# Cyclical encoding preserves the fact that 23:00 is close to 00:00.
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

features = [
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "is_weekend", "exam_period", "campus_event", "location"
]
target = "occupancy_count"

# Hold out the latest 20% of dates to mimic predicting future days.
cutoff = df["date"].quantile(0.80)
train_df = df[df["date"] <= cutoff].copy()
test_df = df[df["date"] > cutoff].copy()
X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

numeric_features = [c for c in features if c != "location"]
categorical_features = ["location"]

def make_preprocessor(scale=False):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric = Pipeline(numeric_steps)
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, numeric_features),
        ("cat", categorical, categorical_features),
    ])

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=8, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=3, random_state=42, n_jobs=-1),
    "KNN Regression": KNeighborsRegressor(n_neighbors=9, weights="distance"),
}

results = []
trained = {}
for name, estimator in models.items():
    pipe = Pipeline([
        ("preprocess", make_preprocessor(scale=(name == "KNN Regression"))),
        ("model", estimator),
    ])
    pipe.fit(X_train, y_train)
    pred = np.clip(pipe.predict(X_test), 0, None)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    results.append({
        "Model": name,
        "MAE": round(mean_absolute_error(y_test, pred), 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2_score(y_test, pred), 4),
    })
    trained[name] = pipe

metrics = pd.DataFrame(results).sort_values("RMSE")
metrics.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
best_name = metrics.iloc[0]["Model"]
joblib.dump({"model": trained[best_name], "features": features, "model_name": best_name},
            MODEL_DIR / "best_model.joblib")

with open(REPORT_DIR / "training_summary.json", "w") as f:
    json.dump({
        "best_model": best_name,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_end_date": str(train_df["date"].max().date()),
        "test_start_date": str(test_df["date"].min().date()),
    }, f, indent=2)

print(metrics.to_string(index=False))
print(f"\nBest model: {best_name}")
print(f"Saved model to {MODEL_DIR / 'best_model.joblib'}")
