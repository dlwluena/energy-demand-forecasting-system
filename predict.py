from app.utils import prepare_data
import joblib
import pandas as pd
import os

# ---------------------------
# CREATE OUTPUT FOLDER
# ---------------------------
os.makedirs("outputs", exist_ok=True)

# ---------------------------
# LOAD DATA
# ---------------------------
df = prepare_data("data/PJME_hourly.csv")

# ---------------------------
# LOAD MODELS
# ---------------------------
xgb_model = joblib.load("models/xgb_model.pkl")
rf_model = joblib.load("models/rf_model.pkl")
linear_model = joblib.load("models/linear_model.pkl")

# ---------------------------
# FEATURES
# ---------------------------
features = [
    "hour","day_of_week","month","day_of_year",
    "lag_24","lag_168","rolling_mean_24","rolling_mean_168"
]

# ---------------------------
# PREDICTIONS
# ---------------------------
df["xgb"] = xgb_model.predict(df[features])
df["rf"] = rf_model.predict(df[features])
df["linear"] = linear_model.predict(df[features])

# ---------------------------
# SAVE OUTPUT
# ---------------------------
results = df[["PJME_MW", "xgb", "rf", "linear"]].rename(
    columns={"PJME_MW": "actual"}
)

results.to_csv("outputs/predictions.csv")

print("Multi-model predictions saved.")