from app.utils import prepare_data
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import joblib
import os

# ---------------------------
# CREATE MODEL FOLDER
# ---------------------------
os.makedirs("models", exist_ok=True)

# ---------------------------
# LOAD DATA
# ---------------------------
df = prepare_data("data/PJME_hourly.csv")

features = [
    "hour","day_of_week","month","day_of_year",
    "lag_24","lag_168","rolling_mean_24","rolling_mean_168"
]

X = df[features]
y = df["PJME_MW"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ---------------------------
# TRAIN MODELS
# ---------------------------

# XGBoost
xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, "models/xgb_model.pkl")

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, "models/rf_model.pkl")

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
joblib.dump(linear_model, "models/linear_model.pkl")

print("All models trained and saved.")