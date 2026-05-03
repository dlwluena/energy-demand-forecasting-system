# LOGIC

import os
import pandas as pd
import numpy as np

# ---------------------------
# LOAD DATA
# ---------------------------

def load_and_clean(path):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    full_path = os.path.join(base_dir, path)

    df = pd.read_csv(full_path)
    
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime")

    return df
# ----------------------------
# FEATURE LOGIC 
# ----------------------------

def create_features(df):
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["day_of_year"] = df.index.dayofyear

    df["lag_24"] = df["PJME_MW"].shift(24)
    df["lag_168"] = df["PJME_MW"].shift(168)

    df["rolling_mean_24"] = df["PJME_MW"].rolling(24).mean()
    df["rolling_mean_168"] = df["PJME_MW"].rolling(168).mean()

    df = df.dropna()

    return df

# ----------------------------
# PIPELINE
# ----------------------------

def prepare_data(path):
    df = load_and_clean(path)
    df = create_features(df)
    return df

# ---------------------------
# METRICS
# ---------------------------
def calculate_metrics(df, pred_col):
    actual = df["actual"]
    pred = df[pred_col]

    mae = np.mean(np.abs(actual - pred))
    rmse = np.sqrt(np.mean((actual - pred)**2))
    r2 = 1 - (np.sum((actual - pred)**2) / 
              np.sum((actual - np.mean(actual))**2))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100

    return mae, rmse, r2, mape


# ---------------------------
# ERROR ANALYSIS
# ---------------------------
def add_error_columns(df, pred_col):
    df["error"] = df["actual"] - df[pred_col]
    df["abs_error"] = np.abs(df["error"])
    return df


# ---------------------------
# PEAK ERROR ANALYSIS
# ---------------------------
def peak_error_analysis(df, pred_col, threshold_percent=90):
    threshold = np.percentile(df["actual"], threshold_percent)
    peak_df = df[df["actual"] >= threshold]

    peak_mae = np.mean(np.abs(peak_df["actual"] - peak_df[pred_col]))
    return peak_mae, peak_df


