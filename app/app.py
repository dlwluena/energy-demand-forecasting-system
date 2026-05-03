import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from utils import load_and_clean, calculate_metrics, add_error_columns, peak_error_analysis

# ---------------------------
# TITLE & DESCRIPTION
# ---------------------------
st.title("Energy Demand Forecast")

st.write(
    "This dashboard predicts electricity demand using machine learning and time series analysis."
)

# ---------------------------
# LOAD DATA
# ---------------------------
df_full = pd.read_csv("outputs/predictions.csv", index_col=0)
df_full.index = pd.to_datetime(df_full.index)

df_full = df_full.sort_index()  
df_full = df_full.rename(columns={"PJME_MW": "actual"})

pred_col = "xgb"

df = df_full.copy()

# ---------------------------
# DATA RANGE SELECTOR
# ---------------------------
option = st.selectbox(
    "Select Data Range",
    ("Last 100", "Last 500", "Last 1000"),
    key="data_range"
)

if option == "Last 100":
    df = df.tail(100)
elif option == "Last 500":
    df = df.tail(500)
else:
    df = df.tail(1000)
    
model_option = st.selectbox(
    "Select Model",
    ("XGBoost", "Random Forest", "Linear"),
    key="model_select"
)
# --------------------------
if model_option == "XGBoost":
    pred_col = "xgb"
elif model_option == "Random Forest":
    pred_col = "rf"
else:
    pred_col = "linear"

df = add_error_columns(df, pred_col)
df["abs_error"] = np.abs(df["error"])

df["hour"] = df.index.hour
# ---------------------------
# METRICS
# ---------------------------
mae, rmse, r2, mape = calculate_metrics(df_full, pred_col)
col1, col2, col3, col4 = st.columns(4)

col1.metric("MAE", round(mae, 2))
col2.metric("RMSE", round(rmse, 2))
col3.metric("R² Score", round(r2, 3))
col4.metric("MAPE (%)", round(mape, 2))

st.divider()

# ---------------------------
# MAIN GRAPH
# ---------------------------
st.subheader("Actual vs Predicted")

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(df.index, df["actual"], label="Actual")
ax.plot(df.index, df[pred_col], label=model_option)
ax.legend()
ax.set_title("Energy Demand Prediction")
ax.set_xlabel("Datetime")
ax.set_ylabel("Demand (MW)")

st.pyplot(fig)

st.subheader("Model Comparison")

fig_compare, ax_compare = plt.subplots(figsize=(12,5))
ax_compare.plot(df.index, df["actual"], label="Actual")
ax_compare.plot(df.index, df["xgb"], label="XGBoost")
ax_compare.plot(df.index, df["rf"], label="Random Forest")
ax_compare.plot(df.index, df["linear"], label="Linear")
ax_compare.legend()
ax_compare.set_title("Comparison of All Models")
ax_compare.set_xlabel("Datetime")
ax_compare.set_ylabel("Demand (MW)")

st.pyplot(fig_compare)

metrics_df = pd.DataFrame({
    "Model": ["XGBoost", "Random Forest", "Linear"],
    "MAE": [
        np.mean(np.abs(df_full["actual"] - df_full["xgb"])),
        np.mean(np.abs(df_full["actual"] - df_full["rf"])),
        np.mean(np.abs(df_full["actual"] - df_full["linear"]))
    ],
    "RMSE": [
        np.sqrt(np.mean((df_full["actual"] - df_full["xgb"])**2)),
        np.sqrt(np.mean((df_full["actual"] - df_full["rf"])**2)),
        np.sqrt(np.mean((df_full["actual"] - df_full["linear"])**2))
    ]
})

best_model = metrics_df.sort_values("MAE").iloc[0]
st.success(f"Best model: {best_model['Model']}")

st.caption("Lower MAE and RMSE indicate better model performance.")


st.dataframe(metrics_df)

# ---------------------------
# ERROR OVER TIME
# ---------------------------
st.subheader("Error Over Time")

fig2, ax2 = plt.subplots(figsize=(12,4))
ax2.plot(df.index, df["error"])
ax2.set_title("Prediction Error Over Time")
ax2.set_xlabel("Datetime")
ax2.set_ylabel("Error")

st.pyplot(fig2)

# ---------------------------
# ERROR HISTOGRAM
# ---------------------------
st.subheader("Error Histogram")

fig3, ax3 = plt.subplots()
sns.histplot(df["error"], bins=50, kde=True, ax=ax3)

ax3.set_title("Distribution of Prediction Errors (Actual - Predicted)")
ax3.set_xlabel("Error")
ax3.set_ylabel("Frequency")

st.pyplot(fig3)

# ---------------------------
# ERROR BY HOUR
# ---------------------------
st.subheader("Error by Hour")

hour_error = df.groupby("hour")["abs_error"].mean()

worst_hour = hour_error.idxmax()
peak_hours = hour_error.sort_values(ascending=False).head(3).index.tolist()


fig_hour, ax_hour = plt.subplots(figsize=(10,4))
ax_hour.bar(hour_error.index, hour_error.values)

ax_hour.set_title("Average Error by Hour")
ax_hour.set_xlabel("Hour of Day")
ax_hour.set_ylabel("Average Absolute Error")

st.pyplot(fig_hour)

st.warning(f"Highest prediction error occurs around hour: {worst_hour}:00")

st.write(f"Top 3 worst hours: {peak_hours}")

if worst_hour in range(15, 20):
    st.info("High errors occur during evening peak demand hours.")
elif worst_hour in range(6, 10):
    st.info("Morning demand transitions are difficult to predict.")
else:
    st.info("Errors are relatively evenly distributed throughout the day.")

st.write(f"Average hourly error: {round(hour_error.mean(), 2)}")

## ---------------------------
# ERROR BY DAY OF WEEK
# ---------------------------
st.subheader("Error by Day of Week")

df["day"] = df.index.day_name()

day_error = df.groupby("day")["abs_error"].mean()


days_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
day_error = day_error.reindex(days_order)


day_error = day_error.dropna()


fig_day, ax_day = plt.subplots(figsize=(10,4))
ax_day.bar(day_error.index, day_error.values)

ax_day.set_title("Average Error by Day")
ax_day.set_xlabel("Day")
ax_day.set_ylabel("Average Absolute Error")

st.pyplot(fig_day)


worst_day = day_error.idxmax()

st.warning(f"Highest error occurs on: {worst_day}")


best_day = day_error.idxmin()
st.info(f"Lowest error occurs on: {best_day}")
# ---------------------------
# ERROR SEVERITY PIE
# ---------------------------
st.subheader("Error Severity Distribution")

error_abs = df["abs_error"]

bins = [
    (error_abs < 1000).sum(),
    ((error_abs >= 1000) & (error_abs < 3000)).sum(),
    (error_abs >= 3000).sum()
]

labels = ["Low Error", "Medium Error", "High Error"]

fig4, ax4 = plt.subplots()
ax4.pie(bins, labels=labels, autopct="%1.1f%%")

st.pyplot(fig4)

# ---------------------------
# CORRELATION HEATMAP
# ---------------------------
st.subheader("Correlation Analysis")

corr = df[["actual", "xgb", "rf", "linear"]].corr()

fig5, ax5 = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax5)

st.pyplot(fig5)

# ---------------------------
# WORST PREDICTIONS
# ---------------------------
st.subheader("Worst Predictions")

worst = df.sort_values("abs_error", ascending=False).head(10)
st.write(worst)

st.write("Average error of worst cases:", round(worst["abs_error"].mean(), 2))

# ---------------------------
# ROLLING COMPARISON
# ---------------------------
st.subheader("Rolling Average Comparison")

df["rolling_actual"] = df["actual"].rolling(24).mean()
df["rolling_pred"] = df[pred_col].rolling(24).mean()

fig6, ax6 = plt.subplots(figsize=(12,5))
ax6.plot(df.index, df["rolling_actual"], label="Actual Rolling")
ax6.plot(df.index, df["rolling_pred"], label="Predicted Rolling")
ax6.legend()
ax6.set_title("24-Hour Rolling Average Comparison")
ax6.set_xlabel("Datetime")
ax6.set_ylabel("Demand (MW)")

st.pyplot(fig6)
# --------------------------

st.subheader("Peak Demand Error Analysis")

peak_mae, peak_df = peak_error_analysis(df, pred_col)

col1, col2 = st.columns(2)

col1.metric("Overall MAE", round(mae, 2))
col2.metric("Peak MAE", round(peak_mae, 2))

st.write("Peak demand samples:")
st.write(peak_df.head(10))

increase = ((peak_mae - mae) / mae) * 100
st.write(f"Peak error is {round(increase, 1)}% higher than average error.")

# ---------------------------
# INSIGHTS
# ---------------------------
if peak_mae > mae:
    st.warning("Model performs worse during peak demand periods ⚠️")
else:
    st.success("Model handles peak demand well ")
    
st.subheader("Peak Demand Predictions")

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(peak_df.index, peak_df["actual"], label="Actual")
ax.plot(peak_df.index, peak_df[pred_col], label=model_option)
ax.legend()

st.pyplot(fig)

st.subheader("Model Insights")

# ---------------------------
# ACCURACY
# ---------------------------
if mape < 5:
    st.success("Model has high accuracy.")
elif mape < 10:
    st.warning("Model has moderate accuracy.")
else:
    st.error("Model has low accuracy.")

# ---------------------------
# PEAK PERFORMANCE
# ---------------------------
if peak_mae > mae * 2:
    st.warning("Model performs significantly worse during peak demand periods.")
else:
    st.success("Model handles peak demand reasonably well.")

# ---------------------------
# ERROR TIMING
# ---------------------------
st.info(f"Highest error occurs around {worst_hour}:00 and on {worst_day}.")

if worst_hour in [7,8,9,17,18,19]:
    st.warning("Demand transitions during rush hours are harder to predict.")

# ---------------------------
# MODEL COMPARISON
# ---------------------------
best_model = metrics_df.sort_values("MAE").iloc[0]["Model"]
st.success(f"Best performing model: {best_model}")

if best_model == "Random Forest":
    st.info("Random Forest captures nonlinear patterns slightly better in this dataset.")

# ---------------------------
# CORRELATION INSIGHT
# ---------------------------
st.info("Models show high correlation, indicating similar learned patterns.")

# ---------------------------
# ERROR DISTRIBUTION
# ---------------------------
st.warning("A significant portion of predictions fall into high-error category, mainly during peak demand periods.")

# ---------------------------
# FINAL SUMMARY
# ---------------------------
st.write("""
### Key Insights:
- Model captures general demand trends effectively.
- Errors increase during sudden demand changes.
- Peak demand periods remain challenging.
""")

# ---------------------------
# FUTURE IMPROVEMENTS
# ---------------------------
st.info("""
Adding external features such as weather (temperature, humidity) and holidays
could significantly improve performance, especially during peak demand periods.
""")
# ---------------------------
# EXTRA INFO
# ---------------------------
st.divider()
st.write("Max Demand:", int(df["actual"].max()), "MW")

