# Energy Demand Forecasting System

## Overview

This project predicts hourly electricity demand using machine learning and time-series feature engineering.

Unlike basic forecasting implementations, this project emphasizes:

* Model comparison
* Error analysis
* Interpretability
* Real-world applicability

It includes a full end-to-end pipeline: data preprocessing, feature engineering, model training, evaluation, and an interactive dashboard.

---

## Real-World Impact

Accurate energy demand forecasting is critical for:

* Grid stability and planning
* Renewable energy integration
* Reducing operational costs

---

## Dataset

The dataset is based on the PJME hourly electricity demand data.

* Time resolution: hourly
* Target variable: electricity demand (MW)
* Contains long-term temporal patterns

---

## Objectives

* Predict electricity demand using historical time-series data
* Compare multiple machine learning models
* Analyze prediction errors and model behavior
* Provide an interactive visualization dashboard

---

## Feature Engineering

To capture temporal dependencies, the following features were created:

### Time-Based Features

* Hour of day
* Day of week
* Month
* Day of year

### Lag Features

* `lag_24` → previous day demand
* `lag_168` → previous week demand

### Rolling Statistics

* `rolling_mean_24` → 24-hour average
* `rolling_mean_168` → weekly average

These features allow the model to learn repeating patterns and seasonality.

---

## Models Used

| Model             | Description                               |
| ----------------- | ----------------------------------------- |
| Linear Regression | Baseline model                            |
| Random Forest     | Captures non-linear relationships         |
| XGBoost           | Gradient boosting (final model)           |

---

## Evaluation Metrics

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* R² Score
* MAPE (Mean Absolute Percentage Error)

---

## Results & Analysis

### Key Observations

* XGBoost provides stable and strong performance across time
* Random Forest performs well on short-term fluctuations
* Linear Regression underperforms due to limited complexity

---

## Error Analysis

A major focus of this project is understanding model behavior:

### Error Characteristics

* Errors are centered around zero → low bias
* Occasional large deviations → outliers

### Time-Based Insights

* Higher errors during peak demand hours (evening)
* Certain weekdays are harder to predict

### Peak Demand Behavior

* Peak demand predictions show significantly higher error
* Indicates missing external features (weather, holidays)

---

## Visualization Dashboard

An interactive dashboard was built using **Streamlit**:

### Features:

* Model performance comparison
* Actual vs predicted visualization
* Error distribution (histogram)
* Error by hour and day
* Peak demand analysis
* Data range selection

---

## Installation

```bash
git clone https://github.com/dlwluena/energy-demand-forecasting-system.git
cd energy-demand-forecasting-system

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

### Train the model

```bash
python train.py
```

### Generate predictions

```bash
python predict.py
```

### Run dashboard

```bash
streamlit run app/app.py
```

---

## Project Structure

```
energy-demand-forecasting-system/
│
├── app/                # Streamlit dashboard
├── data/               # Raw & processed data
├── notebooks/          # EDA & experiments
├── models/             # Saved models
├── outputs/            # Predictions
│
├── train.py            # Training pipeline
├── predict.py          # Prediction pipeline
├── requirements.txt
└── README.md
```

---

## Future Improvements

* Integrating weather and external datasets
* Hyperparameter tuning
* Advanced architectures (GRU, Transformers)
* Real-time forecasting system

---

## Key Takeaways

* Electricity demand shows strong temporal patterns
* Peak demand is significantly harder to predict
* Feature engineering is critical for performance
* Error analysis provides actionable insights

---

## Author

**Handan Vural**
Focus: AI, Time-Series, Energy Systems
