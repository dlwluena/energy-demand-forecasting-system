# Energy Demand Forecasting System

## Overview

This project focuses on predicting **hourly electricity demand** using machine learning and time-series feature engineering.

Unlike basic forecasting implementations, this system is designed with a **real-world perspective**, emphasizing:

- Model comparison  
- Error analysis  
- Interpretability  
- Practical applicability  

It implements a complete end-to-end pipeline:
> Data preprocessing → Feature engineering → Model training → Evaluation → Interactive dashboard

---

## Real-World Impact

Accurate energy demand forecasting is essential for:

- Grid stability and load balancing  
- Renewable energy integration  
- Reducing operational and energy costs  

---

## Dataset

The model is trained on the **PJME hourly electricity demand dataset**.

- Time resolution: Hourly  
- Target: Electricity demand (MW)  
- Includes long-term seasonal and temporal patterns  

---

## Objectives

- Predict electricity demand using historical data  
- Compare multiple machine learning models  
- Analyze model errors and behavior  
- Provide an interactive visualization dashboard  

---

## Feature Engineering

To capture temporal dependencies, several feature groups were designed:

### Time-Based Features
- Hour of day  
- Day of week  
- Month  
- Day of year  

### Lag Features
- `lag_24` → Previous day demand  
- `lag_168` → Previous week demand  

### Rolling Statistics
- `rolling_mean_24` → 24-hour average  
- `rolling_mean_168` → Weekly average  

> These features enable the model to learn seasonality and repeating patterns effectively.

---

## Models Used

| Model             | Role |
|------------------|-----|
| Linear Regression | Baseline model |
| Random Forest     | Captures non-linear relationships |
| XGBoost           | Final high-performance model |

---

## Evaluation Metrics

- MAE (Mean Absolute Error)  
- RMSE (Root Mean Squared Error)  
- R² Score  
- MAPE (Mean Absolute Percentage Error)  

---

## Results & Insights

### Key Observations

- XGBoost delivers the most stable and accurate predictions  
- Random Forest captures short-term fluctuations effectively  
- Linear Regression underperforms due to limited complexity  

---

## Error Analysis

A major focus of this project is understanding **why models fail**, not just how they perform.

### Error Characteristics
- Errors are centered around zero → low bias  
- Occasional large deviations → presence of outliers  

### Temporal Insights
- Higher prediction errors during peak demand hours (evenings)  
- Certain weekdays are more difficult to predict  

### Peak Demand Behavior
- Peak demand shows significantly higher error  
- Indicates missing external features such as:
  - Weather data  
  - Holidays  

---

## Visualization Dashboard

An interactive dashboard was developed using **Streamlit**.

### Features

- Model performance comparison  
- Actual vs predicted demand visualization  
- Error distribution (histogram)  
- Error by hour and day  
- Peak demand analysis  
- Adjustable data range  

---

## Installation

```bash
git clone https://github.com/dlwluena/energy-demand-forecasting-system.git
cd energy-demand-forecasting-system

python3 -m venv .venv
```
---

## Project Structure

energy-demand-forecasting-system/
│
├── app/                # Streamlit dashboard
├── notebooks/          # EDA & experiments
├── train.py            # Training pipeline
├── predict.py          # Prediction pipeline
├── requirements.txt
└── README.md

Note: Large files such as datasets and trained models are excluded for repository efficiency.
---

## Usage
Train the model
python train.py

# Generate predictions
python predict.py

# Run the dashboard
streamlit run app/app.py

---
## Future Improvements
Integrating weather and external data sources
Hyperparameter optimization
Deep learning models (GRU, Transformers)
Real-time forecasting system
---

## Key Takeaways
Electricity demand exhibits strong temporal patterns
Peak demand is significantly harder to predict
Feature engineering is critical for performance
Error analysis provides actionable insights

