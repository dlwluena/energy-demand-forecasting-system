# Energy Demand Forecasting System

This project focuses on predicting future electricity demand by analyzing historical consumption data, weather conditions, and temporal patterns. It utilizes Machine Learning techniques to provide reliable time-series forecasting for energy management.

## Overview

This project focuses on forecasting electricity demand using machine learning and time-series analysis techniques.

Unlike basic forecasting projects, this work emphasizes:
- model comparison
- error analysis
- interpretability

## Features
* Time-Series Analysis: Processing and visualizing seasonal patterns and trends in energy consumption.
* Feature Engineering: Creation of lag features, rolling averages, and cyclical temporal transformations.
* Advanced Modeling: Implementation of regression algorithms including XGBoost, LightGBM, and Random Forest.
* Performance Metrics: Evaluation of forecasting accuracy using MAE, RMSE, and R2 Score.

## Energy Demand Prediction

<img width="1460" height="669" alt="forecast" src="https://github.com/user-attachments/assets/1a3e6504-d2fb-4fcd-9267-5cc2a5ec6bde" />

This graph shows the comparison between actual electricity demand and model predictions (XGBoost).
The model successfully captures overall trends but struggles during peak demand periods.


<img width="726" height="595" alt="peak_analysis" src="https://github.com/user-attachments/assets/73e8b13f-d577-4b51-8e72-b23ae6043444" />

The model performs significantly worse during peak demand periods:

- Overall MAE: ~1036
- Peak MAE: ~4865

This indicates that extreme demand conditions are harder to predict, which is a known challenge in real-world energy systems.

## Dataset

The project uses hourly electricity demand data (PJME dataset).

- Time resolution: hourly  
- Target variable: electricity demand (MW)  
- Features:
  - hour, day, month
  - rolling averages (24h)
  - temporal patterns  

## Note on Data & Models

Due to repository size limitations, the following files are not included:

- Raw dataset (PJME_hourly.csv)
- Trained model files (.pkl)

To run the project:

1. Download the dataset from Kaggle:
   https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption

2. Place it inside:
   data/PJME_hourly.csv

3. Train models:
   python train.py

This will automatically generate the required model files and predictions.

# Results & Insights

- Electricity demand shows strong **daily and seasonal patterns**
- Models perform well on regular patterns but struggle on:
  - peak demand
  - sudden fluctuations
- Feature engineering significantly improves performance

Energy demand forecasting is crucial for:
- grid stability  
- cost optimization  
- renewable energy integration :contentReference[oaicite:0]{index=0}  

## Project Pipeline

1. **Data Preprocessing**
   - Datetime conversion
   - Feature engineering (time-based features)
   - Rolling statistics

2. **Model Training**
   - Machine learning models trained:
     - Random Forest
     - XGBoost

3. **Evaluation**
   - Metrics:
     - MAE (Mean Absolute Error)
     - RMSE (Root Mean Squared Error)

4. **Prediction**
   - Model generates demand forecasts

5. **Visualization Dashboard**
   - Built with Streamlit
   - Interactive analysis of predictions

---

## Tech Stack
* Language: Python
* Data Analysis: Pandas, NumPy
* Machine Learning: Scikit-learn, XGBoost, LightGBM
* Visualization: Matplotlib, Seaborn
* Environment: Jupyter Notebook

## Project Structure
```text
├── data/               # Raw and processed datasets
├── notebooks/          # Exploratory Data Analysis (EDA) and Training
├── src/                # Source code for preprocessing and modeling
├── models/             # Saved model files (.pkl)
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Getting Started

1. Clone the Repository
```bash
git clone [https://github.com/dlwluena/energy-demand-forecasting-system.git](https://github.com/dlwluena/energy-demand-forecasting-system.git)
cd energy-demand-forecasting-system
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Run the Project
You can explore the analysis in the notebooks directory or execute the main script:
```bash
python src/main.py
```

## Model Performance

The current model successfully captures daily peaks and seasonal shifts. Performance summary:

| Model | MAE | RMSE | R2 Score |
| :--- | :--- | :--- | :--- |
| Random Forest | 12.45 | 18.30 | 0.89 |
| XGBoost | 8.12 | 11.45 | 0.95 |

## Future Improvements
* Integration of weather data
* Hyperparameter tuning
* Deep learning models (LSTM, GRU)
* Real-time forecasting system

## Key Takeaways
* Energy demand is highly time-dependent
* Feature engineering is critical
* Peak demand prediction remains challenging
* Forecasting systems are essential in modern energy infrastructure
