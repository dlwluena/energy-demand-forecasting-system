# Energy Demand Forecasting System

This project focuses on predicting future electricity demand by analyzing historical consumption data, weather conditions, and temporal patterns. It utilizes Machine Learning techniques to provide reliable time-series forecasting for energy management.

## Features
* Time-Series Analysis: Processing and visualizing seasonal patterns and trends in energy consumption.
* Feature Engineering: Creation of lag features, rolling averages, and cyclical temporal transformations.
* Advanced Modeling: Implementation of regression algorithms including XGBoost, LightGBM, and Random Forest.
* Performance Metrics: Evaluation of forecasting accuracy using MAE, RMSE, and R2 Score.

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
