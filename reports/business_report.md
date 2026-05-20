# Executive Summary

This report summarizes the AI-Powered Retail Forecasting Intelligence Platform built on historical Superstore sales data. The platform provides interactive analytics, forecasting, and automated business insights suitable for executive decision-making.

## Dataset Overview

- Source: `data/Sample - Superstore.csv`
- Records: retail orders with `Order Date`, `Region`, `Category`, `Sales`, `Profit`, etc.
- Granularity: order-level; transformed to monthly aggregates for forecasting.

## Methodology

- Data cleaning and time-series preparation performed in `notebook/sales_forecasting.ipynb`.
- Monthly sales aggregated and used to train a Linear Regression forecasting model.
- Model persisted to `models/forecasting_model.pkl` for reproducible dashboard inference.

## Forecasting Approach

- Baseline model: Linear Regression on sequential month index (robust, interpretable).
- Forecast horizon: configurable in the dashboard (1–24 months).
- Evaluation metrics: MAE, RMSE, R² displayed on the dashboard.

## Business Insights

- Top revenue-generating categories are highlighted with contribution shares.
- Regional strengths and weaknesses are surfaced through ranking visuals.
- Automated insight generation provides quick recommendations for growth or intervention.

## Recommendations

- Consider implementing seasonal models (Prophet) for improved seasonality capture.
- Add promotional campaign variables to improve short-term forecasting.
- Deploy the dashboard for stakeholders and gather feedback for feature prioritization.

## Future Improvements

- Ensemble forecasting (XGBoost + Prophet)
- Model monitoring, retraining pipeline, and unit tests
- Pack into a Docker container and deploy to Streamlit Cloud or AWS
