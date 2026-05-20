"""
Lightweight training script to fit and save a LinearRegression forecasting model.
Run: python scripts/train_model.py
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "Sample - Superstore.csv"
MODEL_PATH = BASE / "models" / "forecasting_model.pkl"


def prepare_monthly(df):
    df = df.copy()
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Order Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("Order Month")["Sales"].sum().reset_index()
    monthly = monthly.sort_values("Order Month")
    monthly["month_index"] = np.arange(len(monthly))
    return monthly


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["Order Date"]) 
    monthly = prepare_monthly(df)
    X = monthly[["month_index"]].values
    y = monthly["Sales"].values
    model = LinearRegression()
    model.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print("Saved model to", MODEL_PATH)


if __name__ == "__main__":
    main()
