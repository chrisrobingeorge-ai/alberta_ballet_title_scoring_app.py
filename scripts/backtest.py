"""
Backtest scaffold (lightweight). This just verifies the data loads and
the forecast API returns the expected keys. No heavy ML required.
"""

import importlib
import pandas as pd

def run_backtest(data_path: str):
    df = pd.read_csv(data_path)
    assert {"show_title", "single_tickets_calgary", "single_tickets_edmonton"}.issubset(df.columns)

    svc = importlib.import_module("service.forecast")
    predict = getattr(svc, "predict")
    sample_title = str(df["show_title"].iloc[0])
    out = predict(sample_title, "YYC", "2025-10-17T19:30:00-06:00")

    required = {"point", "interval", "drivers"}
    assert required.issubset(out.keys()), "forecast output missing required keys"

    print("Backtest scaffold OK")

if __name__ == "__main__":
    run_backtest("data/productions/history_city_sales.csv")
