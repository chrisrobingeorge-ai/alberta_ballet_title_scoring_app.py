import pandas as pd
from data.leakage import filter_leakage


def test_filter_leakage():
    df = pd.DataFrame({
        "show_title_id": [1, 2],
        "late_sales_share": [0.1, 0.2],
        "city": ["Calgary", "Edmonton"]
    })
    # Allowed set comes from CSV; test should not error and should return subset of columns
    out = filter_leakage(df, forecast_time_only=True)
    assert set(out.columns).issubset(set(df.columns))
