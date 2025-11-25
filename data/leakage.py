import pandas as pd
from config.registry import load_leakage_audit


def filter_leakage(df_X: pd.DataFrame, forecast_time_only: bool = True) -> pd.DataFrame:
    """Keep only features allowed at forecast time if requested."""
    audit = load_leakage_audit()
    if forecast_time_only:
        allowed = set(
            audit[audit["Allowed at Forecast Time (Y/N)"].str.upper() == "Y"]["Feature Name"]
        )
        keep = [c for c in df_X.columns if c in allowed]
        return df_X[keep]
    return df_X
