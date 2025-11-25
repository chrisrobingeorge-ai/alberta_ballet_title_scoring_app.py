from pathlib import Path
import pandas as pd
import joblib


def load_model(model_path: str | None = None):
    """Load a trained model from disk."""
    path = Path(model_path or (Path(__file__).parent.parent / "models" / "title_demand_rf.pkl"))
    return joblib.load(path)


def score_dataframe(df_features: pd.DataFrame, model=None) -> pd.Series:
    """Score a DataFrame using the trained model."""
    model = model or load_model()
    return pd.Series(model.predict(df_features), index=df_features.index, name="forecast_single_tickets")
