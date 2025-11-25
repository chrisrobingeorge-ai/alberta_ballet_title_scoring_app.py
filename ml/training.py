from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

from ml.dataset import build_dataset

MODELS_DIR = Path(__file__).parent.parent / "models"


def train_baseline_model(save_path: Path | None = None) -> dict:
    """Train a baseline RandomForest model for title demand forecasting."""
    X, y = build_dataset(
        theme_filters=["Production Attributes", "Historical Sales Trends", "Timing & Schedule Factors"],
        status=None,  # include all available + derived when present
        forecast_time_only=True,
    )
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    pipe = Pipeline([("pre", pre), ("rf", model)])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xte)

    metrics = {
        "mae": float(mean_absolute_error(yte, preds)),
        "r2": float(r2_score(yte, preds)),
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "features_used": X.columns.tolist(),
    }

    MODELS_DIR.mkdir(exist_ok=True)
    out_path = save_path or (MODELS_DIR / "title_demand_rf.pkl")
    joblib.dump(pipe, out_path)

    return {"model_path": str(out_path), "metrics": metrics}
