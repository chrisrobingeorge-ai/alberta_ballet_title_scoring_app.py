from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings

from ml.dataset import build_dataset
from ml.time_splits import chronological_train_test_split, assert_chronological_split

MODELS_DIR = Path(__file__).parent.parent / "models"


def train_baseline_model(save_path: Path | None = None, date_column: str | None = None) -> dict:
    """Train a baseline RandomForest model for title demand forecasting.
    
    Uses chronological train/test split when a date column is available
    to prevent future data from leaking into predictions.
    
    Args:
        save_path: Optional path to save the trained model
        date_column: Name of date column for chronological split (e.g., 'end_date').
                    If None and data has date column, will try to find it.
                    Falls back to random split if no date available.
    
    Returns:
        Dictionary with model path and training metrics
    """
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

    # Try to use chronological split if date column is available
    date_col = date_column
    if date_col is None:
        # Try to find a date column
        for col_name in ["end_date", "start_date", "date", "opening_date", "performance_date"]:
            if col_name in X.columns:
                date_col = col_name
                break
    
    if date_col and date_col in X.columns:
        # Use chronological split to prevent future leakage
        # Combine X and y for splitting, then separate
        combined = X.copy()
        combined["_target"] = y.values
        
        train_df, test_df = chronological_train_test_split(combined, date_col, test_ratio=0.2)
        
        Xtr = train_df.drop(columns=["_target"])
        ytr = train_df["_target"]
        Xte = test_df.drop(columns=["_target"])
        yte = test_df["_target"]
        
        # Verify chronological ordering
        assert_chronological_split(train_df, test_df, date_col)
    else:
        # Fallback to random split with warning
        warnings.warn(
            "No date column found for chronological split. "
            "Using random split which may allow future data leakage. "
            "Consider adding date information to enable time-aware splitting.",
            UserWarning
        )
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit preprocessor on training data only (prevent leakage)
    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xte)

    metrics = {
        "mae": float(mean_absolute_error(yte, preds)),
        "r2": float(r2_score(yte, preds)),
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "features_used": X.columns.tolist(),
        "time_aware_split": date_col is not None and date_col in X.columns,
    }

    MODELS_DIR.mkdir(exist_ok=True)
    out_path = save_path or (MODELS_DIR / "title_demand_rf.pkl")
    joblib.dump(pipe, out_path)

    return {"model_path": str(out_path), "metrics": metrics}
