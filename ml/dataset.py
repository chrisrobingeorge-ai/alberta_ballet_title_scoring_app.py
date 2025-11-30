import pandas as pd
from data.loader import load_history_sales
from data.features import derive_basic_features, get_feature_list, apply_registry_renames
from data.leakage import filter_leakage

TARGET_COL = "total_single_tickets"  # simple baseline target


def ensure_no_target_in_features(X: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Remove the target column from features to prevent target leakage.

    Args:
        X: Feature DataFrame.
        target_col: Name of the target column to exclude.

    Returns:
        DataFrame with the target column removed if present.
    """
    if target_col in X.columns:
        return X.drop(columns=[target_col])
    return X


def build_dataset(theme_filters=None, status=None, forecast_time_only=True) -> tuple[pd.DataFrame, pd.Series]:
    """Build a dataset for training or scoring."""
    raw = load_history_sales()
    fe = derive_basic_features(apply_registry_renames(raw))

    # Select features from registry
    features = get_feature_list(theme_filters=theme_filters, status=status)
    # Limit to columns present
    features = [f for f in features if f in fe.columns]
    X = fe[features].copy()

    # Enforce leakage policy
    X = filter_leakage(X, forecast_time_only=forecast_time_only)

    # Ensure target column is not used as a feature (prevents target leakage)
    X = ensure_no_target_in_features(X, TARGET_COL)

    # Basic target for a baseline model (city-agnostic total single tickets)
    if TARGET_COL not in fe.columns:
        raise ValueError(f"Target '{TARGET_COL}' not found in dataset.")
    y = fe[TARGET_COL].copy()
    # Drop rows with missing target
    mask = y.notna()
    return X.loc[mask], y.loc[mask]
