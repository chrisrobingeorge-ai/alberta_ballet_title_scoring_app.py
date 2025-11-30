import pandas as pd
from data.loader import load_history_sales, load_past_runs
from data.features import derive_basic_features, get_feature_list, apply_registry_renames
from data.leakage import filter_leakage
from utils.canonicalize_titles import canonicalize_title, fuzzy_match_title

TARGET_COL = "total_single_tickets"  # simple baseline target

# Canonical date column name used for time-aware cross-validation.
# This column is of dtype datetime64[ns] and represents the run end date
# for each production. Training code and CV splitters should use this
# column name to discover the date field.
DATE_COL = "end_date"


def _merge_with_past_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Merge history sales data with past_runs to add date columns.

    Joins on canonicalized title to handle minor title variations
    (e.g., em-dash vs hyphen). Uses fuzzy matching as a fallback for
    titles that don't match exactly after canonicalization.

    Args:
        df: DataFrame from load_history_sales() with 'show_title' column.

    Returns:
        DataFrame with start_date and end_date columns added.
        Rows without a matching past_run will have NaT for date columns.
    """
    past_runs = load_past_runs()
    if past_runs.empty:
        # No past_runs data available - return as is
        df = df.copy()
        df[DATE_COL] = pd.NaT
        df["start_date"] = pd.NaT
        return df

    # Create canonical title columns for matching
    df = df.copy()
    df["_canonical_title"] = df["show_title"].apply(canonicalize_title)

    past_runs = past_runs.copy()
    past_runs["_canonical_title"] = past_runs["title"].apply(canonicalize_title)

    # Build a lookup dict from canonical title to (start_date, end_date)
    # Use first occurrence for each canonical title
    past_runs_lookup = {
        row["_canonical_title"]: (row["start_date"], row["end_date"])
        for row in past_runs.iloc[::-1].to_dict("records")
    }

    # Match each history row to past_runs
    start_dates = []
    end_dates = []
    past_runs_titles = past_runs["title"].tolist()

    for _, row in df.iterrows():
        canonical = row["_canonical_title"]
        if canonical in past_runs_lookup:
            start_dates.append(past_runs_lookup[canonical][0])
            end_dates.append(past_runs_lookup[canonical][1])
        else:
            # Try fuzzy matching as fallback
            match = fuzzy_match_title(row["show_title"], past_runs_titles, threshold=80)
            if match:
                match_canonical = canonicalize_title(match)
                if match_canonical in past_runs_lookup:
                    start_dates.append(past_runs_lookup[match_canonical][0])
                    end_dates.append(past_runs_lookup[match_canonical][1])
                else:
                    start_dates.append(pd.NaT)
                    end_dates.append(pd.NaT)
            else:
                start_dates.append(pd.NaT)
                end_dates.append(pd.NaT)

    df["start_date"] = start_dates
    df["end_date"] = end_dates

    # Clean up temporary column
    df = df.drop(columns=["_canonical_title"])

    # Ensure date columns are datetime64[ns]
    for col in ["start_date", "end_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


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
    """Build a dataset for training or scoring.

    The returned DataFrame X includes a DATE_COL ('end_date') column of dtype
    datetime64[ns] suitable for time-aware cross-validation. This date column
    is merged from past_runs.csv based on show title matching.

    Args:
        theme_filters: Optional list of themes to filter features by.
        status: Optional status filter for features.
        forecast_time_only: If True, apply leakage filtering for forecast time.

    Returns:
        Tuple of (X, y) where:
        - X: Feature DataFrame including DATE_COL for time-based CV.
        - y: Target Series (total_single_tickets).
    """
    raw = load_history_sales()

    # Merge with past_runs to add date columns (start_date, end_date)
    raw = _merge_with_past_runs(raw)

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

    # Add the date column for time-aware CV (this is not a feature for prediction)
    # The DATE_COL is kept separate from ML features but included for CV splitting
    if DATE_COL in fe.columns:
        X[DATE_COL] = fe[DATE_COL].values

    # Basic target for a baseline model (city-agnostic total single tickets)
    if TARGET_COL not in fe.columns:
        raise ValueError(f"Target '{TARGET_COL}' not found in dataset.")
    y = fe[TARGET_COL].copy()
    # Drop rows with missing target
    mask = y.notna()
    return X.loc[mask], y.loc[mask]
