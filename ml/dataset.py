"""
Legacy/Prototype Dataset Builder Module

.. deprecated::
    This module is part of the **legacy baseline pipeline** and is NOT recommended
    for production use. It has known data leakage risks and limitations.

    **Use the safe modelling dataset pipeline instead:**

    1. Build dataset: ``python scripts/build_modelling_dataset.py``
    2. Train model:   ``python scripts/train_safe_model.py --tune``
    3. Backtest:      ``python scripts/backtest_timeaware.py``

    The safe pipeline (scripts/build_modelling_dataset.py) provides:
    - Explicit leakage prevention with forbidden column assertions
    - Only forecast-time-available features are used
    - Prior-season aggregates computed correctly
    - Comprehensive diagnostics and data quality reports

    This legacy module is retained for backward compatibility and prototyping only.

External Data Join Logic
------------------------
After loading history_city_sales.csv, the dataset builder joins with:

1. **Marketing spend data** (by city and date):
   - Key: 'show_title' and 'start_date' or 'end_date'
   - Source: productions/marketing_spend_per_ticket.csv
   - Join type: LEFT join to preserve all show rows

2. **Weather data** (by city and date):
   - Key: 'city' and 'start_date' (or 'end_date')
   - Source: environment/weatherstats_calgary_daily.csv, weatherstats_edmonton_daily.csv
   - Join type: LEFT join to preserve all show rows

3. **Economic indicators** (by date):
   - Key: 'start_date' or 'end_date' (temporal matching)
   - Sources: economics/oil_price.csv, economics/unemployment_by_city.csv,
             economics/commodity_price_index.csv, economics/boc_cpi_monthly.csv
   - Join type: LEFT join (merge_asof for temporal alignment)

4. **Baseline signals** (by show_title):
   - Key: 'show_title' (via canonicalized title matching)
   - Source: productions/baselines.csv
   - Join type: LEFT join to preserve all show rows
"""

import pandas as pd
from data.loader import (
    load_history_sales,
    load_past_runs,
    load_baselines,
    load_marketing_spend,
    join_history_with_weather,
    join_history_with_marketing_spend,
    join_history_with_external_data,
)
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


def _merge_with_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Merge history sales data with baseline signals by show_title.

    JOIN LOGIC:
    -----------
    - Key: 'show_title' (via canonicalized title matching)
    - Join Type: LEFT join to preserve all show rows
    - Source: productions/baselines.csv

    Baseline signals include wiki, trends, youtube, spotify scores
    that provide familiarity and engagement metrics for each title.

    Args:
        df: DataFrame with 'show_title' column.

    Returns:
        DataFrame with baseline signal columns added (wiki, trends, youtube, spotify,
        category, gender). Rows without a matching baseline will have NaN for
        these columns.
    """
    baselines = load_baselines(fallback_empty=True)
    if baselines.empty:
        return df

    df = df.copy()

    # Create canonical title columns for matching
    df["_canonical_title"] = df["show_title"].apply(canonicalize_title)

    baselines = baselines.copy()
    if "title" in baselines.columns:
        baselines["_canonical_title"] = baselines["title"].apply(canonicalize_title)
    else:
        return df

    # Select baseline columns to merge (exclude internal columns)
    baseline_cols = ["_canonical_title"]
    for col in ["wiki", "trends", "youtube", "spotify", "category", "gender"]:
        if col in baselines.columns:
            baseline_cols.append(col)

    # Perform LEFT join on canonical title
    merged = df.merge(
        baselines[baseline_cols],
        on="_canonical_title",
        how="left",
        suffixes=("", "_baseline")
    )

    # Remove temporary column
    merged = merged.drop(columns=["_canonical_title"], errors="ignore")

    # Remove any duplicate columns from merge
    dup_cols = [c for c in merged.columns if c.endswith("_baseline")]
    if dup_cols:
        merged = merged.drop(columns=dup_cols)

    return merged


def _merge_with_external_data(
    df: pd.DataFrame,
    date_key: str = "start_date",
    include_weather: bool = True,
    include_marketing: bool = True
) -> pd.DataFrame:
    """Merge history sales data with external datasets.

    JOIN LOGIC:
    -----------
    This function joins history data with multiple external data sources:

    1. Marketing spend data (by city and date):
       - Key: 'show_title' and date_key ('start_date' or 'end_date')
       - Join Type: LEFT join to preserve all show rows
       - Source: productions/marketing_spend_per_ticket.csv

    2. Weather data (by city and date):
       - Key: 'city' and date_key ('start_date' or 'end_date')
       - Join Type: LEFT join to preserve all show rows
       - Source: environment/weatherstats_calgary_daily.csv, weatherstats_edmonton_daily.csv

    All joins are LEFT joins to ensure no show rows are lost, even if
    external data is missing.

    Args:
        df: DataFrame with date columns (start_date, end_date) and city column.
        date_key: Which date column to use for joining ('start_date' or 'end_date').
        include_weather: Whether to join weather data.
        include_marketing: Whether to join marketing spend data.

    Returns:
        DataFrame with external data columns added. Original rows are preserved.
    """
    if df.empty:
        return df

    result = df.copy()

    # Join marketing spend data
    if include_marketing:
        result = join_history_with_marketing_spend(
            result,
            date_key=date_key,
            fallback_empty=True
        )

    # Join weather data
    if include_weather:
        city_column = "city" if "city" in result.columns else None
        if city_column and date_key in result.columns:
            result = join_history_with_weather(
                result,
                date_key=date_key,
                city_column=city_column,
                fallback_empty=True
            )

    return result


def build_dataset(
    theme_filters=None,
    status=None,
    forecast_time_only=True,
    include_external_data: bool = True,
    include_baselines: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """Build a dataset for training or scoring.

    After loading history_city_sales.csv, this function joins with:

    1. **Marketing spend data** (by city and date):
       - Key: 'show_title' and 'start_date' or 'end_date'
       - Join Type: LEFT join to preserve all show rows

    2. **Weather data** (by city and date):
       - Key: 'city' and 'start_date' (or 'end_date')
       - Join Type: LEFT join to preserve all show rows

    3. **Economic indicators** (by date) - via derive_basic_features:
       - Key: 'start_date' or 'end_date' (temporal matching)
       - Join Type: LEFT join (merge_asof for temporal alignment)

    4. **Baseline signals** (by show_title):
       - Key: 'show_title' (via canonicalized title matching)
       - Join Type: LEFT join to preserve all show rows

    The returned DataFrame X includes a DATE_COL ('end_date') column of dtype
    datetime64[ns] suitable for time-aware cross-validation. This date column
    is merged from past_runs.csv based on show title matching.

    Args:
        theme_filters: Optional list of themes to filter features by.
        status: Optional status filter for features.
        forecast_time_only: If True, apply leakage filtering for forecast time.
        include_external_data: If True, join with marketing spend and weather data.
        include_baselines: If True, join with baseline signals (wiki, trends, etc.).

    Returns:
        Tuple of (X, y) where:
        - X: Feature DataFrame including DATE_COL for time-based CV.
        - y: Target Series (total_single_tickets).
    """
    raw = load_history_sales()

    # Merge with past_runs to add date columns (start_date, end_date)
    raw = _merge_with_past_runs(raw)

    # Merge with baseline signals by show_title
    if include_baselines:
        raw = _merge_with_baselines(raw)

    # Merge with external data (marketing spend, weather) by city and date
    if include_external_data:
        raw = _merge_with_external_data(
            raw,
            date_key="start_date",
            include_weather=True,
            include_marketing=True
        )

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
