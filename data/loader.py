from pathlib import Path
from functools import lru_cache
from typing import Optional
from dataclasses import dataclass
import logging
import os
import warnings

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"

# Configure logging
logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Raised when data loading fails."""
    pass


def _get_file_mtime(path: Path) -> float:
    """Get file modification time for cache invalidation."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _clean_dataframe(
    df: pd.DataFrame,
    drop_unnamed: bool = True,
    drop_empty_unnamed_only: bool = False,
    numeric_columns: Optional[list] = None
) -> pd.DataFrame:
    """Clean a DataFrame by dropping unnamed columns and converting numeric strings.
    
    Args:
        df: DataFrame to clean
        drop_unnamed: If True, drop columns matching 'unnamed:' pattern (case-insensitive)
        drop_empty_unnamed_only: If True, only drop unnamed columns that are entirely empty/NaN
        numeric_columns: List of column names to convert to numeric using pd.to_numeric
        
    Returns:
        Cleaned DataFrame (copy of input, original is not modified)
    """
    if df.empty:
        return df
    
    # Create a copy to avoid mutating the input DataFrame
    df = df.copy()
    
    # Drop unnamed columns (e.g., 'Unnamed: 0', 'unnamed:_0')
    if drop_unnamed:
        unnamed_cols = [c for c in df.columns if 'unnamed' in str(c).lower()]
        if unnamed_cols:
            if drop_empty_unnamed_only:
                # Only drop unnamed columns that are entirely empty/NaN
                cols_to_drop = [c for c in unnamed_cols if df[c].isna().all()]
            else:
                cols_to_drop = unnamed_cols
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
    
    # Convert specified columns to numeric
    if numeric_columns:
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


@lru_cache(maxsize=16)
def _load_history_sales_cached(path: str, mtime: float) -> pd.DataFrame:
    """
    Load history sales CSV with caching.
    
    Args:
        path: Path to CSV file
        mtime: File modification time (used for cache key)
        
    Returns:
        DataFrame with normalized column names
    """
    df = pd.read_csv(path, thousands=",")
    # Normalize column names
    df.columns = [
        c.strip().lower().replace(" - ", "_").replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]
    return df


@lru_cache(maxsize=16)
def _load_baselines_cached(path: str, mtime: float) -> pd.DataFrame:
    """
    Load baselines CSV with caching.
    
    Args:
        path: Path to CSV file
        mtime: File modification time (used for cache key)
        
    Returns:
        DataFrame with normalized column names
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def load_history_sales(
    csv_name: str = "productions/history_city_sales.csv",
    fallback_empty: bool = False
) -> pd.DataFrame:
    """Load historical show-level sales (Calgary/Edmonton).
    
    This function is cached based on file modification time.
    
    Args:
        csv_name: Name of the CSV file to load
        fallback_empty: If True, return empty DataFrame on error instead of raising
        
    Returns:
        DataFrame with normalized column names
        
    Raises:
        DataLoadError: If file cannot be loaded and fallback_empty is False
    """
    path = DATA_DIR / csv_name
    
    try:
        if not path.exists():
            if fallback_empty:
                warnings.warn(f"History sales file not found: {path}. Using empty DataFrame.")
                return pd.DataFrame()
            raise DataLoadError(f"History sales file not found: {path}")
        
        # Use cached loader with file mtime for cache invalidation
        mtime = _get_file_mtime(path)
        return _load_history_sales_cached(str(path), mtime).copy()
        
    except DataLoadError:
        raise
    except Exception as e:
        if fallback_empty:
            warnings.warn(f"Error loading history sales: {e}. Using empty DataFrame.")
            return pd.DataFrame()
        raise DataLoadError(f"Error loading history sales from {path}: {e}")


def load_baselines(
    csv_name: str = "productions/baselines.csv",
    fallback_empty: bool = True
) -> pd.DataFrame:
    """Load baseline signals (wiki, trends, youtube, spotify) for all titles.
    
    This function is cached based on file modification time.
    
    The baselines file contains both:
    - Historical titles (source='historical'): Alberta Ballet performances with ticket data
    - Reference titles (source='external_reference'): Well-known titles without AB history
    
    Args:
        csv_name: Name of the CSV file to load
        fallback_empty: If True, return empty DataFrame on error (default True for backwards compat)
        
    Returns:
        DataFrame with columns: title, wiki, trends, youtube, spotify, category, 
                                gender, source, notes
    """
    path = DATA_DIR / csv_name
    
    try:
        if not path.exists():
            if fallback_empty:
                warnings.warn(f"Baselines file not found: {path}. Using empty DataFrame.")
                return pd.DataFrame()
            raise DataLoadError(f"Baselines file not found: {path}")
        
        # Use cached loader with file mtime for cache invalidation
        mtime = _get_file_mtime(path)
        return _load_baselines_cached(str(path), mtime).copy()
        
    except DataLoadError:
        raise
    except Exception as e:
        if fallback_empty:
            warnings.warn(f"Error loading baselines: {e}. Using empty DataFrame.")
            return pd.DataFrame()
        raise DataLoadError(f"Error loading baselines from {path}: {e}")


def load_reference_baselines(csv_name: str = "productions/baselines.csv") -> pd.DataFrame:
    """Load reference baseline signals for titles without historical ticket data.
    
    DEPRECATED: All baselines are now in a single baselines.csv file with a 'source' 
    column. Use load_baselines() and filter by source='external_reference' instead.
    
    This function is kept for backward compatibility but now loads from baselines.csv
    and filters to only external reference titles.
    
    Returns:
        DataFrame with columns: title, wiki, trends, youtube, spotify, category, 
                                gender, source, notes
    """
    df = load_baselines(csv_name)
    if df.empty:
        return df
    # Filter to only external reference titles
    if "source" in df.columns:
        df = df[df["source"] == "external_reference"]
    return df


def load_all_baselines(
    include_reference: bool = True,
    baselines_path: str = "productions/baselines.csv",
    reference_path: str = "productions/baselines.csv"
) -> pd.DataFrame:
    """Load all baseline signals, optionally including reference titles.
    
    All baselines are now stored in a single baselines.csv file with a 'source' column
    that distinguishes between:
    - 'historical': Alberta Ballet performances with ticket data
    - 'external_reference': Well-known titles without AB history
    
    Use this for k-NN similarity matching when you want the broadest signal comparison.
    
    Args:
        include_reference: Whether to include reference titles without AB history
        baselines_path: Path to baselines CSV (contains all titles)
        reference_path: Ignored, kept for backward compatibility
        
    Returns:
        DataFrame with columns: title, wiki, trends, youtube, spotify,
                                category, gender, source
    """
    baselines = load_baselines(baselines_path)
    
    if baselines.empty:
        return pd.DataFrame()
    
    # Filter to historical only if include_reference is False
    if not include_reference:
        if "source" in baselines.columns:
            baselines = baselines[baselines["source"] == "historical"]
    
    return baselines


# =============================================================================
# TICKET COLUMN DEFINITIONS & ACCURACY GUIDE
# =============================================================================
# The history_city_sales.csv file contains single ticket data.
# This app focuses on single ticket estimation only.
#
# ACTUAL DATA (from Box Office / Tessitura):
# ------------------------------------------
# 1. single_tickets_calgary / single_tickets_edmonton
#    - Source: Actual box office sales data
#    - What it is: The real number of single tickets sold in each city
#    - Use case: Ground truth for training & validation
#
# 2. total_single_tickets
#    - Source: Sum of single_tickets_calgary + single_tickets_edmonton
#    - What it is: Total single tickets across both cities (ACTUAL)
#    - Use case: Ground truth target variable for ML models
#
# MODEL PREDICTIONS (from external model - your prior forecasting system):
# ------------------------------------------------------------------------
# 3. yourmodel_single_tickets_calgary / yourmodel_single_tickets_edmonton
#    - Source: External forecasting model (the "YourModel" system)
#    - What it is: PREDICTED single tickets by city from a prior model
#    - Note: This appears to be a pre-existing forecast, not this app
#
# 4. yourmodel_total_single_tickets
#    - Source: Sum of YourModel predictions across cities
#    - What it is: PREDICTED total singles from the external model
#    - Use case: Benchmark comparison to evaluate new model accuracy
#
# APP-GENERATED FORECASTS (from streamlit_app.py):
# ------------------------------------------------
# 5. EstimatedTickets / EstimatedTickets_Final
#    - Source: This Streamlit app's scoring algorithm
#    - What it is: Predicted single tickets using familiarity/motivation signals
#    - How calculated:
#      a) Compute Familiarity & Motivation indices (Wikipedia, Trends, etc.)
#      b) Convert to TicketIndex using regression on historical data
#      c) Apply seasonality factor for the chosen run month
#      d) Apply remount decay if the title ran recently
#      e) Apply post-COVID adjustment factor (e.g., 0.85)
#    - EstimatedTickets = before remount decay
#    - EstimatedTickets_Final = after remount decay & post-COVID haircut
#
# ACCURACY COMPARISON:
# --------------------
# To evaluate which forecast is more accurate, compare to total_single_tickets:
#
# | Column                        | What it Predicts | How to Assess Accuracy         |
# |-------------------------------|------------------|--------------------------------|
# | yourmodel_total_single_tickets| Singles (prior)  | Compare MAE vs total_single_tickets |
# | EstimatedTickets_Final        | Singles          | Compare MAE vs total_single_tickets |
#
# To run a quick accuracy check, use the Model Validation page in the app or:
#   - MAE = mean absolute error (lower is better)
#   - RMSE = root mean squared error (penalizes big misses)
#   - R² = correlation (closer to 1.0 is better)
#
# =============================================================================
# EXTERNAL FACTORS INTEGRATION
# =============================================================================
# The app is NOT currently programmed to automatically pull external factors.
# To integrate external factors (economic indicators, weather, etc.) into the
# model, you will need to:
#
# 1. CREATE CSV FILES in the data/ directory with external factor data.
#    Example files to create:
#      - data/external_economic.csv (alberta_unemployment_rate, alberta_cpi_index, etc.)
#      - data/external_weather.csv (weather_severity_index by city/date)
#
# 2. ADD LOADER FUNCTIONS below to read each external factor file.
#
# 3. MERGE the data in build_modelling_dataset() or a similar function.
#    The merge key will typically be:
#      - show_title + production_season (for per-show factors)
#      - opening_date or month_of_opening (for time-based factors)
#      - city (for city-specific factors)
#
# EXAMPLE: To add economic factors, create data/external_economic.csv with:
#   production_season,alberta_unemployment_rate,alberta_cpi_index,wti_oil_price_avg
#   2023-24,5.8,157.2,78.50
#   2024-25,6.1,162.3,82.00
#
# Then add a loader function:
#
# def load_external_economic(csv_name: str = "external_economic.csv") -> pd.DataFrame:
#     """Load external economic indicators by production season."""
#     path = DATA_DIR / csv_name
#     if not path.exists():
#         return pd.DataFrame()  # Return empty if file doesn't exist
#     df = pd.read_csv(path)
#     df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
#     return df
#
# And merge in your dataset builder:
#
# def build_dataset_with_external():
#     sales = load_history_sales()
#     economic = load_external_economic()
#     if not economic.empty:
#         sales = sales.merge(economic, on="production_season", how="left")
#     return sales
#
# =============================================================================
# PROMPT FOR FUTURE EXTERNAL FACTORS INTEGRATION
# =============================================================================
# Copy this prompt when you're ready to add external factors:
#
# """
# I've created CSV files for external factors in the data/ directory.
# Please update the codebase to integrate these factors into the model:
#
# 1. Add loader functions in data/loader.py for each CSV file listed below
# 2. Update scripts/build_modelling_dataset.py to merge these factors
# 3. Add the new features to config/ml_feature_inventory_alberta_ballet.csv
# 4. Update config/ml_leakage_audit_alberta_ballet.csv to mark them as safe
#
# My external factor files are:
# - [FILE1.csv] with columns [col1, col2, ...] joined on [production_season]
# - [FILE2.csv] with columns [col1, col2, ...] joined on [month_of_opening]
#
# Please ensure the merge uses left joins to preserve all historical records.
# """
#
# =============================================================================


# =============================================================================
# EXTERNAL FACTORS LOADER FUNCTIONS
# =============================================================================
# These functions load external factor files for enhancing ML model predictions.
# The merged external factors file can be used to enhance ML model predictions.

@lru_cache(maxsize=8)
def _load_external_factors_cached(path: str, mtime: float) -> pd.DataFrame:
    """
    Load external factors CSV with caching.
    
    Args:
        path: Path to CSV file
        mtime: File modification time (used for cache key)
        
    Returns:
        DataFrame with normalized column names
    """
    df = pd.read_csv(path)
    df.columns = [
        c.strip().lower().replace(" - ", "_").replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]
    return df


def load_external_factors(
    csv_name: str = "economics/external_factors.csv",
    fallback_empty: bool = True
) -> pd.DataFrame:
    """Load merged external factors data (economic, demographic, tourism, etc.).
    
    This file contains external factors that can enhance ML model predictions.
    
    Expected columns may include:
    - year: Calendar year (key for joining)
    - city: City name (Calgary/Edmonton) for city-specific data
    - alberta_unemployment_rate: Provincial unemployment rate
    - alberta_cpi_index: Consumer Price Index for Alberta
    - alberta_real_gdp_growth_rate: GDP growth percentage
    - wti_oil_price_avg: Average WTI oil price in USD
    - exchange_rate_cad_usd: CAD to USD exchange rate
    - population_city: City population
    - median_household_income_city: Median household income
    - tourism_visitation_index: Tourism activity index
    - arts_sector_confidence_index: Arts sector health index
    
    Args:
        csv_name: Name of the CSV file to load
        fallback_empty: If True, return empty DataFrame on error (default True)
        
    Returns:
        DataFrame with external factors data
    """
    path = DATA_DIR / csv_name
    
    try:
        if not path.exists():
            if fallback_empty:
                return pd.DataFrame()
            raise DataLoadError(f"External factors file not found: {path}")
        
        mtime = _get_file_mtime(path)
        return _load_external_factors_cached(str(path), mtime).copy()
        
    except DataLoadError:
        raise
    except Exception as e:
        if fallback_empty:
            warnings.warn(f"Error loading external factors: {e}. Using empty DataFrame.")
            return pd.DataFrame()
        raise DataLoadError(f"Error loading external factors from {path}: {e}")


def load_history_with_external_factors(
    history_csv: str = "productions/history_city_sales.csv",
    external_csv: str = "economics/external_factors.csv",
    join_on: Optional[list] = None
) -> pd.DataFrame:
    """Load history sales data merged with external factors.
    
    This is a convenience function that combines the historical sales data
    with external factors (economic, demographic, etc.) for enhanced modeling.
    
    The merge is performed on common columns:
    - If 'year' or 'season_year' exists in both: merge on year
    - If 'city' exists in both: additionally merge on city
    
    Args:
        history_csv: Name of the history sales CSV
        external_csv: Name of the external factors CSV
        join_on: Optional list of columns to join on. If None, auto-detects.
        
    Returns:
        DataFrame with history data merged with external factors
    """
    history = load_history_sales(history_csv, fallback_empty=True)
    external = load_external_factors(external_csv, fallback_empty=True)
    
    if history.empty:
        return history
    
    if external.empty:
        logger.info("No external factors file found. Returning history only.")
        return history
    
    # Suffix constant for consistency
    _MERGE_SUFFIX = "_ext"
    
    # Auto-detect join keys if not specified
    if join_on is None:
        join_on = []
        left_on = []
        right_on = []
        
        # Check for year column
        history_year_col = None
        if "year" in history.columns:
            history_year_col = "year"
        elif "season_year" in history.columns:
            history_year_col = "season_year"
        
        external_year_col = "year" if "year" in external.columns else None
        
        if history_year_col and external_year_col:
            left_on.append(history_year_col)
            right_on.append(external_year_col)
        
        # Check for city column
        if "city" in history.columns and "city" in external.columns:
            left_on.append("city")
            right_on.append("city")
        
        if not left_on:
            logger.warning("No common join keys found between history and external factors.")
            return history
        
        # Perform merge with potentially different column names
        try:
            merged = history.merge(
                external,
                left_on=left_on,
                right_on=right_on,
                how="left",
                suffixes=("", _MERGE_SUFFIX)
            )
            
            # Remove duplicate columns with the suffix
            merged = merged.loc[:, ~merged.columns.str.endswith(_MERGE_SUFFIX)]
            
            logger.info(f"Merged history with external factors on {left_on}. "
                       f"Result: {len(merged)} rows, {len(merged.columns)} columns.")
            return merged
            
        except Exception as e:
            logger.warning(f"Failed to merge external factors: {e}. Returning history only.")
            return history
    
    # If join_on is explicitly provided, use it directly
    try:
        merged = history.merge(
            external,
            on=join_on,
            how="left",
            suffixes=("", _MERGE_SUFFIX)
        )
        
        # Remove duplicate columns with the suffix
        merged = merged.loc[:, ~merged.columns.str.endswith(_MERGE_SUFFIX)]
        
        logger.info(f"Merged history with external factors on {join_on}. "
                   f"Result: {len(merged)} rows, {len(merged.columns)} columns.")
        return merged
        
    except Exception as e:
        logger.warning(f"Failed to merge external factors: {e}. Returning history only.")
        return history


# =============================================================================
# ECONOMIC DATA LOADERS
# =============================================================================
# These functions load economic indicator files for market sentiment analysis.

@lru_cache(maxsize=8)
def _load_csv_cached(path: str, mtime: float) -> pd.DataFrame:
    """Generic cached CSV loader with normalized column names."""
    df = pd.read_csv(path)
    df.columns = [
        c.strip().lower().replace(" - ", "_").replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]
    return df


def load_oil_prices(
    csv_name: str = "economics/oil_price.csv",
    fallback_empty: bool = True
) -> pd.DataFrame:
    """Load historical oil prices (WCS and WTI).
    
    Oil prices are a key economic indicator for Alberta's economy and can
    influence discretionary spending patterns including arts attendance.
    
    Expected columns:
    - date: Date of the price observation (YYYY-MM-DD)
    - wcs_oil_price: Western Canadian Select price in USD
    - oil_series: Which series (WCS or WTI)
    
    Args:
        csv_name: Path to oil price CSV relative to data directory
        fallback_empty: If True, return empty DataFrame on error
        
    Returns:
        DataFrame with oil price data
    """
    path = DATA_DIR / csv_name
    
    try:
        if not path.exists():
            if fallback_empty:
                return pd.DataFrame()
            raise DataLoadError(f"Oil price file not found: {path}")
        
        mtime = _get_file_mtime(path)
        return _load_csv_cached(str(path), mtime).copy()
        
    except DataLoadError:
        raise
    except Exception as e:
        if fallback_empty:
            warnings.warn(f"Error loading oil prices: {e}. Using empty DataFrame.")
            return pd.DataFrame()
        raise DataLoadError(f"Error loading oil prices from {path}: {e}")


def load_unemployment_rates(
    csv_name: str = "economics/unemployment_by_city.csv",
    fallback_empty: bool = True
) -> pd.DataFrame:
    """Load unemployment rates by city (Calgary, Edmonton, Alberta).
    
    Unemployment rates indicate economic health and consumer confidence,
    which can affect arts attendance.
    
    Expected columns:
    - date: Date of the observation (YYYY-MM-DD)
    - unemployment_rate: Unemployment rate as percentage
    - region: Geographic region (Alberta, Calgary, Edmonton, Lethbridge)
    
    Args:
        csv_name: Path to unemployment CSV relative to data directory
        fallback_empty: If True, return empty DataFrame on error
        
    Returns:
        DataFrame with unemployment rate data
    """
    path = DATA_DIR / csv_name
    
    try:
        if not path.exists():
            if fallback_empty:
                return pd.DataFrame()
            raise DataLoadError(f"Unemployment file not found: {path}")
        
        mtime = _get_file_mtime(path)
        return _load_csv_cached(str(path), mtime).copy()
        
    except DataLoadError:
        raise
    except Exception as e:
        if fallback_empty:
            warnings.warn(f"Error loading unemployment rates: {e}. Using empty DataFrame.")
            return pd.DataFrame()
        raise DataLoadError(f"Error loading unemployment rates from {path}: {e}")


def load_segment_priors(
    csv_name: str = "productions/segment_priors.csv",
    fallback_empty: bool = True
) -> pd.DataFrame:
    """Load segment prior weights by region and category.
    
    Segment priors represent the relative propensity of different audience
    segments to attend shows of different categories. Used for segment-based
    ticket estimation adjustments.
    
    Expected columns:
    - region: Geographic region (Province, Calgary, Edmonton)
    - category: Show category (classic_romance, family_classic, etc.)
    - segment: Audience segment name
    - weight: Prior weight multiplier
    
    Args:
        csv_name: Path to segment priors CSV relative to data directory
        fallback_empty: If True, return empty DataFrame on error
        
    Returns:
        DataFrame with segment prior weights
    """
    path = DATA_DIR / csv_name
    
    try:
        if not path.exists():
            if fallback_empty:
                return pd.DataFrame()
            raise DataLoadError(f"Segment priors file not found: {path}")
        
        mtime = _get_file_mtime(path)
        return _load_csv_cached(str(path), mtime).copy()
        
    except DataLoadError:
        raise
    except Exception as e:
        if fallback_empty:
            warnings.warn(f"Error loading segment priors: {e}. Using empty DataFrame.")
            return pd.DataFrame()
        raise DataLoadError(f"Error loading segment priors from {path}: {e}")


def load_audience_analytics(
    csv_name: str = "audiences/live_analytics.csv",
    fallback_empty: bool = True
) -> pd.DataFrame:
    """Load audience analytics data from live event research.
    
    Contains demographic and behavioral data about audiences for different
    show categories, useful for understanding audience composition.
    
    Args:
        csv_name: Path to audience analytics CSV relative to data directory
        fallback_empty: If True, return empty DataFrame on error
        
    Returns:
        DataFrame with audience analytics data (empty unnamed columns dropped)
    """
    path = DATA_DIR / csv_name
    
    try:
        if not path.exists():
            if fallback_empty:
                return pd.DataFrame()
            raise DataLoadError(f"Audience analytics file not found: {path}")
        
        mtime = _get_file_mtime(path)
        df = _load_csv_cached(str(path), mtime).copy()
        
        # Clean the DataFrame: drop only empty unnamed columns (preserve structure)
        df = _clean_dataframe(df, drop_unnamed=True, drop_empty_unnamed_only=True)
        
        return df
        
    except DataLoadError:
        raise
    except Exception as e:
        if fallback_empty:
            warnings.warn(f"Error loading audience analytics: {e}. Using empty DataFrame.")
            return pd.DataFrame()
        raise DataLoadError(f"Error loading audience analytics from {path}: {e}")


def load_past_runs(
    csv_name: str = "productions/past_runs.csv",
    fallback_empty: bool = True
) -> pd.DataFrame:
    """Load past production run dates.
    
    Contains start and end dates for historical productions, useful for
    calculating remount timing and seasonality factors.
    
    Expected columns:
    - title: Show title
    - start_date: Run start date
    - end_date: Run end date
    
    Args:
        csv_name: Path to past runs CSV relative to data directory
        fallback_empty: If True, return empty DataFrame on error
        
    Returns:
        DataFrame with past run dates
    """
    path = DATA_DIR / csv_name
    
    try:
        if not path.exists():
            if fallback_empty:
                return pd.DataFrame()
            raise DataLoadError(f"Past runs file not found: {path}")
        
        mtime = _get_file_mtime(path)
        df = _load_csv_cached(str(path), mtime).copy()
        
        # Parse dates if present
        for col in ['start_date', 'end_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
        
    except DataLoadError:
        raise
    except Exception as e:
        if fallback_empty:
            warnings.warn(f"Error loading past runs: {e}. Using empty DataFrame.")
            return pd.DataFrame()
        raise DataLoadError(f"Error loading past runs from {path}: {e}")


# =============================================================================
# ECONOMIC SENTIMENT CALCULATOR
# =============================================================================

def get_economic_sentiment_factor(
    run_date: Optional[pd.Timestamp] = None,
    city: Optional[str] = None,
    baseline_oil_price: float = 60.0,
    baseline_unemployment: float = 6.0,
    oil_weight: float = 0.4,
    unemployment_weight: float = 0.6,
    min_factor: float = 0.85,
    max_factor: float = 1.10
) -> float:
    """Calculate an economic sentiment adjustment factor.
    
    Combines oil price and unemployment data to produce a multiplier
    that can adjust ticket estimates based on economic conditions.
    
    Higher oil prices and lower unemployment generally correlate with
    higher discretionary spending, which benefits arts attendance.
    
    Args:
        run_date: Date of the planned show run (defaults to current date)
        city: City for unemployment lookup (Calgary, Edmonton, or None for Alberta)
        baseline_oil_price: Reference oil price (USD) for neutral factor
        baseline_unemployment: Reference unemployment rate (%) for neutral factor
        oil_weight: Weight for oil price component (0-1)
        unemployment_weight: Weight for unemployment component (0-1)
        min_factor: Minimum allowed factor (floor)
        max_factor: Maximum allowed factor (ceiling)
        
    Returns:
        Economic sentiment factor (1.0 = neutral, >1.0 = favorable, <1.0 = unfavorable)
    """
    if run_date is None:
        run_date = pd.Timestamp.now()
    
    # Normalize weights
    total_weight = oil_weight + unemployment_weight
    if total_weight > 0:
        oil_weight = oil_weight / total_weight
        unemployment_weight = unemployment_weight / total_weight
    else:
        return 1.0
    
    # Get oil price data
    oil_df = load_oil_prices()
    oil_factor = 1.0
    if not oil_df.empty and 'date' in oil_df.columns:
        oil_df['date'] = pd.to_datetime(oil_df['date'], errors='coerce')
        # Filter to WCS (Western Canadian Select - more relevant for Alberta)
        wcs_df = oil_df[oil_df.get('oil_series', '') == 'WCS'].copy()
        if wcs_df.empty:
            wcs_df = oil_df.copy()
        
        # Find most recent price before run date
        recent = wcs_df[wcs_df['date'] <= run_date].sort_values('date', ascending=False)
        if not recent.empty:
            # Get price column safely
            if 'wcs_oil_price' in recent.columns:
                price_col = 'wcs_oil_price'
            else:
                # Fallback: find a numeric column that's not 'date'
                numeric_cols = recent.select_dtypes(include=['float64', 'int64']).columns
                price_col = numeric_cols[0] if len(numeric_cols) > 0 else None
            
            if price_col is not None:
                current_price = recent.iloc[0][price_col]
                if pd.notna(current_price) and current_price > 0:
                    # Oil factor: higher prices = more positive sentiment
                    # Range roughly 0.8 to 1.2 based on typical oil price ranges
                    oil_factor = 0.8 + 0.4 * min(1.5, max(0.5, current_price / baseline_oil_price))
    
    # Get unemployment data
    unemp_df = load_unemployment_rates()
    unemp_factor = 1.0
    if not unemp_df.empty and 'date' in unemp_df.columns:
        unemp_df['date'] = pd.to_datetime(unemp_df['date'], errors='coerce')
        
        # Filter by city/region
        region = city if city in ['Calgary', 'Edmonton'] else 'Alberta'
        if 'region' in unemp_df.columns:
            region_df = unemp_df[unemp_df['region'] == region].copy()
        else:
            region_df = unemp_df.copy()
        
        # Find most recent rate before run date
        recent = region_df[region_df['date'] <= run_date].sort_values('date', ascending=False)
        if not recent.empty and 'unemployment_rate' in recent.columns:
            current_rate = recent.iloc[0]['unemployment_rate']
            if pd.notna(current_rate) and current_rate > 0:
                # Unemployment factor: lower rates = more positive sentiment
                # Inverse relationship: low unemployment is good for spending
                unemp_factor = 0.8 + 0.4 * min(1.5, max(0.5, baseline_unemployment / current_rate))
    
    # Combine factors with weights
    combined = (oil_weight * oil_factor) + (unemployment_weight * unemp_factor)
    
    # Clip to min/max range
    return float(max(min_factor, min(max_factor, combined)))


def get_segment_weight(
    region: str,
    category: str,
    segment: str,
    default: float = 1.0
) -> float:
    """Get segment prior weight for a specific region/category/segment combination.
    
    Args:
        region: Geographic region (Province, Calgary, Edmonton)
        category: Show category (classic_romance, family_classic, etc.)
        segment: Audience segment name
        default: Default weight if not found
        
    Returns:
        Segment weight multiplier
    """
    df = load_segment_priors()
    if df.empty:
        return default
    
    # Normalize inputs for matching
    region_lower = region.lower().strip()
    category_lower = category.lower().strip()
    segment_lower = segment.lower().strip()
    
    # Try exact match first
    mask = (
        df['region'].str.lower().str.strip() == region_lower
    ) & (
        df['category'].str.lower().str.strip() == category_lower
    ) & (
        df['segment'].str.lower().str.strip() == segment_lower
    )
    
    matches = df[mask]
    if not matches.empty and 'weight' in matches.columns:
        return float(matches.iloc[0]['weight'])
    
    # Try Province-level fallback
    if region_lower != 'province':
        mask = (
            df['region'].str.lower().str.strip() == 'province'
        ) & (
            df['category'].str.lower().str.strip() == category_lower
        ) & (
            df['segment'].str.lower().str.strip() == segment_lower
        )
        matches = df[mask]
        if not matches.empty and 'weight' in matches.columns:
            return float(matches.iloc[0]['weight'])
    
    return default


def get_all_segment_weights(region: str, category: str) -> dict:
    """Get all segment weights for a region/category combination.
    
    Args:
        region: Geographic region (Province, Calgary, Edmonton)
        category: Show category
        
    Returns:
        Dictionary mapping segment names to weights
    """
    df = load_segment_priors()
    if df.empty:
        return {}
    
    region_lower = region.lower().strip()
    category_lower = category.lower().strip()
    
    # Try region-specific first
    mask = (
        df['region'].str.lower().str.strip() == region_lower
    ) & (
        df['category'].str.lower().str.strip() == category_lower
    )
    matches = df[mask]
    
    # Fallback to Province
    if matches.empty and region_lower != 'province':
        mask = (
            df['region'].str.lower().str.strip() == 'province'
        ) & (
            df['category'].str.lower().str.strip() == category_lower
        )
        matches = df[mask]
    
    if matches.empty:
        return {}
    
    return dict(zip(matches['segment'], matches['weight']))


# =============================================================================
# PREDICTHQ EVENT DATA LOADERS
# =============================================================================
# These functions load PredictHQ event data for demand forecasting.
# The data provides ML-ready features like predicted attendance, event ranks,
# and demand impact scores.

def load_predicthq_events(
    csv_name: str = "predicthq/predicthq_events.csv",
    fallback_empty: bool = True
) -> pd.DataFrame:
    """Load PredictHQ event data for demand intelligence.
    
    This file contains aggregated PredictHQ features for production run windows.
    These features help predict demand by quantifying competing events.
    
    Expected columns:
    - show_title: Show title (key for joining)
    - city: City name (Calgary/Edmonton)
    - phq_start_date: Start date of the run window
    - phq_end_date: End date of the run window
    - phq_attendance_sum: Total predicted attendance for all events
    - phq_attendance_sports: Attendance for sports events
    - phq_attendance_concerts: Attendance for concerts
    - phq_event_count: Count of significant events (rank >= 30)
    - phq_rank_max: Maximum event rank (0-100)
    - phq_rank_avg: Average event rank
    - phq_holidays_flag: Whether holidays overlap the run (0/1)
    - phq_severe_weather_flag: Whether severe weather overlaps (0/1)
    - phq_demand_impact_score: Composite demand impact score
    
    Args:
        csv_name: Path to PredictHQ events CSV relative to data directory
        fallback_empty: If True, return empty DataFrame on error
        
    Returns:
        DataFrame with PredictHQ event features
    """
    path = DATA_DIR / csv_name
    
    try:
        if not path.exists():
            if fallback_empty:
                return pd.DataFrame()
            raise DataLoadError(f"PredictHQ events file not found: {path}")
        
        mtime = _get_file_mtime(path)
        return _load_csv_cached(str(path), mtime).copy()
        
    except DataLoadError:
        raise
    except Exception as e:
        if fallback_empty:
            warnings.warn(f"Error loading PredictHQ events: {e}. Using empty DataFrame.")
            return pd.DataFrame()
        raise DataLoadError(f"Error loading PredictHQ events from {path}: {e}")


def load_history_with_predicthq(
    history_csv: str = "productions/history_city_sales.csv",
    predicthq_csv: str = "predicthq/predicthq_events.csv",
    join_on: Optional[list] = None
) -> pd.DataFrame:
    """Load history sales data merged with PredictHQ event features.
    
    This is a convenience function that combines historical sales data
    with PredictHQ demand intelligence features for enhanced ML modeling.
    
    The merge is performed on common columns:
    - If 'show_title' exists in both: merge on show_title
    - If 'city' exists in both: additionally merge on city
    
    Args:
        history_csv: Name of the history sales CSV
        predicthq_csv: Name of the PredictHQ events CSV
        join_on: Optional list of columns to join on. If None, auto-detects.
        
    Returns:
        DataFrame with history data merged with PredictHQ features
    """
    history = load_history_sales(history_csv, fallback_empty=True)
    predicthq = load_predicthq_events(predicthq_csv, fallback_empty=True)
    
    if history.empty:
        return history
    
    if predicthq.empty:
        logger.info("No PredictHQ events file found. Returning history only.")
        return history
    
    # Suffix constant for consistency
    _MERGE_SUFFIX = "_phq"
    
    # Auto-detect join keys if not specified
    if join_on is None:
        left_on = []
        right_on = []
        
        # Check for show_title column
        history_title_col = None
        if "show_title" in history.columns:
            history_title_col = "show_title"
        elif "title" in history.columns:
            history_title_col = "title"
        
        predicthq_title_col = None
        if "show_title" in predicthq.columns:
            predicthq_title_col = "show_title"
        elif "title" in predicthq.columns:
            predicthq_title_col = "title"
        
        if history_title_col and predicthq_title_col:
            left_on.append(history_title_col)
            right_on.append(predicthq_title_col)
        
        # Check for city column
        if "city" in history.columns and "city" in predicthq.columns:
            left_on.append("city")
            right_on.append("city")
        
        if not left_on:
            logger.warning("No common join keys found between history and PredictHQ data.")
            return history
        
        # Perform merge with potentially different column names
        try:
            merged = history.merge(
                predicthq,
                left_on=left_on,
                right_on=right_on,
                how="left",
                suffixes=("", _MERGE_SUFFIX)
            )
            
            # Remove duplicate columns with the suffix
            merged = merged.loc[:, ~merged.columns.str.endswith(_MERGE_SUFFIX)]
            
            logger.info(f"Merged history with PredictHQ data on {left_on}. "
                       f"Result: {len(merged)} rows, {len(merged.columns)} columns.")
            return merged
            
        except Exception as e:
            logger.warning(f"Failed to merge PredictHQ data: {e}. Returning history only.")
            return history
    
    # If join_on is explicitly provided, use it directly
    try:
        merged = history.merge(
            predicthq,
            on=join_on,
            how="left",
            suffixes=("", _MERGE_SUFFIX)
        )
        
        # Remove duplicate columns with the suffix
        merged = merged.loc[:, ~merged.columns.str.endswith(_MERGE_SUFFIX)]
        
        logger.info(f"Merged history with PredictHQ data on {join_on}. "
                   f"Result: {len(merged)} rows, {len(merged.columns)} columns.")
        return merged
        
    except Exception as e:
        logger.warning(f"Failed to merge PredictHQ data: {e}. Returning history only.")
        return history


# =============================================================================
# NEW ECONOMIC DATA LOADERS
# =============================================================================
# Loaders for Nanos Consumer Confidence, Commodity Prices, and CPI data


def load_nanos_consumer_confidence(
    csv_name: str = "economics/nanos_consumer_confidence.csv",
    fallback_empty: bool = True
) -> pd.DataFrame:
    """Load Nanos Consumer Confidence Index data.
    
    The Bloomberg Nanos Canadian Consumer Confidence Index (BNCCI) provides
    weekly tracking of consumer sentiment across Canada with demographic
    breakdowns by region, age, income, and home ownership.
    
    Expected columns:
    - category: Main category (BNCCI, Demographics, Expectations Index, etc.)
    - subcategory: Sub-grouping within category
    - metric: Specific metric name
    - year_or_period: Date or period identifier (YYYY-MM-DD or descriptive)
    - value: Numeric value
    - unit: Unit of measurement (index, percent, count)
    
    Args:
        csv_name: Path to Nanos CSV relative to data directory
        fallback_empty: If True, return empty DataFrame on error
        
    Returns:
        DataFrame with Nanos consumer confidence data (unnamed columns dropped,
        value column converted to numeric)
    """
    path = DATA_DIR / csv_name
    
    try:
        if not path.exists():
            if fallback_empty:
                return pd.DataFrame()
            raise DataLoadError(f"Nanos consumer confidence file not found: {path}")
        
        mtime = _get_file_mtime(path)
        df = _load_csv_cached(str(path), mtime).copy()
        
        # Clean the DataFrame: drop unnamed columns and convert value to numeric
        df = _clean_dataframe(df, drop_unnamed=True, numeric_columns=['value'])
        
        return df
        
    except DataLoadError:
        raise
    except Exception as e:
        if fallback_empty:
            warnings.warn(f"Error loading Nanos consumer confidence: {e}. Using empty DataFrame.")
            return pd.DataFrame()
        raise DataLoadError(f"Error loading Nanos consumer confidence from {path}: {e}")


def load_nanos_better_off(
    csv_name: str = "economics/nanos_better_off.csv",
    fallback_empty: bool = True
) -> pd.DataFrame:
    """Load Nanos Better Off survey data.
    
    Survey data on consumer financial outlook, cost of living impacts,
    and housing worry - useful for understanding consumer spending patterns.
    
    Expected columns:
    - category: Main category (Cost of living, Future standard of living, etc.)
    - subcategory: Regional or demographic breakdown
    - metric: Specific metric name
    - period: Survey period (e.g., Sep-25)
    - value: Numeric value
    - unit: Unit of measurement
    
    Args:
        csv_name: Path to Nanos Better Off CSV relative to data directory
        fallback_empty: If True, return empty DataFrame on error
        
    Returns:
        DataFrame with Nanos Better Off survey data (unnamed columns dropped,
        value column converted to numeric)
    """
    path = DATA_DIR / csv_name
    
    try:
        if not path.exists():
            if fallback_empty:
                return pd.DataFrame()
            raise DataLoadError(f"Nanos Better Off file not found: {path}")
        
        mtime = _get_file_mtime(path)
        df = _load_csv_cached(str(path), mtime).copy()
        
        # Clean the DataFrame: drop unnamed columns and convert value to numeric
        df = _clean_dataframe(df, drop_unnamed=True, numeric_columns=['value'])
        
        return df
        
    except DataLoadError:
        raise
    except Exception as e:
        if fallback_empty:
            warnings.warn(f"Error loading Nanos Better Off data: {e}. Using empty DataFrame.")
            return pd.DataFrame()
        raise DataLoadError(f"Error loading Nanos Better Off data from {path}: {e}")


def load_nanos_arts_donors(
    csv_name: str = "audiences/nanos_arts_donors.csv",
    fallback_empty: bool = True
) -> pd.DataFrame:
    """Load Nanos Arts Donors research data.
    
    Contains data on arts giving patterns, donor demographics, and 
    donation trends - useful for understanding arts audience engagement.
    
    Expected columns:
    - section: Main section (e.g., Annual giving breakdown)
    - subcategory: Subsection (e.g., Arts share)
    - metric: Specific metric name (e.g., Avg %)
    - year_or_period: Year or period of observation
    - value: Numeric value
    - unit: Unit of measurement (e.g., percent)
    - notes: Additional notes
    - source_page: Source page reference
    
    Args:
        csv_name: Path to Nanos arts donors CSV relative to data directory
        fallback_empty: If True, return empty DataFrame on error
        
    Returns:
        DataFrame with arts giving breakdown by year (year, res__arts_share_giving)
    """
    path = DATA_DIR / csv_name
    
    try:
        if not path.exists():
            if fallback_empty:
                return pd.DataFrame()
            raise DataLoadError(f"Nanos Arts Donors file not found: {path}")
        
        mtime = _get_file_mtime(path)
        df = _load_csv_cached(str(path), mtime).copy()
        
        # Clean the DataFrame
        df = _clean_dataframe(df, drop_unnamed=True, numeric_columns=['value'])
        
        # Filter for overall arts share metric (Annual giving breakdown -> Arts share -> Avg %)
        if 'section' in df.columns and 'subcategory' in df.columns and 'metric' in df.columns:
            arts_share = df[
                (df['section'] == 'Annual giving breakdown') &
                (df['subcategory'] == 'Arts share') &
                (df['metric'] == 'Avg %')
            ].copy()
            
            # Create tidy format with year and arts_share_giving
            if not arts_share.empty and 'year_or_period' in arts_share.columns:
                result = pd.DataFrame({
                    'year': pd.to_numeric(arts_share['year_or_period'], errors='coerce'),
                    'res__arts_share_giving': arts_share['value'].values
                })
                result = result.dropna(subset=['year'])
                result['year'] = result['year'].astype(int)
                return result
        
        return pd.DataFrame()
        
    except DataLoadError:
        raise
    except Exception as e:
        if fallback_empty:
            warnings.warn(f"Error loading Nanos Arts Donors data: {e}. Using empty DataFrame.")
            return pd.DataFrame()
        raise DataLoadError(f"Error loading Nanos Arts Donors data from {path}: {e}")


def load_commodity_price_index(
    csv_name: str = "economics/commodity_price_index.csv",
    fallback_empty: bool = True
) -> pd.DataFrame:
    """Load Bank of Canada Commodity Price Index data.
    
    Contains commodity price indices including Energy (A.ENER) which is
    particularly relevant for Alberta's economy and consumer sentiment.
    
    Expected columns:
    - date: Date of observation (YYYY-MM-DD)
    - A.BCPI: Total commodity price index
    - A.BCNE: Non-energy commodities
    - A.ENER: Energy commodities (key for Alberta)
    - A.MTLS: Metals
    - A.FOPR: Forestry products
    - A.AGRI: Agriculture
    - A.FISH: Fisheries
    
    Args:
        csv_name: Path to commodity price CSV relative to data directory
        fallback_empty: If True, return empty DataFrame on error
        
    Returns:
        DataFrame with commodity price indices
    """
    path = DATA_DIR / csv_name
    
    try:
        if not path.exists():
            if fallback_empty:
                return pd.DataFrame()
            raise DataLoadError(f"Commodity price index file not found: {path}")
        
        mtime = _get_file_mtime(path)
        df = _load_csv_cached(str(path), mtime).copy()
        
        # Parse date column if present
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        return df
        
    except DataLoadError:
        raise
    except Exception as e:
        if fallback_empty:
            warnings.warn(f"Error loading commodity price index: {e}. Using empty DataFrame.")
            return pd.DataFrame()
        raise DataLoadError(f"Error loading commodity price index from {path}: {e}")


def load_boc_cpi_monthly(
    csv_name: str = "economics/boc_cpi_monthly.csv",
    fallback_empty: bool = True
) -> pd.DataFrame:
    """Load Bank of Canada monthly CPI data.
    
    Consumer Price Index data for calculating inflation adjustment factors.
    
    Expected columns:
    - date: Date of observation (YYYY-MM-DD)
    - V41690973: CPI All-items (Canada)
    - V41690914: CPI All-items seasonally adjusted
    - STATIC_TOTALCPICHANGE: Year-over-year CPI change percentage
    - CPI_TRIM, CPI_MEDIAN, CPI_COMMON: Core inflation measures
    
    Args:
        csv_name: Path to CPI CSV relative to data directory
        fallback_empty: If True, return empty DataFrame on error
        
    Returns:
        DataFrame with monthly CPI data
    """
    path = DATA_DIR / csv_name
    
    try:
        if not path.exists():
            if fallback_empty:
                return pd.DataFrame()
            raise DataLoadError(f"BOC CPI monthly file not found: {path}")
        
        mtime = _get_file_mtime(path)
        df = _load_csv_cached(str(path), mtime).copy()
        
        # Parse date column if present
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        return df
        
    except DataLoadError:
        raise
    except Exception as e:
        if fallback_empty:
            warnings.warn(f"Error loading BOC CPI monthly: {e}. Using empty DataFrame.")
            return pd.DataFrame()
        raise DataLoadError(f"Error loading BOC CPI monthly from {path}: {e}")


def load_census_data(
    city: str,
    csv_name: Optional[str] = None,
    fallback_empty: bool = True
) -> pd.DataFrame:
    """Load census data for a specific city.
    
    2021 Census data including population, demographics, income, housing.
    
    Args:
        city: City name ('Calgary' or 'Edmonton')
        csv_name: Optional override for CSV path
        fallback_empty: If True, return empty DataFrame on error
        
    Returns:
        DataFrame with census data for the specified city
    """
    city_lower = city.lower()
    if csv_name is None:
        csv_name = f"demographics/{city_lower}_census_2021.csv"
    
    path = DATA_DIR / csv_name
    
    try:
        if not path.exists():
            if fallback_empty:
                return pd.DataFrame()
            raise DataLoadError(f"Census file not found for {city}: {path}")
        
        mtime = _get_file_mtime(path)
        df = pd.read_csv(str(path))
        return df
        
    except DataLoadError:
        raise
    except Exception as e:
        if fallback_empty:
            warnings.warn(f"Error loading census data for {city}: {e}. Using empty DataFrame.")
            return pd.DataFrame()
        raise DataLoadError(f"Error loading census data for {city} from {path}: {e}")


# =============================================================================
# DATA REGISTRY VALIDATION
# =============================================================================


@dataclass
class DataRegistryReport:
    """Report from data registry validation."""
    source_name: str
    path: str
    exists: bool
    row_count: int
    column_count: int
    date_coverage_start: Optional[str]
    date_coverage_end: Optional[str]
    null_counts: dict
    outlier_flags: dict
    validation_errors: list
    validation_warnings: list


def validate_data_source(
    path: str,
    expected_columns: Optional[list] = None,
    date_column: Optional[str] = None,
    required_columns: Optional[list] = None
) -> DataRegistryReport:
    """Validate a data source file and return a report.
    
    Checks:
    - File existence
    - Schema validation (columns, dtypes)
    - Date coverage
    - Null value counts
    - Basic outlier detection
    
    Args:
        path: Path to the data file
        expected_columns: List of expected column names
        date_column: Name of the date column for coverage analysis
        required_columns: Columns that must not be null
        
    Returns:
        DataRegistryReport with validation results
    """
    source_name = Path(path).name
    full_path = DATA_DIR / path if not Path(path).is_absolute() else Path(path)
    
    report = DataRegistryReport(
        source_name=source_name,
        path=str(full_path),
        exists=False,
        row_count=0,
        column_count=0,
        date_coverage_start=None,
        date_coverage_end=None,
        null_counts={},
        outlier_flags={},
        validation_errors=[],
        validation_warnings=[]
    )
    
    # Check file existence
    if not full_path.exists():
        report.validation_errors.append(f"File not found: {full_path}")
        return report
    
    report.exists = True
    
    try:
        df = pd.read_csv(str(full_path))
        report.row_count = len(df)
        report.column_count = len(df.columns)
        
        # Normalize column names for checking
        df.columns = [c.strip().lower().replace(' ', '_').replace('-', '_') for c in df.columns]
        
        # Check expected columns
        if expected_columns:
            expected_lower = [c.lower().replace(' ', '_').replace('-', '_') for c in expected_columns]
            missing = set(expected_lower) - set(df.columns)
            if missing:
                report.validation_warnings.append(f"Missing expected columns: {missing}")
        
        # Check required columns for nulls
        if required_columns:
            required_lower = [c.lower().replace(' ', '_').replace('-', '_') for c in required_columns]
            for col in required_lower:
                if col in df.columns:
                    null_count = df[col].isna().sum()
                    if null_count > 0:
                        report.validation_errors.append(f"Required column '{col}' has {null_count} null values")
        
        # Calculate null counts for all columns
        for col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                report.null_counts[col] = int(null_count)
        
        # Check date coverage if date column specified
        if date_column:
            date_col_lower = date_column.lower().replace(' ', '_').replace('-', '_')
            if date_col_lower in df.columns:
                try:
                    dates = pd.to_datetime(df[date_col_lower], errors='coerce')
                    valid_dates = dates.dropna()
                    if len(valid_dates) > 0:
                        report.date_coverage_start = str(valid_dates.min().date())
                        report.date_coverage_end = str(valid_dates.max().date())
                except Exception:
                    report.validation_warnings.append(f"Could not parse dates in column '{date_column}'")
        
        # Basic outlier detection for numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols[:10]:  # Limit to first 10 numeric columns
            try:
                values = df[col].dropna()
                if len(values) > 10:
                    q1, q3 = values.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 3 * iqr
                    upper_bound = q3 + 3 * iqr
                    outliers = ((values < lower_bound) | (values > upper_bound)).sum()
                    if outliers > 0:
                        report.outlier_flags[col] = int(outliers)
            except Exception:
                pass
                
    except Exception as e:
        report.validation_errors.append(f"Error reading file: {str(e)}")
    
    return report


def generate_data_registry_report(
    output_dir: str = "artifacts/data_registry",
    sources_config: Optional[str] = None
) -> tuple:
    """Generate a comprehensive data registry validation report.
    
    Args:
        output_dir: Directory to write report files
        sources_config: Optional path to data sources CSV config
        
    Returns:
        Tuple of (DataFrame with report data, Markdown report string)
    """
    from datetime import datetime
    from config.registry import load_data_sources
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data sources registry
    sources_df = load_data_sources()
    
    reports = []
    
    # Define validation configs for each source type
    validation_configs = {
        'nanos_consumer_confidence.csv': {
            'expected_columns': ['category', 'subcategory', 'metric', 'year_or_period', 'value', 'unit'],
            'date_column': 'year_or_period'
        },
        'nanos_better_off.csv': {
            'expected_columns': ['category', 'subcategory', 'metric', 'period', 'value', 'unit'],
            'date_column': 'period'
        },
        'commodity_price_index.csv': {
            'expected_columns': ['date', 'A.BCPI', 'A.ENER'],
            'date_column': 'date',
            'required_columns': ['date', 'A.ENER']
        },
        'boc_cpi_monthly.csv': {
            'expected_columns': ['date', 'V41690973', 'STATIC_TOTALCPICHANGE'],
            'date_column': 'date',
            'required_columns': ['date']
        },
        'calgary_census_2021.csv': {
            'date_column': None
        },
        'edmonton_census_2021.csv': {
            'date_column': None
        }
    }
    
    # Validate each source that has a Path defined
    for _, row in sources_df.iterrows():
        path = row.get('Path', '')
        if pd.isna(path) or not path:
            continue
            
        source_file = Path(path).name
        config = validation_configs.get(source_file, {})
        
        report = validate_data_source(
            path=path,
            expected_columns=config.get('expected_columns'),
            date_column=config.get('date_column'),
            required_columns=config.get('required_columns')
        )
        
        reports.append({
            'source_name': report.source_name,
            'path': report.path,
            'exists': report.exists,
            'row_count': report.row_count,
            'column_count': report.column_count,
            'date_start': report.date_coverage_start,
            'date_end': report.date_coverage_end,
            'null_columns': len(report.null_counts),
            'outlier_columns': len(report.outlier_flags),
            'errors': len(report.validation_errors),
            'warnings': len(report.validation_warnings),
            'error_details': '; '.join(report.validation_errors),
            'warning_details': '; '.join(report.validation_warnings)
        })
    
    # Create DataFrame report
    report_df = pd.DataFrame(reports)
    
    # Generate Markdown report
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    md_lines = [
        f"# Data Registry Validation Report",
        f"",
        f"Generated: {timestamp}",
        f"",
        f"## Summary",
        f"",
        f"- Total sources validated: {len(reports)}",
        f"- Sources found: {sum(1 for r in reports if r['exists'])}",
        f"- Sources with errors: {sum(1 for r in reports if r['errors'] > 0)}",
        f"- Sources with warnings: {sum(1 for r in reports if r['warnings'] > 0)}",
        f"",
        f"## Source Details",
        f""
    ]
    
    for r in reports:
        status = "✅" if r['exists'] and r['errors'] == 0 else "⚠️" if r['exists'] else "❌"
        md_lines.extend([
            f"### {status} {r['source_name']}",
            f"",
            f"- **Path**: `{r['path']}`",
            f"- **Exists**: {r['exists']}",
            f"- **Rows**: {r['row_count']:,}",
            f"- **Columns**: {r['column_count']}",
            f"- **Date Range**: {r['date_start']} to {r['date_end']}" if r['date_start'] else "- **Date Range**: N/A",
            f"- **Columns with Nulls**: {r['null_columns']}",
            f"- **Columns with Outliers**: {r['outlier_columns']}",
        ])
        
        if r['error_details']:
            md_lines.append(f"- **Errors**: {r['error_details']}")
        if r['warning_details']:
            md_lines.append(f"- **Warnings**: {r['warning_details']}")
        
        md_lines.append("")
    
    md_report = "\n".join(md_lines)
    
    # Write outputs
    csv_path = output_path / "data_registry_report.csv"
    md_path = output_path / "data_registry_report.md"
    
    report_df.to_csv(csv_path, index=False)
    with open(md_path, 'w') as f:
        f.write(md_report)
    
    logger.info(f"Data registry report written to {output_path}")
    
    return report_df, md_report


# =============================================================================
# WEATHER DATA LOADERS
# =============================================================================
# These functions load daily weather data for Calgary and Edmonton.
# Weather data can be used to adjust ticket estimates based on conditions
# that may affect attendance (extreme cold, snowfall, etc.).


def load_weather_calgary(
    csv_name: str = "environment/weatherstats_calgary_daily.csv",
    fallback_empty: bool = True
) -> pd.DataFrame:
    """Load daily weather data for Calgary.
    
    This file contains historical weather observations from Calgary including
    temperature, precipitation, wind, and other conditions that may affect
    attendance at live events.
    
    Expected columns (key ones for scoring):
    - date: Date of observation (YYYY-MM-DD)
    - min_temperature: Minimum temperature (°C)
    - avg_temperature: Average temperature (°C)
    - max_temperature: Maximum temperature (°C)
    - precipitation: Total precipitation (mm)
    - snow: Snowfall (cm)
    - snow_on_ground: Snow depth on ground (cm)
    - max_wind_speed: Maximum wind speed (km/h)
    - min_windchill: Minimum wind chill factor
    
    Args:
        csv_name: Path to Calgary weather CSV relative to data directory
        fallback_empty: If True, return empty DataFrame on error
        
    Returns:
        DataFrame with Calgary daily weather data
    """
    path = DATA_DIR / csv_name
    
    try:
        if not path.exists():
            if fallback_empty:
                return pd.DataFrame()
            raise DataLoadError(f"Calgary weather file not found: {path}")
        
        mtime = _get_file_mtime(path)
        df = _load_csv_cached(str(path), mtime).copy()
        
        # Parse date column if present
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Add city identifier
        df['city'] = 'Calgary'
        
        return df
        
    except DataLoadError:
        raise
    except Exception as e:
        if fallback_empty:
            warnings.warn(f"Error loading Calgary weather: {e}. Using empty DataFrame.")
            return pd.DataFrame()
        raise DataLoadError(f"Error loading Calgary weather from {path}: {e}")


def load_weather_edmonton(
    csv_name: str = "environment/weatherstats_edmonton_daily.csv",
    fallback_empty: bool = True
) -> pd.DataFrame:
    """Load daily weather data for Edmonton.
    
    This file contains historical weather observations from Edmonton including
    temperature, precipitation, wind, and other conditions that may affect
    attendance at live events.
    
    Expected columns (key ones for scoring):
    - date: Date of observation (YYYY-MM-DD)
    - min_temperature: Minimum temperature (°C)
    - avg_temperature: Average temperature (°C)
    - max_temperature: Maximum temperature (°C)
    - precipitation: Total precipitation (mm)
    - snow: Snowfall (cm)
    - snow_on_ground: Snow depth on ground (cm)
    - max_wind_speed: Maximum wind speed (km/h)
    - min_windchill: Minimum wind chill factor
    
    Args:
        csv_name: Path to Edmonton weather CSV relative to data directory
        fallback_empty: If True, return empty DataFrame on error
        
    Returns:
        DataFrame with Edmonton daily weather data
    """
    path = DATA_DIR / csv_name
    
    try:
        if not path.exists():
            if fallback_empty:
                return pd.DataFrame()
            raise DataLoadError(f"Edmonton weather file not found: {path}")
        
        mtime = _get_file_mtime(path)
        df = _load_csv_cached(str(path), mtime).copy()
        
        # Parse date column if present
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Add city identifier
        df['city'] = 'Edmonton'
        
        return df
        
    except DataLoadError:
        raise
    except Exception as e:
        if fallback_empty:
            warnings.warn(f"Error loading Edmonton weather: {e}. Using empty DataFrame.")
            return pd.DataFrame()
        raise DataLoadError(f"Error loading Edmonton weather from {path}: {e}")


def load_weather_all_cities(fallback_empty: bool = True) -> pd.DataFrame:
    """Load and combine weather data for both Calgary and Edmonton.
    
    Returns:
        Combined DataFrame with weather data for both cities, with 'city' column
    """
    calgary_df = load_weather_calgary(fallback_empty=fallback_empty)
    edmonton_df = load_weather_edmonton(fallback_empty=fallback_empty)
    
    if calgary_df.empty and edmonton_df.empty:
        return pd.DataFrame()
    
    if calgary_df.empty:
        return edmonton_df
    
    if edmonton_df.empty:
        return calgary_df
    
    return pd.concat([calgary_df, edmonton_df], ignore_index=True)


def get_weather_impact_factor(
    run_date: Optional[pd.Timestamp] = None,
    city: Optional[str] = None,
    min_temp_threshold: float = -20.0,
    extreme_cold_factor: float = 0.90,
    heavy_snow_threshold: float = 10.0,
    heavy_snow_factor: float = 0.95,
    min_factor: float = 0.85,
    max_factor: float = 1.05
) -> float:
    """Calculate a weather impact factor for ticket estimation.
    
    This factor adjusts ticket estimates based on weather conditions
    on or around the show date. Extreme cold or heavy snowfall may
    reduce attendance, while mild conditions have neutral impact.
    
    Args:
        run_date: Date of the planned show run (defaults to current date)
        city: City name (Calgary, Edmonton, or None for average)
        min_temp_threshold: Temperature below which cold penalty applies (°C)
        extreme_cold_factor: Multiplier for extreme cold days
        heavy_snow_threshold: Snowfall above which snow penalty applies (cm)
        heavy_snow_factor: Multiplier for heavy snow days
        min_factor: Minimum allowed factor (floor)
        max_factor: Maximum allowed factor (ceiling)
        
    Returns:
        Weather impact factor (1.0 = neutral, <1.0 = adverse conditions)
    """
    if run_date is None:
        run_date = pd.Timestamp.now()
    
    # Load weather data for the appropriate city
    if city and 'calg' in city.lower():
        weather_df = load_weather_calgary()
    elif city and 'edm' in city.lower():
        weather_df = load_weather_edmonton()
    else:
        weather_df = load_weather_all_cities()
    
    if weather_df.empty or 'date' not in weather_df.columns:
        return 1.0  # No weather data available, neutral factor
    
    # Filter to city if specified
    if city and 'city' in weather_df.columns:
        city_key = 'Calgary' if 'calg' in city.lower() else 'Edmonton' if 'edm' in city.lower() else None
        if city_key:
            weather_df = weather_df[weather_df['city'] == city_key]
    
    # Find weather data for the run date (or nearby dates)
    weather_df = weather_df.sort_values('date', ascending=False)
    
    # Look for exact date match first
    exact_match = weather_df[weather_df['date'] == run_date]
    if exact_match.empty:
        # Use most recent available data as proxy (for future dates)
        if not weather_df.empty:
            exact_match = weather_df.head(1)
    
    if exact_match.empty:
        return 1.0  # No weather data available
    
    row = exact_match.iloc[0]
    factor = 1.0
    
    # Check for extreme cold
    min_temp = row.get('min_temperature', None)
    if min_temp is not None and pd.notna(min_temp):
        try:
            min_temp = float(min_temp)
            if min_temp < min_temp_threshold:
                factor *= extreme_cold_factor
        except (ValueError, TypeError):
            pass
    
    # Check for heavy snowfall
    snow = row.get('snow', None)
    if snow is not None and pd.notna(snow):
        try:
            snow = float(snow)
            if snow > heavy_snow_threshold:
                factor *= heavy_snow_factor
        except (ValueError, TypeError):
            pass
    
    # Clip to min/max range
    return float(max(min_factor, min(max_factor, factor)))


def get_monthly_weather_summary(
    month: int,
    city: Optional[str] = None
) -> dict:
    """Get summary weather statistics for a given month.
    
    Useful for understanding typical weather conditions during a planned
    show run month.
    
    Args:
        month: Month number (1-12)
        city: City name (Calgary, Edmonton, or None for combined)
        
    Returns:
        Dictionary with summary statistics:
        - avg_temp_mean: Average of average temperatures
        - min_temp_mean: Average of minimum temperatures
        - snow_mean: Average snowfall
        - precipitation_mean: Average precipitation
        - extreme_cold_days: Count of days below -20°C
    """
    if city and 'calg' in city.lower():
        weather_df = load_weather_calgary()
    elif city and 'edm' in city.lower():
        weather_df = load_weather_edmonton()
    else:
        weather_df = load_weather_all_cities()
    
    if weather_df.empty or 'date' not in weather_df.columns:
        return {}
    
    # Filter to the specified month
    weather_df['month'] = weather_df['date'].dt.month
    month_data = weather_df[weather_df['month'] == month]
    
    if month_data.empty:
        return {}
    
    result = {}
    
    # Average temperature statistics
    if 'avg_temperature' in month_data.columns:
        avg_temps = pd.to_numeric(month_data['avg_temperature'], errors='coerce')
        result['avg_temp_mean'] = float(avg_temps.mean()) if avg_temps.notna().any() else None
    
    if 'min_temperature' in month_data.columns:
        min_temps = pd.to_numeric(month_data['min_temperature'], errors='coerce')
        result['min_temp_mean'] = float(min_temps.mean()) if min_temps.notna().any() else None
        result['extreme_cold_days'] = int((min_temps < -20).sum())
    
    if 'snow' in month_data.columns:
        snow = pd.to_numeric(month_data['snow'], errors='coerce')
        result['snow_mean'] = float(snow.mean()) if snow.notna().any() else None
    
    if 'precipitation' in month_data.columns:
        precip = pd.to_numeric(month_data['precipitation'], errors='coerce')
        result['precipitation_mean'] = float(precip.mean()) if precip.notna().any() else None
    
    return result


# =============================================================================
# LIVE ANALYTICS DATA LOADER
# =============================================================================
# Functions for loading and using audience analytics data for segment adjustments.


def load_live_analytics_raw(
    csv_name: str = "audiences/live_analytics.csv",
    fallback_empty: bool = True
) -> pd.DataFrame:
    """Load raw live analytics data.
    
    This data contains audience demographic and behavioral breakdowns by
    show category, useful for understanding segment composition.
    
    Args:
        csv_name: Path to live analytics CSV relative to data directory
        fallback_empty: If True, return empty DataFrame on error
        
    Returns:
        DataFrame with raw live analytics data (empty unnamed columns dropped)
    """
    path = DATA_DIR / csv_name
    
    try:
        if not path.exists():
            if fallback_empty:
                return pd.DataFrame()
            raise DataLoadError(f"Live analytics file not found: {path}")
        
        # This file has a non-standard format, load with header on first row
        df = pd.read_csv(str(path))
        
        # Clean the DataFrame: drop only empty unnamed columns (preserve structure)
        df = _clean_dataframe(df, drop_unnamed=True, drop_empty_unnamed_only=True)
        
        return df
        
    except DataLoadError:
        raise
    except Exception as e:
        if fallback_empty:
            warnings.warn(f"Error loading live analytics: {e}. Using empty DataFrame.")
            return pd.DataFrame()
        raise DataLoadError(f"Error loading live analytics from {path}: {e}")


def get_live_analytics_category_factors() -> dict:
    """Parse live analytics data to extract category-based adjustment factors.
    
    The live_analytics.csv contains audience metrics by show category.
    The file has a complex structure with:
    - Columns 2-7: CLIENT EVENT SUMMARY (percentages)
    - Columns 9-15: INDEX REPORT (indices where 100 = average)
    
    This function extracts index values to compute engagement factors.
    
    Returns:
        Dictionary mapping category names to adjustment factors:
        {
            'pop_ip': {'engagement_factor': 1.05, 'high_spender_index': 164, ...},
            'classic_romance': {'engagement_factor': 1.02, 'high_spender_index': 145, ...},
            ...
        }
    """
    df = load_live_analytics_raw()
    
    if df.empty:
        return {}
    
    # The INDEX REPORT section has indices in columns 9-14
    # Column 9 = Pop Mus Bal (pop_ip)
    # Column 10 = Classic Bal (classic_romance)
    # Column 11 = CM Bill (contemporary)
    # Column 12 = Family Bal (family_classic)
    # Column 13 = CSNA (romantic_tragedy - we skip this)
    # Column 14 = Contemp Narr (dramatic)
    index_category_mapping = {
        9: 'pop_ip',
        10: 'classic_romance',
        11: 'contemporary',
        12: 'family_classic',
        # 13 is CSNA - skip
        14: 'dramatic',
    }
    
    # Also map the CLIENT EVENT SUMMARY percentages (columns 2-7)
    pct_category_mapping = {
        2: 'pop_ip',
        3: 'classic_romance',
        4: 'contemporary',
        5: 'family_classic',
        6: 'romantic_tragedy',
        7: 'dramatic'
    }
    
    result = {}
    
    try:
        for _, row in df.iterrows():
            row_data = row.tolist()
            if len(row_data) < 15:
                continue
            
            # Check for key metrics in the first two columns
            first_col = str(row_data[0]).lower() if pd.notna(row_data[0]) else ''
            second_col = str(row_data[1]).lower() if len(row_data) > 1 and pd.notna(row_data[1]) else ''
            
            # Extract high spender index from INDEX REPORT section (column 9+)
            if 'high spender' in second_col or 'high_spenders' in first_col:
                for col_idx, category in index_category_mapping.items():
                    if col_idx < len(row_data) and pd.notna(row_data[col_idx]):
                        try:
                            val = str(row_data[col_idx]).replace(',', '').replace('n/a', '')
                            if val:
                                index = float(val)
                                if category not in result:
                                    result[category] = {}
                                result[category]['high_spender_index'] = index
                        except (ValueError, TypeError):
                            pass
            
            # Extract active buyers index from INDEX REPORT section
            if 'active buyer' in second_col:
                for col_idx, category in index_category_mapping.items():
                    if col_idx < len(row_data) and pd.notna(row_data[col_idx]):
                        try:
                            val = str(row_data[col_idx]).replace(',', '').replace('n/a', '')
                            if val:
                                index = float(val)
                                if category not in result:
                                    result[category] = {}
                                result[category]['active_buyer_index'] = index
                        except (ValueError, TypeError):
                            pass
            
            # Extract repeat buyers index from INDEX REPORT section
            if 'repeat buyer' in second_col:
                for col_idx, category in index_category_mapping.items():
                    if col_idx < len(row_data) and pd.notna(row_data[col_idx]):
                        try:
                            val = str(row_data[col_idx]).replace(',', '').replace('n/a', '')
                            if val:
                                index = float(val)
                                if category not in result:
                                    result[category] = {}
                                result[category]['repeat_buyer_index'] = index
                        except (ValueError, TypeError):
                            pass
            
            # Extract arts attendance index (dim_13_pct Live: Major: Arts)
            if 'major: arts' in second_col or 'major arts' in second_col:
                for col_idx, category in index_category_mapping.items():
                    if col_idx < len(row_data) and pd.notna(row_data[col_idx]):
                        try:
                            val = str(row_data[col_idx]).replace(',', '').replace('n/a', '')
                            if val:
                                index = float(val)
                                if category not in result:
                                    result[category] = {}
                                result[category]['arts_attendance_index'] = index
                        except (ValueError, TypeError):
                            pass
        
        # Also add romantic_tragedy from the percentage columns (map from family_classic-ish)
        if 'family_classic' in result and 'romantic_tragedy' not in result:
            result['romantic_tragedy'] = result['family_classic'].copy()
        
        # Calculate engagement factor based on available indices
        # Higher indices (>100) indicate stronger engagement relative to average
        for category in result:
            data = result[category]
            active_idx = data.get('active_buyer_index', 100)
            repeat_idx = data.get('repeat_buyer_index', 100)
            high_spender_idx = data.get('high_spender_index', 100)
            arts_idx = data.get('arts_attendance_index', 100)
            
            # Engagement factor: weighted average of indices, normalized to 1.0 = 100 index
            # Active and repeat buyers are most relevant for ticket sales
            weighted_index = (
                0.30 * active_idx +
                0.30 * repeat_idx +
                0.25 * high_spender_idx +
                0.15 * arts_idx
            )
            
            # Convert index to factor (100 = 1.0, 120 = 1.05, 80 = 0.95)
            # Apply dampening to avoid extreme adjustments
            raw_factor = weighted_index / 100.0
            engagement = 1.0 + (raw_factor - 1.0) * 0.25  # Dampen by 75%
            engagement = max(0.92, min(1.08, engagement))  # Clip to reasonable range
            
            result[category]['engagement_factor'] = round(engagement, 3)
        
    except Exception as e:
        logger.warning(f"Error parsing live analytics: {e}")
        return {}
    
    return result


def get_category_engagement_factor(category: str) -> float:
    """Get the engagement adjustment factor for a show category.
    
    Based on live analytics data, returns a factor to adjust ticket
    estimates based on the category's audience engagement patterns.
    
    Args:
        category: Show category (e.g., 'pop_ip', 'classic_romance', 'family_classic')
        
    Returns:
        Engagement factor (1.0 = neutral, >1.0 = higher engagement category)
    """
    factors = get_live_analytics_category_factors()
    
    if not factors:
        return 1.0
    
    category_lower = category.lower().strip()
    
    # Direct match
    if category_lower in factors:
        return factors[category_lower].get('engagement_factor', 1.0)
    
    # Map related categories
    category_aliases = {
        'classic_comedy': 'classic_romance',
        'romantic_comedy': 'classic_romance',
        'adult_literary_drama': 'dramatic',
        'contemporary_mixed_bill': 'contemporary',
        'touring_contemporary_company': 'contemporary',
    }
    
    if category_lower in category_aliases:
        mapped = category_aliases[category_lower]
        if mapped in factors:
            return factors[mapped].get('engagement_factor', 1.0)
    
    return 1.0
