from pathlib import Path
from functools import lru_cache
from typing import Optional
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
# These functions load external factor files created by the Data Helper app.
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
    
    This file is typically created by the Data Helper app (pages/6_Data_Helper.py).
    It contains external factors that can enhance ML model predictions.
    
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
        DataFrame with audience analytics data
    """
    path = DATA_DIR / csv_name
    
    try:
        if not path.exists():
            if fallback_empty:
                return pd.DataFrame()
            raise DataLoadError(f"Audience analytics file not found: {path}")
        
        mtime = _get_file_mtime(path)
        return _load_csv_cached(str(path), mtime).copy()
        
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
