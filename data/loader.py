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
    csv_name: str = "history_city_sales.csv",
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
    csv_name: str = "baselines.csv",
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


def load_reference_baselines(csv_name: str = "baselines.csv") -> pd.DataFrame:
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
    baselines_path: str = "baselines.csv",
    reference_path: str = "baselines.csv"
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
