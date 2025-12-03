"""
Economic feature engineering for ballet show predictions.

This module loads economic indicators (CPI, energy prices, oil, unemployment, consumer confidence)
and merges them onto show data based on temporal proximity using start_date.

Features are designed to capture the economic environment at the time of each show.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import warnings

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data"
ECONOMICS_DIR = DATA_DIR / "economics"


def load_cpi_data() -> pd.DataFrame:
    """
    Load Bank of Canada CPI monthly data.
    
    Returns:
        DataFrame with columns: date, V41690973 (All-items CPI)
    """
    path = ECONOMICS_DIR / "boc_cpi_monthly.csv"
    if not path.exists():
        warnings.warn(f"CPI data not found at {path}")
        return pd.DataFrame()
    
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_commodity_prices() -> pd.DataFrame:
    """
    Load Bank of Canada commodity price index data.
    
    Returns:
        DataFrame with columns: date, A.ENER (Energy index), A.BCPI (Overall index)
    """
    path = ECONOMICS_DIR / "commodity_price_index.csv"
    if not path.exists():
        warnings.warn(f"Commodity price data not found at {path}")
        return pd.DataFrame()
    
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Normalize column names to uppercase for consistency
    df.columns = [c.upper() if c.lower() != 'date' else c for c in df.columns]
    
    return df


def load_oil_prices() -> pd.DataFrame:
    """
    Load Western Canadian Select (WCS) oil price data.
    
    Returns:
        DataFrame with columns: date, wcs_oil_price
    """
    path = ECONOMICS_DIR / "oil_price.csv"
    if not path.exists():
        warnings.warn(f"Oil price data not found at {path}")
        return pd.DataFrame()
    
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_unemployment_data() -> pd.DataFrame:
    """
    Load unemployment rate data for Alberta.
    
    Returns:
        DataFrame with columns: date, unemployment_rate, region
    """
    path = ECONOMICS_DIR / "unemployment_by_city.csv"
    if not path.exists():
        warnings.warn(f"Unemployment data not found at {path}")
        return pd.DataFrame()
    
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter to Alberta only
    df = df[df['region'] == 'Alberta'].copy()
    
    return df


def load_consumer_confidence() -> pd.DataFrame:
    """
    Load Nanos consumer confidence data for prairies.
    
    Returns:
        DataFrame with columns: date, consumer_confidence
    """
    path = ECONOMICS_DIR / "nanos_consumer_confidence.csv"
    if not path.exists():
        warnings.warn(f"Consumer confidence data not found at {path}")
        return pd.DataFrame()
    
    df = pd.read_csv(path)
    
    # Parse year_or_period to datetime (handle various formats)
    try:
        df['date'] = pd.to_datetime(df['year_or_period'], errors='coerce')
    except Exception:
        warnings.warn("Could not parse consumer confidence dates")
        return pd.DataFrame()
    
    # Keep only rows with valid dates
    df = df[df['date'].notna()].copy()
    
    # Use the value column
    df = df[['date', 'value']].rename(columns={'value': 'consumer_confidence'})
    
    return df


def compute_boc_factor(df: pd.DataFrame, date_column: str = 'start_date') -> pd.DataFrame:
    """
    Compute Bank of Canada economic factor based on CPI and commodity prices.
    
    This composite factor captures overall economic conditions from BoC data:
    - CPI (inflation)
    - Energy commodity prices
    
    Args:
        df: DataFrame with show data
        date_column: Name of the date column to use for temporal matching
        
    Returns:
        DataFrame with 'Econ_BocFactor' column added
    """
    if df.empty:
        return df.copy()
    
    if date_column not in df.columns:
        warnings.warn(f"Date column '{date_column}' not found. Setting Econ_BocFactor to 1.0")
        df_out = df.copy()
        df_out['Econ_BocFactor'] = 1.0
        return df_out
    
    df_out = df.copy()
    
    # Load economic data
    cpi_data = load_cpi_data()
    commodity_data = load_commodity_prices()
    
    # Create temporary column for merging
    df_out['_temp_date'] = pd.to_datetime(df_out[date_column])
    
    # Initialize factor
    df_out['Econ_BocFactor'] = 1.0
    
    # Merge CPI (inflation component)
    if not cpi_data.empty:
        # Use All-items CPI (V41690973)
        cpi_subset = cpi_data[['date', 'V41690973']].copy()
        cpi_subset = cpi_subset.sort_values('date')
        
        # Merge asof (backward): use most recent CPI before show date
        df_merged = pd.merge_asof(
            df_out.sort_values('_temp_date'),
            cpi_subset,
            left_on='_temp_date',
            right_on='date',
            direction='backward',
            suffixes=('', '_cpi')
        )
        
        # Normalize CPI to baseline (2020-01-01)
        baseline_cpi = cpi_subset[cpi_subset['date'] >= '2020-01-01']['V41690973'].iloc[0] \
                       if len(cpi_subset[cpi_subset['date'] >= '2020-01-01']) > 0 else 100.0
        
        df_merged['cpi_factor'] = df_merged['V41690973'] / baseline_cpi
        df_out['Econ_BocFactor'] = df_merged['cpi_factor'].fillna(1.0)
    
    # Merge commodity prices (energy component)
    if not commodity_data.empty and 'A.ENER' in commodity_data.columns:
        commodity_subset = commodity_data[['date', 'A.ENER']].copy()
        commodity_subset = commodity_subset.sort_values('date')
        
        # Merge asof (backward)
        df_merged = pd.merge_asof(
            df_out.sort_values('_temp_date'),
            commodity_subset,
            left_on='_temp_date',
            right_on='date',
            direction='backward',
            suffixes=('', '_comm')
        )
        
        # Normalize energy index to baseline
        baseline_energy = commodity_subset[commodity_subset['date'] >= '2020-01-01']['A.ENER'].iloc[0] \
                         if len(commodity_subset[commodity_subset['date'] >= '2020-01-01']) > 0 else 1000.0
        
        df_merged['energy_factor'] = df_merged['A.ENER'] / baseline_energy
        
        # Combine CPI and energy factors (weighted average: 50% each)
        df_out['Econ_BocFactor'] = (
            df_out['Econ_BocFactor'] * 0.5 + 
            df_merged['energy_factor'].fillna(1.0) * 0.5
        )
    
    # Forward fill missing values
    df_out['Econ_BocFactor'] = df_out['Econ_BocFactor'].ffill().fillna(1.0)
    
    # Clean up temporary column
    df_out = df_out.drop(columns=['_temp_date'])
    
    return df_out


def compute_alberta_factor(df: pd.DataFrame, date_column: str = 'start_date') -> pd.DataFrame:
    """
    Compute Alberta-specific economic health factor.
    
    This factor captures Alberta's economic conditions:
    - WCS oil prices (major driver of Alberta economy)
    - Alberta unemployment rate
    
    Args:
        df: DataFrame with show data
        date_column: Name of the date column to use for temporal matching
        
    Returns:
        DataFrame with 'Econ_AlbertaFactor' column added
    """
    if df.empty:
        return df.copy()
    
    if date_column not in df.columns:
        warnings.warn(f"Date column '{date_column}' not found. Setting Econ_AlbertaFactor to 1.0")
        df_out = df.copy()
        df_out['Econ_AlbertaFactor'] = 1.0
        return df_out
    
    df_out = df.copy()
    
    # Load economic data
    oil_data = load_oil_prices()
    unemployment_data = load_unemployment_data()
    
    # Create temporary column for merging
    df_out['_temp_date'] = pd.to_datetime(df_out[date_column])
    
    # Initialize factor
    df_out['Econ_AlbertaFactor'] = 1.0
    
    # Merge oil prices
    if not oil_data.empty:
        oil_subset = oil_data[['date', 'wcs_oil_price']].copy()
        oil_subset = oil_subset.sort_values('date')
        
        # Merge asof (backward)
        df_merged = pd.merge_asof(
            df_out.sort_values('_temp_date'),
            oil_subset,
            left_on='_temp_date',
            right_on='date',
            direction='backward',
            suffixes=('', '_oil')
        )
        
        # Normalize oil price to baseline (2020 average)
        baseline_oil = oil_subset[
            (oil_subset['date'] >= '2020-01-01') & 
            (oil_subset['date'] < '2021-01-01')
        ]['wcs_oil_price'].mean()
        
        if pd.isna(baseline_oil) or baseline_oil == 0:
            baseline_oil = 40.0  # Fallback baseline
        
        df_merged['oil_factor'] = df_merged['wcs_oil_price'] / baseline_oil
        df_out['Econ_AlbertaFactor'] = df_merged['oil_factor'].fillna(1.0)
    
    # Merge unemployment (inverse relationship: higher unemployment = worse economy)
    if not unemployment_data.empty:
        unemp_subset = unemployment_data[['date', 'unemployment_rate']].copy()
        unemp_subset = unemp_subset.sort_values('date')
        
        # Merge asof (backward)
        df_merged = pd.merge_asof(
            df_out.sort_values('_temp_date'),
            unemp_subset,
            left_on='_temp_date',
            right_on='date',
            direction='backward',
            suffixes=('', '_unemp')
        )
        
        # Convert unemployment to factor (inverse: lower unemployment = better)
        # Baseline: 7% unemployment = 1.0 factor
        # Lower unemployment (5%) -> higher factor (1.4)
        # Higher unemployment (10%) -> lower factor (0.7)
        baseline_unemp = 7.0
        df_merged['unemp_factor'] = baseline_unemp / df_merged['unemployment_rate'].clip(lower=1.0)
        
        # Combine oil and unemployment factors (weighted: 60% oil, 40% unemployment)
        df_out['Econ_AlbertaFactor'] = (
            df_out['Econ_AlbertaFactor'] * 0.6 + 
            df_merged['unemp_factor'].fillna(1.0) * 0.4
        )
    
    # Forward fill missing values
    df_out['Econ_AlbertaFactor'] = df_out['Econ_AlbertaFactor'].ffill().fillna(1.0)
    
    # Clean up temporary column
    df_out = df_out.drop(columns=['_temp_date'])
    
    return df_out


def add_economic_features(
    df: pd.DataFrame, 
    date_column: str = 'start_date'
) -> pd.DataFrame:
    """
    Add all economic features to a DataFrame.
    
    This function merges economic indicators onto show data based on temporal proximity:
    - Econ_BocFactor: Bank of Canada composite (CPI + energy prices)
    - Econ_AlbertaFactor: Alberta-specific (oil prices + unemployment)
    
    Args:
        df: DataFrame with show data
        date_column: Name of the date column to use for temporal matching (default: 'start_date')
        
    Returns:
        DataFrame with economic feature columns added (original df is not modified)
        
    Raises:
        ValueError: If the specified date column is not found in the DataFrame
    """
    if df.empty:
        return df.copy()
    
    if date_column not in df.columns:
        raise ValueError(
            f"Column '{date_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Create a copy to avoid modifying the original
    df_out = df.copy()
    
    # Add BoC factor (national economic conditions)
    df_out = compute_boc_factor(df_out, date_column=date_column)
    
    # Add Alberta factor (provincial economic conditions)
    df_out = compute_alberta_factor(df_out, date_column=date_column)
    
    return df_out


def get_feature_names() -> list:
    """
    Get the list of feature names created by this module.
    
    Returns:
        List of economic feature column names
    """
    return ['Econ_BocFactor', 'Econ_AlbertaFactor']
