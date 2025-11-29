import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import logging

from config.registry import load_feature_inventory

logger = logging.getLogger(__name__)


def apply_registry_renames(df: pd.DataFrame) -> pd.DataFrame:
    """Optionally map raw columns to registry names if needed."""
    # Example mapping (keep simple unless you need it)
    # return df.rename(columns={"total_single_tickets": "total_single_tickets"})
    return df


def derive_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple derived features based on registry guidance.
    
    Note: This app focuses on single ticket estimation only.
    """
    out = df.copy()
    return out


def get_feature_list(theme_filters=None, status=None) -> list[str]:
    """Pull model feature names from the registry."""
    inv = load_feature_inventory()
    df = inv.copy()
    if theme_filters:
        df = df[df["Theme"].isin(theme_filters)]
    if status:
        df = df[df["Status"].str.contains(status, case=False, na=False)]
    return df["Feature Name"].dropna().unique().tolist()


# =============================================================================
# DATE NORMALIZATION UTILITIES
# =============================================================================


def normalize_to_year_month(date_series: pd.Series) -> pd.Series:
    """Convert dates to Year-Month period format (YYYY-MM).
    
    Used for aligning different data sources on a common time granularity.
    
    Args:
        date_series: Series of dates (can be datetime, string, or Period)
        
    Returns:
        Series of Period[M] objects
    """
    if date_series.empty:
        return pd.Series(dtype='period[M]')
    
    # Handle different input types
    try:
        # First try to convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(date_series):
            date_series = pd.to_datetime(date_series, errors='coerce')
        
        # Convert to period
        return date_series.dt.to_period('M')
    except Exception as e:
        logger.warning(f"Error normalizing dates to year-month: {e}")
        return pd.Series(dtype='period[M]')


def snap_show_date_to_month(show_date_series: pd.Series) -> pd.Series:
    """Snap show dates to Year-Month for joining with monthly economic data.
    
    Preserves the original date for label alignment while providing
    a month-level key for feature joins.
    
    Args:
        show_date_series: Series of show/performance dates
        
    Returns:
        Series of Year-Month strings (YYYY-MM)
    """
    if show_date_series.empty:
        return pd.Series(dtype='str')
    
    try:
        dates = pd.to_datetime(show_date_series, errors='coerce')
        return dates.dt.strftime('%Y-%m')
    except Exception as e:
        logger.warning(f"Error snapping show dates to month: {e}")
        return pd.Series(dtype='str')


# =============================================================================
# ECONOMIC DATA JOIN FUNCTIONS
# =============================================================================


def join_consumer_confidence(
    df: pd.DataFrame,
    nanos_df: pd.DataFrame,
    date_column: str = 'show_date',
    target_metric: str = 'This week',
    region: str = 'Prairies'
) -> pd.DataFrame:
    """Join Nanos consumer confidence data to show data.
    
    Forward-fills weekly confidence data to align with show dates.
    Uses Prairies region as the most relevant for Alberta.
    
    Args:
        df: Show/production DataFrame with date column
        nanos_df: Nanos consumer confidence DataFrame
        date_column: Name of the date column in df
        target_metric: Which metric to use (e.g., 'This week')
        region: Which region to use (default 'Prairies' for Alberta)
        
    Returns:
        DataFrame with consumer_confidence feature added
    """
    if nanos_df.empty:
        logger.warning("Empty Nanos data; skipping consumer confidence join")
        return df
    
    out = df.copy()
    
    try:
        # Filter for headline index
        headline = nanos_df[
            (nanos_df['category'] == 'BNCCI') & 
            (nanos_df['subcategory'] == 'Headline Index') &
            (nanos_df['metric'] == target_metric)
        ].copy()
        
        # Also get regional data if available
        regional = nanos_df[
            (nanos_df['category'] == 'Demographics') & 
            (nanos_df['subcategory'] == 'Region') &
            (nanos_df['metric'] == region)
        ].copy()
        
        if not headline.empty:
            # Extract the latest headline value
            latest_confidence = headline['value'].iloc[0] if len(headline) > 0 else 50.0
            out['consumer_confidence_headline'] = latest_confidence
        else:
            out['consumer_confidence_headline'] = 50.0  # neutral baseline
        
        if not regional.empty:
            regional_value = regional['value'].iloc[0] if len(regional) > 0 else 50.0
            out['consumer_confidence_prairies'] = regional_value
        else:
            out['consumer_confidence_prairies'] = out.get('consumer_confidence_headline', 50.0)
        
        logger.info(f"Added consumer confidence features to {len(out)} rows")
        return out
        
    except Exception as e:
        logger.warning(f"Error joining consumer confidence: {e}")
        out['consumer_confidence_headline'] = 50.0
        out['consumer_confidence_prairies'] = 50.0
        return out


def join_energy_index(
    df: pd.DataFrame,
    commodity_df: pd.DataFrame,
    date_column: str = 'show_date'
) -> pd.DataFrame:
    """Join commodity price Energy index to show data.
    
    Uses monthly Energy index (A.ENER) from Bank of Canada commodity data.
    Energy prices strongly correlate with Alberta's economic health.
    
    Args:
        df: Show/production DataFrame with date column
        commodity_df: Commodity price DataFrame with date and A.ENER columns
        date_column: Name of the date column in df
        
    Returns:
        DataFrame with energy_index feature added
    """
    if commodity_df.empty:
        logger.warning("Empty commodity data; skipping energy index join")
        return df
    
    out = df.copy()
    
    try:
        # Ensure date columns are datetime
        if date_column in out.columns:
            out['_show_year_month'] = snap_show_date_to_month(out[date_column])
        else:
            logger.warning(f"Date column '{date_column}' not found")
            out['energy_index'] = np.nan
            return out
        
        # Prepare commodity data
        comm = commodity_df.copy()
        if 'date' in comm.columns:
            comm['date'] = pd.to_datetime(comm['date'], errors='coerce')
            comm['_year_month'] = comm['date'].dt.strftime('%Y-%m')
        else:
            out['energy_index'] = np.nan
            return out
        
        # Get energy column (A.ENER)
        if 'A.ENER' in comm.columns:
            energy_lookup = comm.set_index('_year_month')['A.ENER'].to_dict()
            out['energy_index'] = out['_show_year_month'].map(energy_lookup)
            
            # Forward-fill any missing values with most recent
            if out['energy_index'].isna().any():
                # Get most recent available value
                sorted_comm = comm.dropna(subset=['A.ENER']).sort_values('date')
                if len(sorted_comm) > 0:
                    latest_energy = sorted_comm['A.ENER'].iloc[-1]
                    out['energy_index'] = out['energy_index'].fillna(latest_energy)
        else:
            out['energy_index'] = np.nan
        
        # Clean up temp column
        out = out.drop(columns=['_show_year_month'], errors='ignore')
        
        logger.info(f"Added energy_index feature to {len(out)} rows")
        return out
        
    except Exception as e:
        logger.warning(f"Error joining energy index: {e}")
        out['energy_index'] = np.nan
        return out


def compute_inflation_adjustment_factor(
    df: pd.DataFrame,
    cpi_df: pd.DataFrame,
    date_column: str = 'show_date',
    base_date: str = '2020-01-01'
) -> pd.DataFrame:
    """Compute inflation adjustment factor from CPI data.
    
    Creates an adjustment factor relative to a base date that can be used
    to normalize ticket prices or revenue across time periods.
    
    Formula: inflation_factor = cpi_current / cpi_base
    (Values > 1 indicate prices have risen since base date)
    
    Args:
        df: Show/production DataFrame with date column
        cpi_df: CPI DataFrame with date and V41690973 (CPI All-items) columns
        date_column: Name of the date column in df
        base_date: Reference date for the base CPI value
        
    Returns:
        DataFrame with inflation_adjustment_factor feature added
    """
    if cpi_df.empty:
        logger.warning("Empty CPI data; skipping inflation factor computation")
        return df
    
    out = df.copy()
    
    try:
        # Prepare CPI data
        cpi = cpi_df.copy()
        if 'date' not in cpi.columns:
            out['inflation_adjustment_factor'] = 1.0
            return out
            
        cpi['date'] = pd.to_datetime(cpi['date'], errors='coerce')
        cpi = cpi.dropna(subset=['date'])
        
        # Get CPI column - V41690973 is CPI All-items
        cpi_col = 'V41690973'
        if cpi_col not in cpi.columns:
            # Try alternative column names
            for alt in ['v41690973', 'CPI', 'cpi']:
                if alt in cpi.columns:
                    cpi_col = alt
                    break
            else:
                out['inflation_adjustment_factor'] = 1.0
                return out
        
        # Get base CPI value
        base_dt = pd.to_datetime(base_date)
        base_cpi_row = cpi[cpi['date'] <= base_dt].sort_values('date')
        if len(base_cpi_row) > 0:
            base_cpi = base_cpi_row[cpi_col].iloc[-1]
        else:
            # Use first available value
            base_cpi = cpi[cpi_col].iloc[0]
        
        # Create year-month lookup for CPI
        cpi['_year_month'] = cpi['date'].dt.strftime('%Y-%m')
        cpi_lookup = cpi.set_index('_year_month')[cpi_col].to_dict()
        
        # Map show dates to CPI values
        if date_column in out.columns:
            out['_show_year_month'] = snap_show_date_to_month(out[date_column])
            out['_current_cpi'] = out['_show_year_month'].map(cpi_lookup)
            
            # Forward-fill missing CPI with most recent
            sorted_cpi = cpi.dropna(subset=[cpi_col]).sort_values('date')
            if len(sorted_cpi) > 0:
                latest_cpi = sorted_cpi[cpi_col].iloc[-1]
                out['_current_cpi'] = out['_current_cpi'].fillna(latest_cpi)
            
            # Compute inflation factor
            out['inflation_adjustment_factor'] = out['_current_cpi'] / base_cpi
            
            # Clean up temp columns
            out = out.drop(columns=['_show_year_month', '_current_cpi'], errors='ignore')
        else:
            out['inflation_adjustment_factor'] = 1.0
        
        logger.info(f"Added inflation_adjustment_factor to {len(out)} rows")
        return out
        
    except Exception as e:
        logger.warning(f"Error computing inflation adjustment factor: {e}")
        out['inflation_adjustment_factor'] = 1.0
        return out


# =============================================================================
# CITY SEGMENTATION FEATURES
# =============================================================================


def add_city_segmentation_features(
    df: pd.DataFrame,
    city_column: str = 'city',
    calgary_census: Optional[pd.DataFrame] = None,
    edmonton_census: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Add city-level segmentation features.
    
    Creates binary/one-hot encoding for cities and adds demographic features
    from census data when available.
    
    Args:
        df: Show/production DataFrame with city column
        city_column: Name of the city column
        calgary_census: Optional Calgary census DataFrame
        edmonton_census: Optional Edmonton census DataFrame
        
    Returns:
        DataFrame with city segmentation features added
    """
    out = df.copy()
    
    # Create city binary features
    if city_column in out.columns:
        city_vals = out[city_column].str.lower().str.strip()
        out['city_calgary'] = (city_vals.str.contains('calgary', na=False)).astype(int)
        out['city_edmonton'] = (city_vals.str.contains('edmonton', na=False)).astype(int)
    else:
        out['city_calgary'] = 0
        out['city_edmonton'] = 0
    
    # Add census-based features if available
    # Extract key demographic metrics that won't leak future info
    try:
        calgary_pop = _extract_census_population(calgary_census) if calgary_census is not None else 1481806
        edmonton_pop = _extract_census_population(edmonton_census) if edmonton_census is not None else 1418118
        calgary_income = _extract_census_median_income(calgary_census) if calgary_census is not None else 100000
        edmonton_income = _extract_census_median_income(edmonton_census) if edmonton_census is not None else 96000
        
        # Add population feature
        out['city_population'] = np.where(
            out['city_calgary'] == 1,
            calgary_pop,
            np.where(out['city_edmonton'] == 1, edmonton_pop, np.nan)
        )
        
        # Add median household income feature
        out['city_median_household_income'] = np.where(
            out['city_calgary'] == 1,
            calgary_income,
            np.where(out['city_edmonton'] == 1, edmonton_income, np.nan)
        )
        
    except Exception as e:
        logger.warning(f"Error extracting census features: {e}")
        out['city_population'] = np.nan
        out['city_median_household_income'] = np.nan
    
    logger.info(f"Added city segmentation features to {len(out)} rows")
    return out


def _extract_census_population(census_df: pd.DataFrame) -> int:
    """Extract total population from census DataFrame."""
    if census_df is None or census_df.empty:
        return 0
    
    try:
        # Look for population row
        pop_mask = census_df.iloc[:, 1].str.contains('Population, 2021', case=False, na=False)
        if pop_mask.any():
            pop_row = census_df[pop_mask].iloc[0]
            # Population is usually in column 3 (Total)
            pop_val = pop_row.iloc[3] if len(pop_row) > 3 else 0
            return int(float(str(pop_val).replace(',', ''))) if pd.notna(pop_val) else 0
    except Exception:
        pass
    return 0


def _extract_census_median_income(census_df: pd.DataFrame) -> float:
    """Extract median household income from census DataFrame."""
    if census_df is None or census_df.empty:
        return 0.0
    
    try:
        # Look for median household income row
        income_mask = census_df.iloc[:, 1].str.contains('Median total income of household', case=False, na=False)
        if income_mask.any():
            income_row = census_df[income_mask].iloc[0]
            # Income value is usually in column 3 (Total)
            income_val = income_row.iloc[3] if len(income_row) > 3 else 0
            return float(str(income_val).replace(',', '').replace('$', '')) if pd.notna(income_val) else 0.0
    except Exception:
        pass
    return 0.0


# =============================================================================
# FEATURE STORE GENERATION
# =============================================================================


def build_feature_store(
    history_df: pd.DataFrame,
    nanos_confidence_df: Optional[pd.DataFrame] = None,
    nanos_better_off_df: Optional[pd.DataFrame] = None,
    commodity_df: Optional[pd.DataFrame] = None,
    cpi_df: Optional[pd.DataFrame] = None,
    calgary_census_df: Optional[pd.DataFrame] = None,
    edmonton_census_df: Optional[pd.DataFrame] = None,
    date_column: str = 'opening_date',
    city_column: str = 'city',
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """Build a comprehensive feature store from all data sources.
    
    Applies all feature engineering transformations:
    - Date normalization
    - Consumer confidence features
    - Energy index
    - Inflation adjustment factor
    - City segmentation features
    
    Args:
        history_df: Base historical show data
        nanos_confidence_df: Nanos consumer confidence data
        nanos_better_off_df: Nanos better off survey data
        commodity_df: Commodity price index data
        cpi_df: CPI monthly data
        calgary_census_df: Calgary census data
        edmonton_census_df: Edmonton census data
        date_column: Name of the date column
        city_column: Name of the city column
        output_path: Optional path to write output parquet file
        
    Returns:
        DataFrame with all features applied
    """
    if history_df.empty:
        logger.warning("Empty history data; returning empty feature store")
        return pd.DataFrame()
    
    features = history_df.copy()
    
    # Normalize column names
    features.columns = [
        c.strip().lower().replace(' - ', '_').replace(' ', '_').replace('-', '_')
        for c in features.columns
    ]
    
    # Find date column (handle variations)
    date_col = None
    for col in [date_column, 'opening_date', 'show_date', 'date', 'performance_date']:
        col_lower = col.lower().replace(' ', '_').replace('-', '_')
        if col_lower in features.columns:
            date_col = col_lower
            break
    
    # Find city column (handle variations)
    city_col = None
    for col in [city_column, 'city', 'location', 'venue_city']:
        col_lower = col.lower().replace(' ', '_').replace('-', '_')
        if col_lower in features.columns:
            city_col = col_lower
            break
    
    # Add year-month column for joins
    if date_col:
        features['year_month'] = snap_show_date_to_month(features[date_col])
    
    # Join consumer confidence
    if nanos_confidence_df is not None and not nanos_confidence_df.empty:
        features = join_consumer_confidence(
            features, 
            nanos_confidence_df,
            date_column=date_col or 'opening_date'
        )
    
    # Join energy index
    if commodity_df is not None and not commodity_df.empty:
        features = join_energy_index(
            features,
            commodity_df,
            date_column=date_col or 'opening_date'
        )
    
    # Compute inflation adjustment factor
    if cpi_df is not None and not cpi_df.empty:
        features = compute_inflation_adjustment_factor(
            features,
            cpi_df,
            date_column=date_col or 'opening_date'
        )
    
    # Add city segmentation features
    if city_col:
        features = add_city_segmentation_features(
            features,
            city_column=city_col,
            calgary_census=calgary_census_df,
            edmonton_census=edmonton_census_df
        )
    
    # Write to parquet if output path specified
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(output_file, index=False)
        logger.info(f"Feature store written to {output_file}")
    
    return features
