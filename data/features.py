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
    
    Creates total_single_tickets from single_tickets if needed for 
    consistent target column naming across the pipeline.
    """
    out = df.copy()
    
    # Create total_single_tickets if only single_tickets exists
    # This ensures consistent target column naming across the pipeline
    if 'total_single_tickets' not in out.columns and 'single_tickets' in out.columns:
        out['total_single_tickets'] = out['single_tickets']
    
    return out


# =============================================================================
# DATE-BASED FEATURE DERIVATION
# =============================================================================


def derive_date_features(
    df: pd.DataFrame,
    start_date_col: str = 'start_date',
    end_date_col: str = 'end_date'
) -> pd.DataFrame:
    """Derive date-based features from start_date and end_date columns.
    
    Extracts temporal features useful for forecasting:
    - Year, month, day of week from opening date
    - Run duration in days
    - Season indicators (spring, summer, autumn, winter)
    - Holiday season flag (Nov-Jan for Alberta Ballet productions)
    
    All features are safe for forecasting as they are derived from 
    planned run dates which are known at forecast time.
    
    Args:
        df: DataFrame with date columns
        start_date_col: Name of the start date column
        end_date_col: Name of the end date column
        
    Returns:
        DataFrame with additional date-based feature columns
    """
    out = df.copy()
    
    # Parse date columns if not already datetime
    for col in [start_date_col, end_date_col]:
        if col in out.columns and not pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = pd.to_datetime(out[col], errors='coerce')
    
    # Extract features from start_date (opening date)
    if start_date_col in out.columns:
        out = _extract_temporal_features(out, start_date_col, prefix='opening')
    
    # Compute run duration if both dates available
    if start_date_col in out.columns and end_date_col in out.columns:
        out['run_duration_days'] = (out[end_date_col] - out[start_date_col]).dt.days
        # Fill NaN with 0 and ensure non-negative
        out['run_duration_days'] = out['run_duration_days'].fillna(0).clip(lower=0)
        logger.info(f"Added run_duration_days (mean={out['run_duration_days'].mean():.1f} days)")
    
    return out


def _extract_temporal_features(
    df: pd.DataFrame,
    date_col: str,
    prefix: str = ''
) -> pd.DataFrame:
    """Extract temporal features from a single date column.
    
    Args:
        df: DataFrame with the date column
        date_col: Name of the date column to extract features from
        prefix: Prefix for feature column names (e.g., 'opening' -> 'opening_year')
        
    Returns:
        DataFrame with additional temporal feature columns
    """
    out = df.copy()
    
    if date_col not in out.columns:
        logger.warning(f"Date column '{date_col}' not found; skipping temporal features")
        return out
    
    # Ensure datetime type
    dates = pd.to_datetime(out[date_col], errors='coerce')
    
    # Add prefix separator if provided
    pref = f"{prefix}_" if prefix else ""
    
    # Year (e.g., 2024)
    out[f'{pref}year'] = dates.dt.year
    
    # Month (1-12)
    out[f'{pref}month'] = dates.dt.month
    
    # Day of week (0=Monday, 6=Sunday)
    out[f'{pref}day_of_week'] = dates.dt.dayofweek
    
    # Week of year (1-52)
    out[f'{pref}week_of_year'] = dates.dt.isocalendar().week.astype('Int64')
    
    # Quarter (1-4)
    out[f'{pref}quarter'] = dates.dt.quarter
    
    # Season indicator (meteorological seasons for Northern Hemisphere)
    out[f'{pref}season'] = _compute_season(dates)
    
    # Create binary season flags for model use
    out[f'{pref}is_winter'] = out[f'{pref}season'].apply(lambda x: 1 if x == 'winter' else 0)
    out[f'{pref}is_spring'] = out[f'{pref}season'].apply(lambda x: 1 if x == 'spring' else 0)
    out[f'{pref}is_summer'] = out[f'{pref}season'].apply(lambda x: 1 if x == 'summer' else 0)
    out[f'{pref}is_autumn'] = out[f'{pref}season'].apply(lambda x: 1 if x == 'autumn' else 0)
    
    # Holiday season flag (Nov, Dec, Jan - important for Alberta Ballet)
    holiday_months = {11, 12, 1}
    out[f'{pref}is_holiday_season'] = dates.dt.month.apply(
        lambda x: 1 if pd.notna(x) and int(x) in holiday_months else 0
    )
    
    # Weekend opening flag (Saturday=5 or Sunday=6)
    out[f'{pref}is_weekend'] = dates.dt.dayofweek.apply(
        lambda x: 1 if pd.notna(x) and x >= 5 else 0
    )
    
    logger.info(f"Added {9} temporal features with prefix '{pref}'")
    
    return out


def _compute_season(dates: pd.Series) -> pd.Series:
    """Compute meteorological season from dates.
    
    Uses Northern Hemisphere meteorological seasons:
    - Winter: December, January, February
    - Spring: March, April, May  
    - Summer: June, July, August
    - Autumn: September, October, November
    
    Args:
        dates: Series of datetime values
        
    Returns:
        Series of season names (winter, spring, summer, autumn)
    """
    months = dates.dt.month
    
    def month_to_season(m):
        if pd.isna(m):
            return np.nan
        m = int(m)
        if m in {12, 1, 2}:
            return 'winter'
        elif m in {3, 4, 5}:
            return 'spring'
        elif m in {6, 7, 8}:
            return 'summer'
        else:  # 9, 10, 11
            return 'autumn'
    
    return months.apply(month_to_season)


def compute_days_to_opening(
    df: pd.DataFrame,
    start_date_col: str = 'start_date',
    reference_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """Compute days until opening from a reference date.
    
    This feature is useful for forecasting models that predict at 
    different lead times before the show opens.
    
    Args:
        df: DataFrame with start date column
        start_date_col: Name of the start date column
        reference_date: Reference date for computing lead time.
                       If None, uses current date.
        
    Returns:
        DataFrame with days_to_opening column added
    """
    out = df.copy()
    
    if start_date_col not in out.columns:
        logger.warning(f"Date column '{start_date_col}' not found; cannot compute days_to_opening")
        out['days_to_opening'] = np.nan
        return out
    
    if reference_date is None:
        reference_date = pd.Timestamp.now()
    
    start_dates = pd.to_datetime(out[start_date_col], errors='coerce')
    out['days_to_opening'] = (start_dates - reference_date).dt.days
    
    logger.info(f"Added days_to_opening feature (reference: {reference_date.date()})")
    
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
    """Join Nanos consumer confidence data to show data using temporal matching.
    
    Matches show dates to historical consumer confidence data via merge_asof.
    Uses Prairies region as the most relevant for Alberta.
    
    Args:
        df: Show/production DataFrame with date column
        nanos_df: Nanos consumer confidence DataFrame with year_or_period column
        date_column: Name of the date column in df
        target_metric: Which metric to use (e.g., 'This week')
        region: Which region to use (default 'Prairies' for Alberta)
        
    Returns:
        DataFrame with consumer_confidence_prairies and consumer_confidence_headline features added
    """
    if nanos_df.empty:
        logger.warning("Empty Nanos data; using default consumer confidence")
        out = df.copy()
        out['consumer_confidence_prairies'] = 50.0
        out['consumer_confidence_headline'] = 50.0
        return out
    
    out = df.copy()
    
    # Check if date column exists and has valid dates
    if date_column not in out.columns or out[date_column].isna().all():
        logger.warning(f"Date column '{date_column}' missing or empty; using default consumer confidence")
        out['consumer_confidence_prairies'] = 50.0
        out['consumer_confidence_headline'] = 50.0
        return out
    
    try:
        # Filter for regional data (Prairies)
        regional = nanos_df[
            (nanos_df['category'] == 'Demographics') & 
            (nanos_df['subcategory'] == 'Region') &
            (nanos_df['metric'] == region)
        ].copy()
        
        # Also get headline index
        headline = nanos_df[
            (nanos_df['category'] == 'BNCCI') & 
            (nanos_df['subcategory'] == 'Headline Index') &
            (nanos_df['metric'] == target_metric)
        ].copy()
        
        # Use regional for consumer_confidence_prairies, headline for consumer_confidence_headline
        # If regional is empty, fall back to headline for both
        confidence_data_prairies = regional if not regional.empty else headline
        confidence_data_headline = headline if not headline.empty else regional
        
        if confidence_data_prairies.empty and confidence_data_headline.empty:
            logger.warning("No suitable consumer confidence data found; using default")
            out['consumer_confidence_prairies'] = 50.0
            out['consumer_confidence_headline'] = 50.0
            return out
        
        # Helper to prepare and merge confidence data
        def prepare_and_merge(conf_data, out_df, col_name, date_col):
            if conf_data.empty:
                out_df[col_name] = 50.0
                return out_df
            
            conf_copy = conf_data.copy()
            conf_copy['_conf_date'] = pd.to_datetime(conf_copy['year_or_period'], errors='coerce')
            conf_copy = conf_copy.dropna(subset=['_conf_date', 'value'])
            
            if conf_copy.empty:
                out_df[col_name] = 50.0
                return out_df
            
            # Sort by date
            conf_copy = conf_copy.sort_values('_conf_date')
            median_val = conf_copy['value'].median()
            
            # Prepare show data
            out_df['_show_date'] = pd.to_datetime(out_df[date_col], errors='coerce')
            out_with_dates = out_df[out_df['_show_date'].notna()].copy()
            out_without_dates = out_df[out_df['_show_date'].isna()].copy()
            
            if not out_with_dates.empty:
                out_with_dates = pd.merge_asof(
                    out_with_dates.sort_values('_show_date'),
                    conf_copy[['_conf_date', 'value']].rename(columns={'value': col_name}),
                    left_on='_show_date',
                    right_on='_conf_date',
                    direction='backward'
                )
                out_with_dates[col_name] = out_with_dates[col_name].fillna(median_val)
                out_df = pd.concat([out_with_dates, out_without_dates], ignore_index=True)
            
            if col_name not in out_df.columns:
                out_df[col_name] = median_val
            else:
                out_df[col_name] = out_df[col_name].fillna(median_val)
            
            out_df = out_df.drop(columns=['_show_date', '_conf_date'], errors='ignore')
            return out_df
        
        # Add prairies confidence
        out = prepare_and_merge(confidence_data_prairies, out, 'consumer_confidence_prairies', date_column)
        
        # Add headline confidence
        out = prepare_and_merge(confidence_data_headline, out, 'consumer_confidence_headline', date_column)
        
        logger.info(f"Added consumer_confidence_prairies and consumer_confidence_headline to {len(out)} rows")
        return out
        
    except Exception as e:
        logger.warning(f"Error joining consumer confidence: {e}")
        out['consumer_confidence_prairies'] = 50.0
        out['consumer_confidence_headline'] = 50.0
        return out


def join_energy_index(
    df: pd.DataFrame,
    commodity_df: pd.DataFrame,
    date_column: str = 'show_date'
) -> pd.DataFrame:
    """Join commodity price Energy index to show data using temporal matching.
    
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
        logger.warning("Empty commodity data; using default energy index")
        df['energy_index'] = 100.0
        return df
    
    out = df.copy()
    
    # Check if date column exists and has valid dates
    if date_column not in out.columns or out[date_column].isna().all():
        logger.warning(f"Date column '{date_column}' missing or empty; using default energy index")
        out['energy_index'] = 100.0
        return out
    
    try:
        # Prepare commodity data
        comm = commodity_df.copy()
        
        # Normalize column names to uppercase
        comm.columns = [c.upper() if c.lower() != 'date' else c for c in comm.columns]
        
        if 'date' not in comm.columns or 'A.ENER' not in comm.columns:
            logger.warning("Commodity data missing 'date' or 'A.ENER' column; using default")
            out['energy_index'] = 100.0
            return out
        
        comm['date'] = pd.to_datetime(comm['date'], errors='coerce')
        comm = comm.dropna(subset=['date', 'A.ENER']).sort_values('date')
        
        if comm.empty:
            logger.warning("No valid commodity data after date parsing; using default")
            out['energy_index'] = 100.0
            return out
        
        # Prepare show data
        out['_show_date'] = pd.to_datetime(out[date_column], errors='coerce')
        out_with_dates = out[out['_show_date'].notna()].copy()
        out_without_dates = out[out['_show_date'].isna()].copy()
        
        if not out_with_dates.empty:
            # Merge using asof (nearest prior date)
            out_with_dates = pd.merge_asof(
                out_with_dates.sort_values('_show_date'),
                comm[['date', 'A.ENER']].rename(columns={'A.ENER': 'energy_index'}),
                left_on='_show_date',
                right_on='date',
                direction='backward'
            )
            
            # Fill any remaining nulls with median
            median_energy = comm['A.ENER'].median()
            out_with_dates['energy_index'] = out_with_dates['energy_index'].fillna(median_energy)
            
            # Combine back
            out = pd.concat([out_with_dates, out_without_dates], ignore_index=True)
        
        # Fill rows without dates with median energy
        if 'energy_index' not in out.columns:
            out['energy_index'] = comm['A.ENER'].median()
        else:
            out['energy_index'] = out['energy_index'].fillna(comm['A.ENER'].median())
        
        # Clean up temporary columns
        out = out.drop(columns=['_show_date', 'date'], errors='ignore')
        
        logger.info(f"Added energy_index to {len(out)} rows (mean={out['energy_index'].mean():.2f})")
        return out
        
    except Exception as e:
        logger.warning(f"Error joining energy index: {e}")
        out['energy_index'] = 100.0
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
        
        # CRITICAL: Sort CPI data by date to ensure order-independent results
        cpi = cpi.sort_values('date').reset_index(drop=True)
        
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
        
        # Get base CPI value - always sort before using iloc
        base_dt = pd.to_datetime(base_date)
        base_cpi_row = cpi[cpi['date'] <= base_dt].sort_values('date')
        if len(base_cpi_row) > 0:
            base_cpi = base_cpi_row[cpi_col].iloc[-1]
        else:
            # Use first available value (cpi is already sorted by date)
            base_cpi = cpi[cpi_col].iloc[0]
        
        # Prepare show data
        if date_column not in out.columns or out[date_column].isna().all():
            logger.warning(f"Date column '{date_column}' missing or empty; using default inflation factor")
            out['inflation_adjustment_factor'] = 1.0
            return out
        
        # Store original index for proper reordering after merge
        out['_orig_idx'] = range(len(out))
        
        out['_show_date'] = pd.to_datetime(out[date_column], errors='coerce')
        out_with_dates = out[out['_show_date'].notna()].copy()
        out_without_dates = out[out['_show_date'].isna()].copy()
        
        if not out_with_dates.empty:
            # Sort by show date for merge_asof, but preserve original order info
            out_with_dates = out_with_dates.sort_values('_show_date')
            
            # Merge using asof (nearest prior date) - CPI is already sorted by date
            out_with_dates = pd.merge_asof(
                out_with_dates,
                cpi[['date', cpi_col]].rename(columns={cpi_col: '_cpi_value'}),
                left_on='_show_date',
                right_on='date',
                direction='backward'
            )
            
            # Compute inflation factor relative to base
            out_with_dates['inflation_adjustment_factor'] = out_with_dates['_cpi_value'] / base_cpi
            out_with_dates['inflation_adjustment_factor'] = out_with_dates['inflation_adjustment_factor'].fillna(1.0)
            
            # Combine back and restore original order
            out = pd.concat([out_with_dates, out_without_dates], ignore_index=True)
            out = out.sort_values('_orig_idx').reset_index(drop=True)
        
        # Fill rows without dates with default
        if 'inflation_adjustment_factor' not in out.columns:
            out['inflation_adjustment_factor'] = 1.0
        else:
            out['inflation_adjustment_factor'] = out['inflation_adjustment_factor'].fillna(1.0)
        
        # Clean up temp columns
        out = out.drop(columns=['_show_date', 'date', '_cpi_value', '_orig_idx'], errors='ignore')
        
        logger.info(f"Added inflation_adjustment_factor to {len(out)} rows (mean={out['inflation_adjustment_factor'].mean():.3f})")
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
    for col in [date_column, 'opening_date', 'start_date', 'show_date', 'date', 'performance_date']:
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
