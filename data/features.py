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
    
    # Week of year (1-52) - use Int64 to handle NaT values
    week_series = dates.dt.isocalendar().week
    out[f'{pref}week_of_year'] = week_series.astype('Int64')
    
    # Quarter (1-4)
    out[f'{pref}quarter'] = dates.dt.quarter
    
    # Season indicator (meteorological seasons for Northern Hemisphere)
    out[f'{pref}season'] = _compute_season(dates)
    
    # Create binary season flags for model use (vectorized)
    season_col = out[f'{pref}season']
    out[f'{pref}is_winter'] = (season_col == 'winter').astype(int)
    out[f'{pref}is_spring'] = (season_col == 'spring').astype(int)
    out[f'{pref}is_summer'] = (season_col == 'summer').astype(int)
    out[f'{pref}is_autumn'] = (season_col == 'autumn').astype(int)
    
    # Holiday season flag (Nov, Dec, Jan - important for Alberta Ballet) - vectorized
    holiday_months = {11, 12, 1}
    out[f'{pref}is_holiday_season'] = dates.dt.month.isin(holiday_months).astype(int)
    
    # Weekend opening flag (Saturday=5 or Sunday=6) - vectorized
    out[f'{pref}is_weekend'] = (dates.dt.dayofweek >= 5).astype(int)
    
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


# =============================================================================
# MARKETING FEATURE ENGINEERING
# =============================================================================


def derive_marketing_features(
    df: pd.DataFrame,
    marketing_spend_col: str = 'marketing_spend',
    tickets_col: str = 'single_tickets',
    date_col: str = 'start_date',
    lag_periods: Optional[list[int]] = None
) -> pd.DataFrame:
    """Derive marketing-related features from external marketing data.
    
    Creates features useful for understanding marketing effectiveness:
    - Lagged marketing spend (from previous productions)
    - Total marketing spend
    - Marketing spend per ticket
    
    These features are safe for forecasting as they use historical data
    or planned marketing budgets known at forecast time.
    
    Args:
        df: DataFrame with marketing spend and ticket data
        marketing_spend_col: Name of the marketing spend column
        tickets_col: Name of the tickets sold column
        date_col: Name of the date column for lag calculation
        lag_periods: List of lag periods (in rows, sorted by date). 
                     Default is [1] for previous production.
    
    Returns:
        DataFrame with marketing features added:
        - marketing_spend_total: Total marketing spend for the production
        - marketing_spend_per_ticket: Spend per ticket sold (for historical)
        - marketing_spend_lag_1, lag_2, etc.: Lagged spend from prior productions
    """
    if lag_periods is None:
        lag_periods = [1]
    
    out = df.copy()
    
    # Ensure marketing spend column exists
    if marketing_spend_col not in out.columns:
        logger.info(f"Marketing spend column '{marketing_spend_col}' not found. "
                   "Creating placeholder features with NaN.")
        out['marketing_spend_total'] = np.nan
        out['marketing_spend_per_ticket'] = np.nan
        for lag in lag_periods:
            out[f'marketing_spend_lag_{lag}'] = np.nan
        return out
    
    # Total marketing spend (may already be in the data)
    out['marketing_spend_total'] = pd.to_numeric(out[marketing_spend_col], errors='coerce')
    
    # Marketing spend per ticket (only for historical data where tickets are known)
    if tickets_col in out.columns:
        tickets = pd.to_numeric(out[tickets_col], errors='coerce')
        spend = pd.to_numeric(out[marketing_spend_col], errors='coerce')
        # Avoid division by zero
        out['marketing_spend_per_ticket'] = np.where(
            tickets > 0,
            spend / tickets,
            np.nan
        )
        logger.info(f"Created marketing_spend_per_ticket feature")
    else:
        out['marketing_spend_per_ticket'] = np.nan
    
    # Lagged marketing spend (based on date ordering)
    if date_col in out.columns:
        # Sort by date for proper lag calculation
        out = out.sort_values(date_col).reset_index(drop=True)
        
        for lag in lag_periods:
            out[f'marketing_spend_lag_{lag}'] = (
                out['marketing_spend_total'].shift(lag)
            )
        
        logger.info(f"Created lagged marketing spend features: {[f'lag_{l}' for l in lag_periods]}")
    else:
        for lag in lag_periods:
            out[f'marketing_spend_lag_{lag}'] = np.nan
    
    return out


# =============================================================================
# WEATHER FEATURE ENGINEERING
# =============================================================================


def derive_weather_features(
    df: pd.DataFrame,
    avg_temp_col: str = 'weather_avg_temperature',
    min_temp_col: str = 'weather_min_temperature',
    max_temp_col: str = 'weather_max_temperature',
    precip_col: str = 'weather_precipitation',
    snow_col: str = 'weather_snow',
    date_col: str = 'start_date'
) -> pd.DataFrame:
    """Derive weather-related features from joined weather data.
    
    Creates features for understanding weather impact on attendance:
    - Temperature features (normalized, extreme cold flags)
    - Precipitation features (rain/snow indicators)
    - Day-of-week weather interactions
    
    These features capture weather conditions that may affect ticket demand
    and should be derived from historical weather data for forecasting.
    
    Args:
        df: DataFrame with weather data columns
        avg_temp_col: Name of the average temperature column
        min_temp_col: Name of the minimum temperature column
        max_temp_col: Name of the maximum temperature column
        precip_col: Name of the precipitation column
        snow_col: Name of the snow column
        date_col: Name of the date column for day-of-week effects
    
    Returns:
        DataFrame with weather features added:
        - weather_temp_normalized: Temperature normalized to Alberta ranges (-40 to +35)
        - weather_extreme_cold_flag: Flag for extreme cold (<-20°C)
        - weather_extreme_mild_flag: Flag for unusually mild conditions
        - weather_heavy_precip_flag: Flag for heavy precipitation
        - weather_day_of_week: Day of week (0=Monday)
        - weather_weekend_flag: Weekend indicator
    """
    out = df.copy()
    
    # Temperature normalization (Alberta range roughly -40 to +35 Celsius)
    if avg_temp_col in out.columns:
        temp = pd.to_numeric(out[avg_temp_col], errors='coerce')
        # Normalize to 0-1 scale where 0 = -40°C, 1 = +35°C
        out['weather_temp_normalized'] = (temp + 40) / 75.0
        out['weather_temp_normalized'] = out['weather_temp_normalized'].clip(0, 1)
        logger.info("Created weather_temp_normalized feature")
    else:
        out['weather_temp_normalized'] = np.nan
    
    # Extreme cold flag (below -20°C)
    if min_temp_col in out.columns:
        min_temp = pd.to_numeric(out[min_temp_col], errors='coerce')
        out['weather_extreme_cold_flag'] = (min_temp < -20).astype(int)
        out['weather_extreme_cold_flag'] = out['weather_extreme_cold_flag'].fillna(0)
        logger.info("Created weather_extreme_cold_flag feature")
    else:
        out['weather_extreme_cold_flag'] = 0
    
    # Extreme mild flag (winter months with temp > 5°C is unusual)
    if avg_temp_col in out.columns and date_col in out.columns:
        temp = pd.to_numeric(out[avg_temp_col], errors='coerce')
        dates = pd.to_datetime(out[date_col], errors='coerce')
        winter_months = dates.dt.month.isin([12, 1, 2])
        out['weather_extreme_mild_flag'] = ((temp > 5) & winter_months).astype(int)
        out['weather_extreme_mild_flag'] = out['weather_extreme_mild_flag'].fillna(0)
        logger.info("Created weather_extreme_mild_flag feature")
    else:
        out['weather_extreme_mild_flag'] = 0
    
    # Heavy precipitation flag (>10mm rain or >10cm snow)
    has_precip = precip_col in out.columns
    has_snow = snow_col in out.columns
    
    if has_precip or has_snow:
        precip = pd.to_numeric(
            out[precip_col] if precip_col in out.columns else 0,
            errors='coerce'
        ).fillna(0)
        snow = pd.to_numeric(
            out[snow_col] if snow_col in out.columns else 0,
            errors='coerce'
        ).fillna(0)
        out['weather_heavy_precip_flag'] = ((precip > 10) | (snow > 10)).astype(int)
        logger.info("Created weather_heavy_precip_flag feature")
    else:
        out['weather_heavy_precip_flag'] = 0
    
    # Day of week effects (from show date)
    if date_col in out.columns:
        dates = pd.to_datetime(out[date_col], errors='coerce')
        out['weather_day_of_week'] = dates.dt.dayofweek
        out['weather_weekend_flag'] = (dates.dt.dayofweek >= 5).astype(int)
        logger.info("Created weather day-of-week features")
    else:
        out['weather_day_of_week'] = np.nan
        out['weather_weekend_flag'] = 0
    
    return out


# =============================================================================
# ECONOMY FEATURE ENGINEERING
# =============================================================================


def derive_economy_features(
    df: pd.DataFrame,
    unemployment_df: Optional[pd.DataFrame] = None,
    oil_price_df: Optional[pd.DataFrame] = None,
    cpi_df: Optional[pd.DataFrame] = None,
    date_col: str = 'start_date',
    city_col: str = 'city'
) -> pd.DataFrame:
    """Derive economy-related features from external economic data.
    
    Creates features for understanding economic context impact on ticket demand:
    - Unemployment rate (provincial and city-level if available)
    - Oil price (critical for Alberta economy)
    - CPI / inflation indicators
    
    These features are joined via merge_asof for temporal matching.
    
    Args:
        df: DataFrame with show data
        unemployment_df: DataFrame with unemployment data (columns: date, unemployment_rate, region)
        oil_price_df: DataFrame with oil price data (columns: date, wcs_oil_price)
        cpi_df: DataFrame with CPI data (columns: date, V41690973)
        date_col: Name of the date column in df
        city_col: Name of the city column in df
    
    Returns:
        DataFrame with economy features added:
        - economy_unemployment_rate: Unemployment rate at show time
        - economy_oil_price: WCS oil price at show time
        - economy_cpi: Consumer Price Index at show time
        - economy_oil_change_3m: 3-month oil price change (%)
        - economy_unemployment_trend: Unemployment trend (change from prior period)
    """
    out = df.copy()
    
    # Ensure date column is datetime
    if date_col not in out.columns:
        logger.warning(f"Date column '{date_col}' not found; returning without economy features")
        out['economy_unemployment_rate'] = np.nan
        out['economy_oil_price'] = np.nan
        out['economy_cpi'] = np.nan
        out['economy_oil_change_3m'] = np.nan
        out['economy_unemployment_trend'] = np.nan
        return out
    
    out[date_col] = pd.to_datetime(out[date_col], errors='coerce')
    
    # Store original order
    out['_orig_idx'] = range(len(out))
    
    # Join unemployment rate
    if unemployment_df is not None and not unemployment_df.empty:
        out = _join_unemployment_rate(out, unemployment_df, date_col, city_col)
    else:
        out['economy_unemployment_rate'] = np.nan
        out['economy_unemployment_trend'] = np.nan
    
    # Join oil price
    if oil_price_df is not None and not oil_price_df.empty:
        out = _join_oil_price(out, oil_price_df, date_col)
    else:
        out['economy_oil_price'] = np.nan
        out['economy_oil_change_3m'] = np.nan
    
    # Join CPI
    if cpi_df is not None and not cpi_df.empty:
        out = _join_cpi(out, cpi_df, date_col)
    else:
        out['economy_cpi'] = np.nan
    
    # Restore original order and clean up
    out = out.sort_values('_orig_idx').reset_index(drop=True)
    out = out.drop(columns=['_orig_idx'], errors='ignore')
    
    return out


def _join_unemployment_rate(
    df: pd.DataFrame,
    unemployment_df: pd.DataFrame,
    date_col: str,
    city_col: str
) -> pd.DataFrame:
    """Join unemployment rate data using temporal matching."""
    unemp = unemployment_df.copy()
    
    # Normalize column names
    unemp.columns = [c.lower().replace(' ', '_').replace('-', '_') for c in unemp.columns]
    
    if 'date' not in unemp.columns:
        df['economy_unemployment_rate'] = np.nan
        df['economy_unemployment_trend'] = np.nan
        return df
    
    unemp['date'] = pd.to_datetime(unemp['date'], errors='coerce')
    unemp = unemp.dropna(subset=['date']).sort_values('date')
    
    # Use Alberta-level data (most comprehensive)
    if 'region' in unemp.columns:
        unemp_alberta = unemp[unemp['region'].str.lower() == 'alberta'].copy()
        if unemp_alberta.empty:
            unemp_alberta = unemp.copy()
    else:
        unemp_alberta = unemp.copy()
    
    if 'unemployment_rate' not in unemp_alberta.columns:
        df['economy_unemployment_rate'] = np.nan
        df['economy_unemployment_trend'] = np.nan
        return df
    
    # Calculate unemployment trend (change from 3 months prior)
    unemp_alberta = unemp_alberta.sort_values('date')
    unemp_alberta['_unemp_lag'] = unemp_alberta['unemployment_rate'].shift(3)
    unemp_alberta['economy_unemployment_trend'] = (
        unemp_alberta['unemployment_rate'] - unemp_alberta['_unemp_lag']
    )
    
    # Sort df for merge_asof
    df_sorted = df.sort_values(date_col)
    
    # Perform merge_asof
    try:
        merged = pd.merge_asof(
            df_sorted,
            unemp_alberta[['date', 'unemployment_rate', 'economy_unemployment_trend']]
                .rename(columns={'unemployment_rate': 'economy_unemployment_rate'}),
            left_on=date_col,
            right_on='date',
            direction='backward'
        )
        merged = merged.drop(columns=['date'], errors='ignore')
        
        # Fill NaN with median
        median_rate = unemp_alberta['unemployment_rate'].median()
        merged['economy_unemployment_rate'] = merged['economy_unemployment_rate'].fillna(median_rate)
        merged['economy_unemployment_trend'] = merged['economy_unemployment_trend'].fillna(0)
        
        logger.info(f"Joined unemployment rate (mean={merged['economy_unemployment_rate'].mean():.2f}%)")
        return merged
    except Exception as e:
        logger.warning(f"Error joining unemployment rate: {e}")
        df['economy_unemployment_rate'] = np.nan
        df['economy_unemployment_trend'] = np.nan
        return df


def _join_oil_price(
    df: pd.DataFrame,
    oil_df: pd.DataFrame,
    date_col: str
) -> pd.DataFrame:
    """Join oil price data using temporal matching."""
    oil = oil_df.copy()
    
    # Normalize column names
    oil.columns = [c.lower().replace(' ', '_').replace('-', '_') for c in oil.columns]
    
    if 'date' not in oil.columns:
        df['economy_oil_price'] = np.nan
        df['economy_oil_change_3m'] = np.nan
        return df
    
    oil['date'] = pd.to_datetime(oil['date'], errors='coerce')
    oil = oil.dropna(subset=['date']).sort_values('date')
    
    # Get price column
    price_col = 'wcs_oil_price'
    if price_col not in oil.columns:
        # Try to find any price column
        for col in oil.columns:
            if 'price' in col.lower():
                price_col = col
                break
        else:
            df['economy_oil_price'] = np.nan
            df['economy_oil_change_3m'] = np.nan
            return df
    
    # Calculate 3-month price change
    oil = oil.sort_values('date')
    oil['_price_lag'] = oil[price_col].shift(3)
    oil['economy_oil_change_3m'] = np.where(
        oil['_price_lag'] > 0,
        (oil[price_col] - oil['_price_lag']) / oil['_price_lag'] * 100,
        np.nan
    )
    
    # Sort df for merge_asof
    df_sorted = df.sort_values(date_col)
    
    # Perform merge_asof
    try:
        merged = pd.merge_asof(
            df_sorted,
            oil[['date', price_col, 'economy_oil_change_3m']]
                .rename(columns={price_col: 'economy_oil_price'}),
            left_on=date_col,
            right_on='date',
            direction='backward'
        )
        merged = merged.drop(columns=['date'], errors='ignore')
        
        # Fill NaN with median
        median_price = oil[price_col].median()
        merged['economy_oil_price'] = merged['economy_oil_price'].fillna(median_price)
        merged['economy_oil_change_3m'] = merged['economy_oil_change_3m'].fillna(0)
        
        logger.info(f"Joined oil price (mean=${merged['economy_oil_price'].mean():.2f})")
        return merged
    except Exception as e:
        logger.warning(f"Error joining oil price: {e}")
        df['economy_oil_price'] = np.nan
        df['economy_oil_change_3m'] = np.nan
        return df


def _join_cpi(
    df: pd.DataFrame,
    cpi_df: pd.DataFrame,
    date_col: str
) -> pd.DataFrame:
    """Join CPI data using temporal matching."""
    cpi = cpi_df.copy()
    
    # Normalize column names
    cpi.columns = [c.lower().replace(' ', '_').replace('-', '_') for c in cpi.columns]
    
    if 'date' not in cpi.columns:
        df['economy_cpi'] = np.nan
        return df
    
    cpi['date'] = pd.to_datetime(cpi['date'], errors='coerce')
    cpi = cpi.dropna(subset=['date']).sort_values('date')
    
    # Get CPI column (V41690973 is CPI All-items)
    cpi_col = None
    for col in ['v41690973', 'cpi', 'cpi_all_items']:
        if col in cpi.columns:
            cpi_col = col
            break
    
    if cpi_col is None:
        df['economy_cpi'] = np.nan
        return df
    
    # Sort df for merge_asof
    df_sorted = df.sort_values(date_col)
    
    # Perform merge_asof
    try:
        merged = pd.merge_asof(
            df_sorted,
            cpi[['date', cpi_col]].rename(columns={cpi_col: 'economy_cpi'}),
            left_on=date_col,
            right_on='date',
            direction='backward'
        )
        merged = merged.drop(columns=['date'], errors='ignore')
        
        # Fill NaN with median
        median_cpi = cpi[cpi_col].median()
        merged['economy_cpi'] = merged['economy_cpi'].fillna(median_cpi)
        
        logger.info(f"Joined CPI (mean={merged['economy_cpi'].mean():.2f})")
        return merged
    except Exception as e:
        logger.warning(f"Error joining CPI: {e}")
        df['economy_cpi'] = np.nan
        return df


# =============================================================================
# BASELINE SIGNAL FEATURE ENGINEERING
# =============================================================================


def derive_baseline_features(
    df: pd.DataFrame,
    baselines_df: Optional[pd.DataFrame] = None,
    title_col: str = 'show_title',
    wiki_col: str = 'wiki',
    trends_col: str = 'trends',
    youtube_col: str = 'youtube',
    spotify_col: str = 'spotify'
) -> pd.DataFrame:
    """Derive baseline signal features from external data sources.
    
    Creates features from cultural visibility signals:
    - Wikipedia article views (familiarity indicator)
    - Google Trends search interest
    - YouTube view counts
    - Spotify play counts (for musical titles)
    
    These signals are combined into composite indices for modeling.
    
    Args:
        df: DataFrame with show data
        baselines_df: DataFrame with baseline signals per title
        title_col: Name of the title column in df
        wiki_col: Name of the Wikipedia column in baselines_df
        trends_col: Name of the Google Trends column in baselines_df
        youtube_col: Name of the YouTube column in baselines_df
        spotify_col: Name of the Spotify column in baselines_df
    
    Returns:
        DataFrame with baseline features added:
        - baseline_wiki: Wikipedia signal (0-100)
        - baseline_trends: Google Trends signal (0-100)
        - baseline_youtube: YouTube signal (0-100)
        - baseline_spotify: Spotify signal (0-100)
        - baseline_familiarity_index: Composite familiarity index
        - baseline_digital_presence: Combined digital presence score
    """
    out = df.copy()
    
    if baselines_df is None or baselines_df.empty:
        logger.info("No baselines data provided; creating placeholder features")
        out['baseline_wiki'] = np.nan
        out['baseline_trends'] = np.nan
        out['baseline_youtube'] = np.nan
        out['baseline_spotify'] = np.nan
        out['baseline_familiarity_index'] = np.nan
        out['baseline_digital_presence'] = np.nan
        return out
    
    baselines = baselines_df.copy()
    
    # Normalize column names
    baselines.columns = [c.lower().strip() for c in baselines.columns]
    
    # Find title column in baselines
    baselines_title_col = None
    for col in ['title', 'show_title', 'canonical_title']:
        if col in baselines.columns:
            baselines_title_col = col
            break
    
    if baselines_title_col is None:
        logger.warning("No title column found in baselines data")
        out['baseline_wiki'] = np.nan
        out['baseline_trends'] = np.nan
        out['baseline_youtube'] = np.nan
        out['baseline_spotify'] = np.nan
        out['baseline_familiarity_index'] = np.nan
        out['baseline_digital_presence'] = np.nan
        return out
    
    # Prepare signal columns
    signal_cols = {
        'baseline_wiki': wiki_col,
        'baseline_trends': trends_col,
        'baseline_youtube': youtube_col,
        'baseline_spotify': spotify_col
    }
    
    # Select and rename columns for merge
    cols_to_merge = [baselines_title_col]
    rename_map = {}
    
    for new_name, old_name in signal_cols.items():
        if old_name in baselines.columns:
            cols_to_merge.append(old_name)
            rename_map[old_name] = new_name
    
    if len(cols_to_merge) == 1:  # Only title column
        logger.warning("No signal columns found in baselines data")
        out['baseline_wiki'] = np.nan
        out['baseline_trends'] = np.nan
        out['baseline_youtube'] = np.nan
        out['baseline_spotify'] = np.nan
        out['baseline_familiarity_index'] = np.nan
        out['baseline_digital_presence'] = np.nan
        return out
    
    baselines_subset = baselines[cols_to_merge].copy()
    baselines_subset = baselines_subset.rename(columns=rename_map)
    
    # Normalize title for matching
    out['_title_match'] = out[title_col].str.lower().str.strip()
    baselines_subset['_title_match'] = baselines_subset[baselines_title_col].str.lower().str.strip()
    
    # Merge on normalized title
    merged = out.merge(
        baselines_subset.drop(columns=[baselines_title_col]),
        on='_title_match',
        how='left'
    )
    merged = merged.drop(columns=['_title_match'], errors='ignore')
    
    # Ensure all columns exist
    for col in ['baseline_wiki', 'baseline_trends', 'baseline_youtube', 'baseline_spotify']:
        if col not in merged.columns:
            merged[col] = np.nan
    
    # Convert to numeric
    for col in ['baseline_wiki', 'baseline_trends', 'baseline_youtube', 'baseline_spotify']:
        merged[col] = pd.to_numeric(merged[col], errors='coerce')
    
    # Compute composite indices
    # Familiarity index: weighted average of wiki and trends (most stable indicators)
    wiki = merged['baseline_wiki'].fillna(50)
    trends = merged['baseline_trends'].fillna(50)
    youtube = merged['baseline_youtube'].fillna(50)
    spotify = merged['baseline_spotify'].fillna(50)
    
    # Familiarity index emphasizes cultural recognition
    merged['baseline_familiarity_index'] = (
        0.40 * wiki +
        0.35 * trends +
        0.15 * youtube +
        0.10 * spotify
    )
    
    # Digital presence: overall online visibility
    merged['baseline_digital_presence'] = (
        0.25 * wiki +
        0.25 * trends +
        0.30 * youtube +
        0.20 * spotify
    )
    
    logger.info(f"Created baseline signal features for {len(merged)} rows")
    
    return merged


# =============================================================================
# COMBINED FEATURE DERIVATION
# =============================================================================


def derive_all_external_features(
    df: pd.DataFrame,
    marketing_df: Optional[pd.DataFrame] = None,
    weather_df: Optional[pd.DataFrame] = None,
    unemployment_df: Optional[pd.DataFrame] = None,
    oil_price_df: Optional[pd.DataFrame] = None,
    cpi_df: Optional[pd.DataFrame] = None,
    baselines_df: Optional[pd.DataFrame] = None,
    date_col: str = 'start_date',
    city_col: str = 'city',
    title_col: str = 'show_title'
) -> pd.DataFrame:
    """Derive all external features from various data sources.
    
    This is a convenience function that applies all feature derivation
    functions in the proper order. Use this after joining external data
    to create model-ready features.
    
    Feature categories created:
    1. Marketing: lagged spend, total spend, spend per ticket
    2. Weather: temperature, precipitation, day-of-week effects  
    3. Economy: unemployment rate, oil price, CPI
    4. Baselines: wiki/trends/youtube/spotify signals
    
    Args:
        df: DataFrame with show data and joined external data
        marketing_df: Marketing spend data (optional, uses columns in df if None)
        weather_df: Weather data (optional, uses columns in df if None)
        unemployment_df: Unemployment rate data
        oil_price_df: Oil price data
        cpi_df: CPI data
        baselines_df: Baseline signals data
        date_col: Name of the date column
        city_col: Name of the city column
        title_col: Name of the title column
    
    Returns:
        DataFrame with all derived features added
    """
    out = df.copy()
    original_cols = len(out.columns)
    
    # 1. Marketing features
    if marketing_df is not None:
        # Join marketing data first
        out = out.merge(marketing_df, on=[title_col, date_col], how='left', suffixes=('', '_mkt'))
    
    out = derive_marketing_features(
        out,
        date_col=date_col,
        lag_periods=[1, 2]  # Previous 2 productions
    )
    
    # 2. Weather features
    out = derive_weather_features(
        out,
        date_col=date_col
    )
    
    # 3. Economy features
    out = derive_economy_features(
        out,
        unemployment_df=unemployment_df,
        oil_price_df=oil_price_df,
        cpi_df=cpi_df,
        date_col=date_col,
        city_col=city_col
    )
    
    # 4. Baseline features
    out = derive_baseline_features(
        out,
        baselines_df=baselines_df,
        title_col=title_col
    )
    
    new_cols = len(out.columns) - original_cols
    logger.info(f"Created {new_cols} derived features from external data")
    
    return out
