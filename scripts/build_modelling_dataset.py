#!/usr/bin/env python3
"""
Build Modelling Dataset Script - PRIMARY RECOMMENDED PIPELINE

This script produces a leak-free modelling dataset for training ticket demand models.
It combines baseline signals, historical sales, and contextual features while ensuring
no current-run ticket information is used as a predictor.

**This is the canonical, recommended ML path for Alberta Ballet ticket prediction.**

The safe modelling dataset pipeline consists of:
1. ``python scripts/build_modelling_dataset.py`` (this script) - Build leak-free dataset
2. ``python scripts/train_safe_model.py --tune`` - Train model with time-aware CV
3. ``python scripts/backtest_timeaware.py`` - Evaluate prediction methods

IMPORTANT: This script is designed to prevent data leakage. Only forecast-time features
are included as predictors. Current-run ticket columns are only allowed as targets or
for computing lagged historical features from PRIOR seasons.

**Note:** The legacy baseline pipeline (ml/dataset.py + ml/training.py) is deprecated
and should not be used for production. It has known leakage risks.

Usage:
    python scripts/build_modelling_dataset.py [options]

Outputs:
    - data/modelling_dataset.csv: The leak-free dataset for model training
    - diagnostics/modelling_dataset_report.json: Diagnostics and data quality report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.canonicalize_titles import canonicalize_title, fuzzy_match_title
from data.features import build_feature_store, derive_date_features
from data.loader import (
    load_nanos_consumer_confidence,
    load_nanos_better_off,
    load_nanos_arts_donors,
    load_commodity_price_index,
    load_boc_cpi_monthly,
    load_census_data
)

# ============================================================================
# SAFETY: Define columns that should NEVER be used as predictors
# ============================================================================
# These columns contain current-run information and would cause data leakage
FORBIDDEN_PREDICTOR_PATTERNS = [
    "single_tickets",
    "single tickets",
    "total_tickets",
    "total tickets",
    "total_single_tickets",
    "total single tickets",
    "yourmodel_",  # Model predictions from history file
    "_tickets_-_",
    "_tickets_calgary",
    "_tickets_edmonton",
]

# Columns explicitly allowed (even if they contain 'ticket' in name)
ALLOWED_PREDICTOR_COLUMNS = {
    "prior_total_tickets",
    "mean_last_3_seasons",
    "ticket_median_prior",
    "years_since_last_ticket",
    "ticket_index_deseason",  # This is derived from prior history
}


def is_forbidden_predictor(col_name: str) -> bool:
    """
    Check if a column name matches forbidden predictor patterns.
    
    This is a safety check to prevent current-run ticket columns from
    being used as predictors, which would cause data leakage.
    
    Args:
        col_name: Column name to check
        
    Returns:
        True if the column should NOT be used as a predictor
    """
    col_lower = col_name.lower().strip()
    
    # Allow explicitly permitted columns
    if col_lower in ALLOWED_PREDICTOR_COLUMNS:
        return False
    
    # Check forbidden patterns
    for pattern in FORBIDDEN_PREDICTOR_PATTERNS:
        if pattern in col_lower:
            return True
    
    return False


def assert_no_leakage(df: pd.DataFrame, feature_cols: List[str], context: str = "") -> None:
    """
    Assert that no forbidden predictor columns are in the feature set.
    
    Args:
        df: DataFrame to check
        feature_cols: List of column names being used as features
        context: Description for error messages
        
    Raises:
        AssertionError: If any forbidden columns are found
    """
    forbidden_found = []
    for col in feature_cols:
        if is_forbidden_predictor(col):
            forbidden_found.append(col)
    
    if forbidden_found:
        raise AssertionError(
            f"DATA LEAKAGE DETECTED{' in ' + context if context else ''}!\n"
            f"The following current-run ticket columns were found in the feature set:\n"
            f"  {forbidden_found}\n"
            f"These columns must NOT be used as predictors as they would cause leakage.\n"
            f"Only prior-season aggregates (e.g., 'prior_total_tickets') are allowed."
        )


def load_combined_history(path: str = "data/productions/history_city_sales.csv") -> pd.DataFrame:
    """Load and preprocess combined historical sales data (city-level rows).
    
    Expected columns:
    - city: Calgary or Edmonton
    - show_title: Title of the show
    - start_date: Run start date
    - end_date: Run end date
    - single_tickets: Tickets sold for this city/run
    
    Adds:
    - opening_date: Canonical date column for feature joins (copy of start_date)
    - month_of_opening: Numeric month (1-12) derived from start_date
    """
    try:
        df = pd.read_csv(path, thousands=",")
    except FileNotFoundError:
        print(f"Warning: Combined history file not found at {path}")
        return pd.DataFrame()
    
    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Parse dates
    for col in ["start_date", "end_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    
    # Create opening_date as canonical date column for economic feature joins
    if "start_date" in df.columns:
        df["opening_date"] = df["start_date"]
        # Extract month of opening (1-12)
        df["month_of_opening"] = df["start_date"].dt.month
    
    # Ensure single_tickets is numeric
    if "single_tickets" in df.columns:
        df["single_tickets"] = pd.to_numeric(df["single_tickets"], errors="coerce").fillna(0)
    
    # Add canonical title
    if "show_title" in df.columns:
        df["canonical_title"] = df["show_title"].apply(canonicalize_title)
    
    return df


def load_baselines(path: str = "data/productions/baselines.csv") -> pd.DataFrame:
    """Load and preprocess baseline signals data."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Warning: Baselines file not found at {path}")
        return pd.DataFrame()
    
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Ensure we have a title column
    if "title" not in df.columns:
        print("Warning: 'title' column not found in baselines")
        return df
    
    # Add canonical title for matching
    df["canonical_title"] = df["title"].apply(canonicalize_title)
    
    return df


# Removed load_past_runs - dates now included in combined history file


def load_optional_csv(path: str, name: str) -> Optional[pd.DataFrame]:
    """Try to load an optional external CSV, return None if not found."""
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        print(f"  ✓ Loaded {name}: {len(df)} rows, {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        print(f"  - {name} not found (optional)")
        return None


def compute_ticket_priors(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ticket priors from historical data (city-level rows).
    
    For each show title, compute aggregates from historical runs across all cities.
    This is safe as we'll only use PRIOR season data during training.
    
    Args:
        history_df: Combined history with columns: city, show_title, single_tickets, canonical_title
    
    Returns:
        DataFrame with ticket priors per canonical_title
    """
    if history_df.empty or "canonical_title" not in history_df.columns:
        return pd.DataFrame()
    
    if "single_tickets" not in history_df.columns:
        print("Warning: No single_tickets column found in history")
        return pd.DataFrame()
    
    # Group by (canonical_title, start_date) to get run-level totals across both cities
    # Then aggregate across all runs for that title
    df = history_df.copy()
    
    # Create run identifier (title + start_date)
    df["_run_id"] = df["canonical_title"].astype(str) + "_" + df["start_date"].astype(str)
    
    # Sum tickets across cities for each run
    run_totals = df.groupby("_run_id").agg(
        canonical_title=("canonical_title", "first"),
        run_total_tickets=("single_tickets", "sum")
    ).reset_index()
    
    # Aggregate across all runs per title
    priors = run_totals.groupby("canonical_title").agg(
        prior_total_tickets=("run_total_tickets", "sum"),
        prior_run_count=("run_total_tickets", "count"),
        ticket_median_prior=("run_total_tickets", "median"),
        ticket_mean_prior=("run_total_tickets", "mean"),
        ticket_std_prior=("run_total_tickets", "std"),
    ).reset_index()
    
    return priors


def compute_remount_features(
    baselines_df: pd.DataFrame,
    history_df: pd.DataFrame,
    reference_date: Optional[date] = None
) -> pd.DataFrame:
    """
    Compute remount-related features for each title from combined history.
    
    Features:
    - years_since_last_run: Numeric years since the most recent run
    - is_remount_recent: Boolean, remounted within 2 years
    - is_remount_medium: Boolean, remounted between 2-4 years
    - run_count_prior: Count of prior runs (distinct start_date values)
    
    Args:
        baselines_df: DataFrame with canonical_title column
        history_df: Combined history with canonical_title, start_date, end_date columns
        reference_date: Reference date for computing years_since_last_run
    """
    if baselines_df.empty:
        return pd.DataFrame()
    
    reference_date = reference_date or date.today()
    
    df = baselines_df[["canonical_title"]].drop_duplicates().copy()
    df["years_since_last_run"] = np.nan
    df["is_remount_recent"] = False
    df["is_remount_medium"] = False
    df["run_count_prior"] = 0
    df["last_run_month"] = np.nan
    
    if history_df.empty or "canonical_title" not in history_df.columns:
        return df
    
    # Count distinct runs per title (group by title + start_date)
    runs_per_title = history_df[history_df["start_date"].notna()].groupby("canonical_title")["start_date"].nunique()
    
    # Compute per title
    for idx, row in df.iterrows():
        canonical = row["canonical_title"]
        runs = history_df[history_df["canonical_title"] == canonical]
        
        if runs.empty:
            continue
        
        # Count distinct runs (unique start dates)
        if canonical in runs_per_title:
            df.at[idx, "run_count_prior"] = runs_per_title[canonical]
        
        # Find most recent run (use end_date, fall back to start_date)
        if "end_date" in runs.columns:
            valid_dates = runs["end_date"].dropna()
            if valid_dates.empty and "start_date" in runs.columns:
                valid_dates = runs["start_date"].dropna()
                
            if not valid_dates.empty:
                last_run = valid_dates.max()
                if pd.notna(last_run):
                    last_run_date = last_run.date() if hasattr(last_run, 'date') else last_run
                    years_since = (reference_date - last_run_date).days / 365.25
                    df.at[idx, "years_since_last_run"] = years_since
                    df.at[idx, "is_remount_recent"] = years_since <= 2
                    df.at[idx, "is_remount_medium"] = 2 < years_since <= 4
                    df.at[idx, "last_run_month"] = last_run.month
    
    return df


def build_modelling_dataset(
    history_path: str = "data/productions/history_city_sales.csv",
    baselines_path: str = "data/productions/baselines.csv",
    output_path: str = "data/modelling_dataset.csv",
    diagnostics_path: str = "diagnostics/modelling_dataset_report.json",
    external_data_dir: str = "data",
    reference_date: Optional[date] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Build the leak-free modelling dataset.
    
    This function:
    1. Loads baseline signals and combined historical sales (city-level rows)
    2. Joins datasets using canonicalized title matching
    3. Computes forecast-time-safe features only
    4. Validates that no current-run ticket columns are included as predictors
    5. Outputs the dataset and diagnostics
    
    Args:
        history_path: Path to combined historical sales CSV (city-level rows with dates)
        baselines_path: Path to baselines CSV
        output_path: Path to save the modelling dataset
        diagnostics_path: Path to save the diagnostics JSON
        external_data_dir: Directory containing optional external CSVs
        reference_date: Reference date for computing temporal features
        verbose: Print progress messages
        
    Returns:
        The modelling dataset as a DataFrame
    """
    diagnostics: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "script": "scripts/build_modelling_dataset.py",
        "inputs": {},
        "outputs": {},
        "warnings": [],
        "unmatched_titles": [],
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("Building Modelling Dataset (Leak-Free)")
        print("=" * 60)
    
    # 1. Load input data
    if verbose:
        print("\n1. Loading input data...")
    
    history_df = load_combined_history(history_path)
    baselines_df = load_baselines(baselines_path)
    
    diagnostics["inputs"]["history_rows"] = len(history_df)
    diagnostics["inputs"]["baseline_titles"] = len(baselines_df)
    
    # Count unique runs in combined history
    if not history_df.empty and "start_date" in history_df.columns:
        unique_runs = history_df[history_df["start_date"].notna()].groupby(["canonical_title", "start_date"]).ngroups
        diagnostics["inputs"]["unique_runs"] = unique_runs
    else:
        unique_runs = 0
        diagnostics["inputs"]["unique_runs"] = 0
    
    if verbose:
        print(f"  - History: {len(history_df)} city-level rows")
        print(f"  - Baselines: {len(baselines_df)} titles")
        print(f"  - Unique runs: {unique_runs} (from combined history)")
    
    # 2. Load optional external data
    if verbose:
        print("\n2. Checking for optional external data...")
    
    weather_df = load_optional_csv(
        os.path.join(external_data_dir, "economics", "weather_daily_city.csv"),
        "weather_daily_city"
    )
    events_df = load_optional_csv(
        os.path.join(external_data_dir, "audiences", "city_events_calendar.csv"),
        "city_events_calendar"
    )
    marketing_df = load_optional_csv(
        os.path.join(external_data_dir, "productions", "marketing_spend_per_ticket.csv"),
        "marketing_spend_per_ticket"
    )
    macro_df = load_optional_csv(
        os.path.join(external_data_dir, "economics", "alberta_macro.csv"),
        "alberta_macro"
    )
    
    # Load economic data for feature store
    if verbose:
        print("\n  Loading economic data for feature store...")
    
    nanos_confidence_df = load_nanos_consumer_confidence(fallback_empty=True)
    nanos_better_off_df = load_nanos_better_off(fallback_empty=True)
    nanos_arts_donors_df = load_nanos_arts_donors(fallback_empty=True)
    commodity_df = load_commodity_price_index(fallback_empty=True)
    cpi_df = load_boc_cpi_monthly(fallback_empty=True)
    
    # Load census data for both cities (pass city name, not path)
    calgary_census_df = load_census_data("Calgary", fallback_empty=True)
    edmonton_census_df = load_census_data("Edmonton", fallback_empty=True)
    
    if verbose:
        econ_loaded = sum([not df.empty for df in [nanos_confidence_df, nanos_better_off_df, commodity_df, cpi_df]])
        census_loaded = sum([not df.empty for df in [calgary_census_df, edmonton_census_df]])
        research_loaded = sum([not df.empty for df in [nanos_arts_donors_df]])
        print(f"  - Loaded {econ_loaded}/4 economic data sources")
        print(f"  - Loaded {census_loaded}/2 census data sources")
        print(f"  - Loaded {research_loaded}/1 research data sources")
    
    # 3. Compute ticket priors from historyres via build_feature_store
    if verbose:
        print("\n3. Building feature store with economic data...")
    
    # Apply economic features to history before computing priors
    # This adds: consumer_confidence_*, energy_index, inflation_adjustment_factor, city_*
    history_enriched = build_feature_store(
        history_df=history_df,
        nanos_confidence_df=nanos_confidence_df,
        nanos_better_off_df=nanos_better_off_df,
        commodity_df=commodity_df,
        cpi_df=cpi_df,
        calgary_census_df=calgary_census_df,
        edmonton_census_df=edmonton_census_df,
        date_column="opening_date",  # Try opening_date if exists, otherwise auto-detect
        city_column="city",
        output_path=None  # Don't write intermediate file
    )
    
    # Apply date-based feature derivation (year, month, day of week, season, etc.)
    # These features are derived from start_date/end_date which are known at forecast time
    if verbose:
        print("\n3b. Deriving date-based features...")
    
    history_enriched = derive_date_features(
        history_enriched,
        start_date_col='start_date',
        end_date_col='end_date'
    )
    
    if verbose:
        date_cols = [c for c in history_enriched.columns if any([
            'opening_' in c,
            'run_duration' in c
        ])]
        print(f"  - Added {len(date_cols)} date-based feature columns")
        if date_cols:
            print(f"  - Date features: {', '.join(date_cols[:6])}{'...' if len(date_cols) > 6 else ''}")
    
    if verbose:
        econ_cols = [c for c in history_enriched.columns if any([
            'consumer_confidence' in c,
            'energy_index' in c,
            'inflation_adjustment' in c,
            'city_population' in c,
            'city_median_household_income' in c
        ])]
        print(f"  - Added {len(econ_cols)} economic feature columns")
        if econ_cols:
            print(f"  - Economic features: {', '.join(econ_cols[:5])}{'...' if len(econ_cols) > 5 else ''}")
    
    # 4. Compute ticket priors from enriched history
    if verbose:
        print("\n4. Computing ticket priors from history...")
    
    ticket_priors = compute_ticket_priors(history_enriched)
    if verbose:
        print(f"  - Computed priors for {len(ticket_priors)} unique titles")
    
    # 5. Computing remount features
    if verbose:
        print("\n5. Computing remount features...")
    
    remount_features = compute_remount_features(
        baselines_df, history_df, reference_date
    )
    if verbose:
        print(f"  - Computed remount features for {len(remount_features)} titles")
    
    # 6. Build the modelling dataset
    if verbose:
        print("\n6. Building modelling dataset...")
    
    if baselines_df.empty:
        print("ERROR: No baseline data available")
        return pd.DataFrame()
    
    # Start with baselines as the base
    df = baselines_df.copy()
    
    # Ensure baseline signal columns are present and properly named
    signal_cols = ["wiki", "trends", "youtube", "spotify"]
    for col in signal_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Add category and gender if present
    if "category" not in df.columns:
        df["category"] = "unknown"
    if "gender" not in df.columns:
        df["gender"] = "na"
    
    # 7. Join with ticket priors
    if verbose:
        print("\n7. Joining datasets...")
    
    if not ticket_priors.empty:
        df = df.merge(
            ticket_priors,
            on="canonical_title",
            how="left"
        )
        matched_priors = df["prior_total_tickets"].notna().sum()
        if verbose:
            print(f"  - Matched {matched_priors}/{len(df)} titles with ticket history")
    
    # Join economic features, date-based features, and month_of_opening from enriched history
    # Extract relevant columns for each title (aggregate if needed)
    if not history_enriched.empty and "canonical_title" in history_enriched.columns:
        # Economic features
        feature_cols_to_join = [c for c in history_enriched.columns if any([
            'consumer_confidence' in c,
            'energy_index' in c,
            'inflation_adjustment' in c,
            'city_population' in c,
            'city_median_household_income' in c,
            'city_calgary' in c,
            'city_edmonton' in c
        ])]
        
        # Date-based features (from derive_date_features)
        date_feature_cols = [c for c in history_enriched.columns if any([
            c.startswith('opening_'),
            c == 'run_duration_days'
        ])]
        feature_cols_to_join.extend(date_feature_cols)
        
        # Also include month_of_opening if available
        if 'month_of_opening' in history_enriched.columns:
            feature_cols_to_join.append('month_of_opening')
        
        # Remove duplicates while preserving order
        feature_cols_to_join = list(dict.fromkeys(feature_cols_to_join))
        
        if feature_cols_to_join:
            # Aggregate features by canonical_title (take mean for numeric, mode for categorical)
            agg_features = history_enriched.groupby("canonical_title")[feature_cols_to_join].agg(
                {col: 'mean' if history_enriched[col].dtype in ['int64', 'float64', 'Int64'] else 'first' 
                 for col in feature_cols_to_join}
            ).reset_index()
            
            df = df.merge(
                agg_features,
                on="canonical_title",
                how="left"
            )
            
            if verbose:
                month_populated = df['month_of_opening'].notna().sum() if 'month_of_opening' in df.columns else 0
                print(f"  - Joined {len(feature_cols_to_join)} economic/temporal/date features")
                if month_populated > 0:
                    print(f"  - month_of_opening populated for {month_populated}/{len(df)} rows")
                # Report date features specifically
                date_features_present = [c for c in date_feature_cols if c in df.columns]
                if date_features_present:
                    print(f"  - Date features joined: {len(date_features_present)} columns")
    
    # 8. Join with remount features
    if not remount_features.empty and "canonical_title" in remount_features.columns:
        df = df.merge(
            remount_features,
            on="canonical_title",
            how="left",
            suffixes=("", "_remount")
        )
        # Handle duplicate columns from merge
        dup_cols = [c for c in df.columns if c.endswith("_remount")]
        if dup_cols:
            df = df.drop(columns=dup_cols)
    
    # 9. Add seasonality features
    if verbose:
        print("\n8. Adding seasonality features...")
    
    # month_of_opening should already be present from enriched history
    # Only fill from last_run_month if missing
    if "month_of_opening" not in df.columns or df["month_of_opening"].isna().all():
        if "last_run_month" in df.columns:
            df["month_of_opening"] = df["last_run_month"]
        else:
            df["month_of_opening"] = np.nan
    elif "last_run_month" in df.columns:
        # Fill any missing month_of_opening values with last_run_month
        df["month_of_opening"] = df["month_of_opening"].fillna(df["last_run_month"])
    
    # Holiday flag based on month
    holiday_months = {11, 12, 1}  # Nov, Dec, Jan
    df["holiday_flag"] = df["month_of_opening"].apply(
        lambda x: 1 if pd.notna(x) and int(x) in holiday_months else 0
    )
    
    # 10. Fill missing values with sensible defaults
    if verbose:
        print("\n9. Handling missing values...")
    
    # Signal columns: fill with median
    for col in signal_cols:
        if col in df.columns:
            median_val = df[col].median() if df[col].notna().any() else 50
            df[col] = df[col].fillna(median_val)
    
    # Prior tickets: fill with 0 (no history)
    prior_cols = ["prior_total_tickets", "prior_run_count", "ticket_median_prior", "ticket_mean_prior"]
    for col in prior_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Remount flags: fill with False/0
    for col in ["is_remount_recent", "is_remount_medium"]:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(int)
    
    # Years since last run: fill with large value (treat as new)
    if "years_since_last_run" in df.columns:
        df["years_since_last_run"] = df["years_since_last_run"].fillna(10)
    
    # 11. Add city median household income from census data
    # Since history is in wide format without a city column, we add this as a static demographic feature
    # All titles get both Calgary and Edmonton income values for context
    try:
        from data.features import _extract_census_median_income
        
        calgary_income = _extract_census_median_income(calgary_census_df) if not calgary_census_df.empty else 100000
        edmonton_income = _extract_census_median_income(edmonton_census_df) if not edmonton_census_df.empty else 96000
        
        # Use average of both cities as a general Alberta market income indicator
        # This provides economic context without needing per-title city assignment
        df['city_median_household_income'] = (calgary_income + edmonton_income) / 2.0
        
        if verbose and len(df) > 0:
            income_val = df['city_median_household_income'].iloc[0]
            print(f"  - Added city_median_household_income: ${income_val:,.0f}")
    except Exception as e:
        if verbose:
            print(f"  - Warning: Could not extract census income: {e}")
        df['city_median_household_income'] = 98000  # Default Alberta market average
    
    # 11b. Add Live Analytics engagement factor by category
    # This provides audience engagement adjustment based on historical category performance
    try:
        from data.loader import get_category_engagement_factor
        
        # Apply engagement factor to each row based on its category
        df['aud__engagement_factor'] = df['category'].apply(
            lambda cat: get_category_engagement_factor(cat) if pd.notna(cat) else 1.0
        )
        
        if verbose and len(df) > 0:
            avg_engagement = df['aud__engagement_factor'].mean()
            print(f"  - Added aud__engagement_factor: mean={avg_engagement:.3f}")
    except Exception as e:
        if verbose:
            print(f"  - Warning: Could not add engagement factor: {e}")
        df['aud__engagement_factor'] = 1.0  # Default neutral factor
    
    # 11c. Add Nanos arts donor research feature by year
    # This provides arts giving context based on the show's year
    try:
        if not nanos_arts_donors_df.empty:
            # Extract year from years_since_last_run to estimate show year
            # For shows with history, estimate year based on current year minus years_since_last_run
            from datetime import date
            current_year = date.today().year
            
            def estimate_year(row):
                if pd.notna(row.get('years_since_last_run')) and row.get('prior_total_tickets', 0) > 0:
                    years_back = row['years_since_last_run']
                    estimated_year = current_year - int(years_back)
                    return max(2020, min(current_year, estimated_year))
                else:
                    return current_year  # Default to current year for cold starts
            
            df['_estimated_year'] = df.apply(estimate_year, axis=1)
            
            # Join arts donor data by year
            df = df.merge(
                nanos_arts_donors_df[['year', 'res__arts_share_giving']],
                left_on='_estimated_year',
                right_on='year',
                how='left'
            )
            
            # Drop temporary columns
            df = df.drop(columns=['_estimated_year', 'year'], errors='ignore')
            
            # Fill missing with median
            if 'res__arts_share_giving' in df.columns:
                median_giving = df['res__arts_share_giving'].median()
                if pd.isna(median_giving):
                    median_giving = 11.5  # Default based on 2023-2025 average
                df['res__arts_share_giving'] = df['res__arts_share_giving'].fillna(median_giving)
                
                if verbose and len(df) > 0:
                    avg_giving = df['res__arts_share_giving'].mean()
                    print(f"  - Added res__arts_share_giving: mean={avg_giving:.1f}%")
        else:
            df['res__arts_share_giving'] = 11.5  # Default if no data
            if verbose:
                print(f"  - Added res__arts_share_giving: using default 11.5%")
    except Exception as e:
        if verbose:
            print(f"  - Warning: Could not add arts donor research feature: {e}")
        df['res__arts_share_giving'] = 11.5  # Default
    
    # 12. Fill missing values for economic features
    econ_feature_names = [
        'consumer_confidence_headline', 'consumer_confidence_prairies',
        'energy_index', 'inflation_adjustment_factor',
        'city_population', 'city_median_household_income',
        'city_calgary', 'city_edmonton'
    ]
    
    for col in econ_feature_names:
        if col in df.columns:
            if 'confidence' in col:
                df[col] = df[col].fillna(50.0)  # Neutral baseline
            elif 'energy_index' in col:
                df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 100.0)
            elif 'inflation' in col:
                df[col] = df[col].fillna(1.0)  # No adjustment
            elif 'population' in col:
                df[col] = df[col].fillna(0)  # Will be handled by city flags
            elif 'income' in col:
                # Already set above, but fill any remaining NaNs
                df[col] = df[col].fillna(98000)  # Alberta market average
            elif 'city_' in col:
                df[col] = df[col].fillna(0)  # Binary flags default to 0
    
    # Fill missing values for date-based features
    date_feature_defaults = {
        'opening_year': 2024,  # Default to current year
        'opening_month': 1,
        'opening_day_of_week': 0,
        'opening_week_of_year': 1,
        'opening_quarter': 1,
        'opening_season': 'winter',
        'opening_is_winter': 0,
        'opening_is_spring': 0,
        'opening_is_summer': 0,
        'opening_is_autumn': 0,
        'opening_is_holiday_season': 0,
        'opening_is_weekend': 0,
        'run_duration_days': 7  # Default to typical week-long run
    }
    
    for col, default_val in date_feature_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default_val)
    
    # 13. Select final feature columns
    if verbose:
        print("\n11. Selecting final features...")
    
    feature_cols = [
        # Identifiers
        "title", "canonical_title", "category", "gender",
        # Baseline signals (forecast-time safe)
        "wiki", "trends", "youtube", "spotify",
        # Lagged historical features (from prior seasons - safe)
        "prior_total_tickets", "prior_run_count", "ticket_median_prior",
        # Remount features (safe - derived from prior runs)
        "years_since_last_run", "is_remount_recent", "is_remount_medium", "run_count_prior",
        # Seasonality (safe - based on planned timing)
        "month_of_opening", "holiday_flag",
        # Date-based features (safe - derived from planned run dates)
        "opening_year", "opening_month", "opening_day_of_week", "opening_week_of_year",
        "opening_quarter", "opening_season",
        "opening_is_winter", "opening_is_spring", "opening_is_summer", "opening_is_autumn",
        "opening_is_holiday_season", "opening_is_weekend",
        "run_duration_days",
        "opening_date",
        # Economic features (safe - macro context known at forecast time)
        "consumer_confidence_prairies", "energy_index",
        "inflation_adjustment_factor", "city_median_household_income",
        # Audience analytics (safe - derived from historical category engagement)
        "aud__engagement_factor",
        # Research features (safe - donor research data by year)
        "res__arts_share_giving",
    ]
    
    # Add target column if we have ticket priors
    if "ticket_median_prior" in df.columns and df["ticket_median_prior"].sum() > 0:
        # Use median prior tickets as target (for training on historical data)
        df["target_ticket_median"] = df["ticket_median_prior"]
    
    # Keep only columns that exist
    final_cols = [c for c in feature_cols if c in df.columns]
    if "target_ticket_median" in df.columns:
        final_cols.append("target_ticket_median")
    
    df_final = df[final_cols].copy()
    
    # 14. SAFETY CHECK: Ensure no forbidden columns
    if verbose:
        print("\n12. Running safety checks...")
    
    # Get feature columns (everything except identifiers and target)
    non_feature_cols = {"title", "canonical_title", "target_ticket_median"}
    feature_only_cols = [c for c in df_final.columns if c not in non_feature_cols]
    
    try:
        assert_no_leakage(df_final, feature_only_cols, "modelling_dataset")
        if verbose:
            print("  ✓ No data leakage detected")
    except AssertionError as e:
        print(f"  ✗ ERROR: {e}")
        raise
    
    # Assert economic features are present
    expected_econ_features = ['consumer_confidence_prairies', 'energy_index', 
                             'inflation_adjustment_factor', 'city_median_household_income']
    present_econ_features = [f for f in expected_econ_features if f in df_final.columns]
    
    if verbose:
        print(f"  ✓ Economic features present: {len(present_econ_features)}/{len(expected_econ_features)}")
        if len(present_econ_features) < len(expected_econ_features):
            missing = set(expected_econ_features) - set(present_econ_features)
            print(f"  ⚠ Missing economic features: {missing}")
    
    # 15. Compute diagnostics
    if verbose:
        print("\n13. Computing diagnostics...")
    
    # Track unmatched titles
    unmatched = []
    if not ticket_priors.empty:
        history_canonicals = set(ticket_priors["canonical_title"])
        for _, row in df_final.iterrows():
            if row["canonical_title"] not in history_canonicals:
                unmatched.append(row.get("title", row["canonical_title"]))
    
    diagnostics["unmatched_titles"] = unmatched[:50]  # First 50
    diagnostics["outputs"]["total_rows"] = len(df_final)
    diagnostics["outputs"]["feature_columns"] = feature_only_cols
    diagnostics["outputs"]["titles_with_history"] = int(df_final["prior_total_tickets"].gt(0).sum())
    diagnostics["outputs"]["titles_without_history"] = int(df_final["prior_total_tickets"].eq(0).sum())
    
    # Missingness per column
    diagnostics["outputs"]["missingness"] = {
        col: float(df_final[col].isna().mean())
        for col in df_final.columns
    }
    
    # 16. Save outputs
    if verbose:
        print("\n14. Saving outputs...")
    
    # Create directories
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(diagnostics_path) or ".", exist_ok=True)
    
    # Save dataset
    df_final.to_csv(output_path, index=False)
    if verbose:
        print(f"  ✓ Saved dataset to {output_path}")
        print(f"    - {len(df_final)} rows, {len(df_final.columns)} columns")
    
    # Save diagnostics
    with open(diagnostics_path, "w") as f:
        json.dump(diagnostics, f, indent=2, default=str)
    if verbose:
        print(f"  ✓ Saved diagnostics to {diagnostics_path}")
    
    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Total titles in dataset: {len(df_final)}")
        print(f"Titles with ticket history: {diagnostics['outputs']['titles_with_history']}")
        print(f"Titles without history (cold start): {diagnostics['outputs']['titles_without_history']}")
        if unmatched:
            print(f"Unmatched titles (first 5): {unmatched[:5]}")
        print(f"\nFeature columns ({len(feature_only_cols)}):")
        for col in feature_only_cols:
            print(f"  - {col}")
    
    return df_final


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build leak-free modelling dataset for ticket demand prediction"
    )
    parser.add_argument(
        "--history", 
        default="data/productions/history_city_sales.csv",
        help="Path to combined historical sales CSV (city-level rows with dates)"
    )
    parser.add_argument(
        "--baselines", 
        default="data/productions/baselines.csv",
        help="Path to baselines CSV"
    )
    parser.add_argument(
        "--output", 
        default="data/modelling_dataset.csv",
        help="Output path for modelling dataset"
    )
    parser.add_argument(
        "--diagnostics", 
        default="diagnostics/modelling_dataset_report.json",
        help="Output path for diagnostics JSON"
    )
    parser.add_argument(
        "--external-dir", 
        default="data",
        help="Directory containing optional external CSVs"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    try:
        build_modelling_dataset(
            history_path=args.history,
            baselines_path=args.baselines,
            output_path=args.output,
            diagnostics_path=args.diagnostics,
            external_data_dir=args.external_dir,
            verbose=not args.quiet
        )
    except AssertionError as e:
        print(f"\nFATAL ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
