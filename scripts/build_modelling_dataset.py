#!/usr/bin/env python3
"""
Build Modelling Dataset Script

This script produces a leak-free modelling dataset for training ticket demand models.
It combines baseline signals, historical sales, and contextual features while ensuring
no current-run ticket information is used as a predictor.

IMPORTANT: This script is designed to prevent data leakage. Only forecast-time features
are included as predictors. Current-run ticket columns are only allowed as targets or
for computing lagged historical features from PRIOR seasons.

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


def load_history(path: str = "data/productions/history_city_sales.csv") -> pd.DataFrame:
    """Load and preprocess historical sales data."""
    try:
        df = pd.read_csv(path, thousands=",")
    except FileNotFoundError:
        print(f"Warning: History file not found at {path}")
        return pd.DataFrame()
    
    # Standardize column names
    df.columns = [c.strip() for c in df.columns]
    
    # Find and normalize title column
    title_col = None
    for col in ["Show Title", "show_title", "Title", "title", "Show_Title"]:
        if col in df.columns:
            title_col = col
            break
    
    if title_col and title_col != "show_title":
        df = df.rename(columns={title_col: "show_title"})
    
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


def load_past_runs(path: str = "data/productions/past_runs.csv") -> pd.DataFrame:
    """Load and preprocess past runs data for remount/seasonality features."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Warning: Past runs file not found at {path}")
        return pd.DataFrame()
    
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Parse dates
    for col in ["start_date", "end_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    
    # Add canonical title
    if "title" in df.columns:
        df["canonical_title"] = df["title"].apply(canonicalize_title)
    
    return df


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
    Compute ticket priors from historical data.
    
    For each show title, compute aggregates from historical runs.
    This is safe as we'll only use PRIOR season data during training.
    """
    if history_df.empty or "show_title" not in history_df.columns:
        return pd.DataFrame()
    
    # Find ticket columns
    ticket_cols = []
    for col in history_df.columns:
        col_lower = col.lower()
        if "ticket" in col_lower and "yourmodel" not in col_lower:
            ticket_cols.append(col)
    
    if not ticket_cols:
        print("Warning: No ticket columns found in history")
        return pd.DataFrame()
    
    # Compute total tickets per row
    df = history_df.copy()
    
    # Try to find standard ticket columns
    single_yyc = None
    single_yeg = None
    for col in df.columns:
        col_lower = col.lower().replace(" ", "_").replace("-", "_")
        if "single" in col_lower and "calgary" in col_lower:
            single_yyc = col
        elif "single" in col_lower and "edmonton" in col_lower:
            single_yeg = col
    
    # Compute total
    if single_yyc and single_yeg:
        df["_total_tickets"] = (
            pd.to_numeric(df[single_yyc], errors="coerce").fillna(0) +
            pd.to_numeric(df[single_yeg], errors="coerce").fillna(0)
        )
    else:
        # Try to find Total Single Tickets column
        total_col = None
        for col in df.columns:
            if "total" in col.lower() and "single" in col.lower() and "ticket" in col.lower():
                total_col = col
                break
        if total_col:
            df["_total_tickets"] = pd.to_numeric(df[total_col], errors="coerce").fillna(0)
        else:
            df["_total_tickets"] = 0
    
    # Aggregate by canonical title
    priors = df.groupby("canonical_title").agg(
        prior_total_tickets=("_total_tickets", "sum"),
        prior_run_count=("_total_tickets", "count"),
        ticket_median_prior=("_total_tickets", "median"),
        ticket_mean_prior=("_total_tickets", "mean"),
        ticket_std_prior=("_total_tickets", "std"),
    ).reset_index()
    
    return priors


def compute_remount_features(
    baselines_df: pd.DataFrame,
    past_runs_df: pd.DataFrame,
    reference_date: Optional[date] = None
) -> pd.DataFrame:
    """
    Compute remount-related features for each title.
    
    Features:
    - years_since_last_run: Numeric years since the most recent run
    - is_remount_recent: Boolean, remounted within 2 years
    - is_remount_medium: Boolean, remounted between 2-4 years
    - run_count_prior: Count of prior runs
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
    
    if past_runs_df.empty or "canonical_title" not in past_runs_df.columns:
        return df
    
    # Compute per title
    for idx, row in df.iterrows():
        canonical = row["canonical_title"]
        runs = past_runs_df[past_runs_df["canonical_title"] == canonical]
        
        if runs.empty:
            continue
        
        df.at[idx, "run_count_prior"] = len(runs)
        
        # Find most recent run
        if "end_date" in runs.columns:
            valid_dates = runs["end_date"].dropna()
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
    past_runs_path: str = "data/productions/past_runs.csv",
    output_path: str = "data/modelling_dataset.csv",
    diagnostics_path: str = "diagnostics/modelling_dataset_report.json",
    external_data_dir: str = "data",
    reference_date: Optional[date] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Build the leak-free modelling dataset.
    
    This function:
    1. Loads baseline signals, historical sales, and past run data
    2. Joins datasets using canonicalized title matching
    3. Computes forecast-time-safe features only
    4. Validates that no current-run ticket columns are included as predictors
    5. Outputs the dataset and diagnostics
    
    Args:
        history_path: Path to historical sales CSV
        baselines_path: Path to baselines CSV
        past_runs_path: Path to past runs CSV
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
    
    history_df = load_history(history_path)
    baselines_df = load_baselines(baselines_path)
    past_runs_df = load_past_runs(past_runs_path)
    
    diagnostics["inputs"]["history_rows"] = len(history_df)
    diagnostics["inputs"]["baseline_titles"] = len(baselines_df)
    diagnostics["inputs"]["past_runs"] = len(past_runs_df)
    
    if verbose:
        print(f"  - History: {len(history_df)} rows")
        print(f"  - Baselines: {len(baselines_df)} titles")
        print(f"  - Past runs: {len(past_runs_df)} records")
    
    # 2. Load optional external data
    if verbose:
        print("\n2. Checking for optional external data...")
    
    weather_df = load_optional_csv(
        os.path.join(external_data_dir, "weather_daily_city.csv"),
        "weather_daily_city"
    )
    events_df = load_optional_csv(
        os.path.join(external_data_dir, "city_events_calendar.csv"),
        "city_events_calendar"
    )
    marketing_df = load_optional_csv(
        os.path.join(external_data_dir, "marketing_spend_per_ticket.csv"),
        "marketing_spend_per_ticket"
    )
    macro_df = load_optional_csv(
        os.path.join(external_data_dir, "alberta_macro.csv"),
        "alberta_macro"
    )
    
    # 3. Compute ticket priors from history
    if verbose:
        print("\n3. Computing ticket priors from history...")
    
    ticket_priors = compute_ticket_priors(history_df)
    if verbose:
        print(f"  - Computed priors for {len(ticket_priors)} unique titles")
    
    # 4. Compute remount features
    if verbose:
        print("\n4. Computing remount features...")
    
    remount_features = compute_remount_features(
        baselines_df, past_runs_df, reference_date
    )
    if verbose:
        print(f"  - Computed remount features for {len(remount_features)} titles")
    
    # 5. Build the modelling dataset
    if verbose:
        print("\n5. Building modelling dataset...")
    
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
    
    # 6. Join with ticket priors
    if verbose:
        print("\n6. Joining datasets...")
    
    if not ticket_priors.empty:
        df = df.merge(
            ticket_priors,
            on="canonical_title",
            how="left"
        )
        matched_priors = df["prior_total_tickets"].notna().sum()
        if verbose:
            print(f"  - Matched {matched_priors}/{len(df)} titles with ticket history")
    
    # 7. Join with remount features
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
    
    # 8. Add seasonality features
    if verbose:
        print("\n7. Adding seasonality features...")
    
    if "last_run_month" in df.columns:
        df["month_of_opening"] = df["last_run_month"]
    else:
        df["month_of_opening"] = np.nan
    
    # Holiday flag based on month
    holiday_months = {11, 12, 1}  # Nov, Dec, Jan
    df["holiday_flag"] = df["month_of_opening"].apply(
        lambda x: 1 if pd.notna(x) and int(x) in holiday_months else 0
    )
    
    # 9. Fill missing values with sensible defaults
    if verbose:
        print("\n8. Handling missing values...")
    
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
    
    # 10. Select final feature columns
    if verbose:
        print("\n9. Selecting final features...")
    
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
    
    # 11. SAFETY CHECK: Ensure no forbidden columns
    if verbose:
        print("\n10. Running safety checks...")
    
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
    
    # 12. Compute diagnostics
    if verbose:
        print("\n11. Computing diagnostics...")
    
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
    
    # 13. Save outputs
    if verbose:
        print("\n12. Saving outputs...")
    
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
        help="Path to historical sales CSV"
    )
    parser.add_argument(
        "--baselines", 
        default="data/productions/baselines.csv",
        help="Path to baselines CSV"
    )
    parser.add_argument(
        "--past-runs", 
        default="data/productions/past_runs.csv",
        help="Path to past runs CSV"
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
            past_runs_path=args.past_runs,
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
