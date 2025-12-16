"""
Social buzz/engagement feature engineering for ballet show predictions.

This module loads social media and online engagement metrics (Wikipedia, Google Trends, 
YouTube, Chartmetric) and merges them onto show data based on title matching.

These features capture the cultural awareness and popularity of each ballet title.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import warnings

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data"
PRODUCTIONS_DIR = DATA_DIR / "productions"


def load_baselines_data() -> pd.DataFrame:
    """
    Load baselines data containing social/buzz metrics for ballet titles.
    
    Returns:
        DataFrame with columns: title, wiki, trends, youtube, chartmetric, category, gender, source
    """
    path = PRODUCTIONS_DIR / "baselines.csv"
    if not path.exists():
        warnings.warn(f"Baselines data not found at {path}")
        return pd.DataFrame()
    
    df = pd.read_csv(path)
    
    # Normalize title column for matching (lowercase, strip whitespace)
    if 'title' in df.columns:
        df['title_normalized'] = df['title'].str.lower().str.strip()
    
    return df


def normalize_title(title: str) -> str:
    """
    Normalize a title for matching (lowercase, strip whitespace).
    
    Args:
        title: Raw title string
        
    Returns:
        Normalized title string
    """
    if pd.isna(title):
        return ""
    return str(title).lower().strip()


def compute_wiki_idx(value: float) -> float:
    """
    Compute Wikipedia engagement index.
    
    The raw value is already a 0-100 score representing Wikipedia pageviews,
    article size, and edit activity.
    
    Args:
        value: Raw Wikipedia score (0-100)
        
    Returns:
        WikiIdx normalized score (0-100)
    """
    if pd.isna(value):
        return 0.0
    return float(value)


def compute_trends_idx(value: float) -> float:
    """
    Compute Google Trends engagement index.
    
    The raw value represents search interest over time (0-100 scale).
    
    Args:
        value: Raw Google Trends score (0-100)
        
    Returns:
        TrendsIdx normalized score (0-100)
    """
    if pd.isna(value):
        return 0.0
    return float(value)


def compute_youtube_idx(value: float) -> float:
    """
    Compute YouTube engagement index.
    
    The raw value represents video views, engagement metrics (0-100 scale).
    
    Args:
        value: Raw YouTube score (0-100)
        
    Returns:
        YouTubeIdx normalized score (0-100)
    """
    if pd.isna(value):
        return 0.0
    return float(value)


def compute_chartmetric_idx(value: float) -> float:
    """
    Compute Chartmetric engagement index.
    
    The raw value represents track popularity and streaming metrics (0-100 scale).
    
    Args:
        value: Raw Chartmetric score (0-100)
        
    Returns:
        ChartmetricIdx normalized score (0-100)
    """
    if pd.isna(value):
        return 0.0
    return float(value)


def add_buzz_features(
    df: pd.DataFrame, 
    title_column: str = 'show_title'
) -> pd.DataFrame:
    """
    Add social buzz/engagement features to a DataFrame.
    
    This function merges social media metrics onto show data based on title matching:
    - WikiIdx: Wikipedia engagement (pageviews, article quality)
    - TrendsIdx: Google Trends search interest
    - YouTubeIdx: YouTube video engagement
    - ChartmetricIdx: Chartmetric streaming popularity
    
    Args:
        df: DataFrame with show data
        title_column: Name of the column containing show titles (default: 'show_title')
        
    Returns:
        DataFrame with buzz feature columns added (original df is not modified)
        
    Raises:
        ValueError: If the specified title column is not found in the DataFrame
    """
    if df.empty:
        return df.copy()
    
    if title_column not in df.columns:
        raise ValueError(
            f"Column '{title_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Create a copy to avoid modifying the original
    df_out = df.copy()
    
    # Load baselines data
    baselines = load_baselines_data()
    
    if baselines.empty:
        warnings.warn("No baselines data available. Setting all buzz features to 0.")
        df_out['WikiIdx'] = 0.0
        df_out['TrendsIdx'] = 0.0
        df_out['YouTubeIdx'] = 0.0
        df_out['ChartmetricIdx'] = 0.0
        return df_out
    
    # Normalize titles for matching
    df_out['_title_normalized'] = df_out[title_column].apply(normalize_title)
    
    # Prepare baselines data for merging
    baselines_subset = baselines[['title_normalized', 'wiki', 'trends', 'youtube', 'chartmetric']].copy()
    
    # Merge on normalized title
    df_merged = df_out.merge(
        baselines_subset,
        left_on='_title_normalized',
        right_on='title_normalized',
        how='left',
        suffixes=('', '_baselines')
    )
    
    # Compute indices (fill missing with 0 - no buzz)
    df_merged['WikiIdx'] = df_merged['wiki'].apply(compute_wiki_idx).fillna(0.0)
    df_merged['TrendsIdx'] = df_merged['trends'].apply(compute_trends_idx).fillna(0.0)
    df_merged['YouTubeIdx'] = df_merged['youtube'].apply(compute_youtube_idx).fillna(0.0)
    df_merged['ChartmetricIdx'] = df_merged['chartmetric'].apply(compute_chartmetric_idx).fillna(0.0)
    
    # Select output columns (drop merge artifacts)
    output_cols = [col for col in df_out.columns if col != '_title_normalized'] + \
                  ['WikiIdx', 'TrendsIdx', 'YouTubeIdx', 'ChartmetricIdx']
    
    df_out = df_merged[output_cols].copy()
    
    # Log matching statistics
    matched = (df_out['WikiIdx'] > 0).sum()
    total = len(df_out)
    if matched < total:
        warnings.warn(
            f"Only {matched}/{total} shows matched to baselines data. "
            f"Missing shows will have buzz features = 0."
        )
    
    return df_out


def compute_composite_buzz_score(
    df: pd.DataFrame,
    weights: Optional[dict] = None
) -> pd.DataFrame:
    """
    Compute a composite buzz score from individual indices.
    
    Args:
        df: DataFrame with WikiIdx, TrendsIdx, YouTubeIdx, ChartmetricIdx columns
        weights: Optional dictionary with weights for each index
                Default: {'wiki': 0.25, 'trends': 0.25, 'youtube': 0.25, 'chartmetric': 0.25}
        
    Returns:
        DataFrame with 'CompositeBuzzScore' column added
    """
    if df.empty:
        return df.copy()
    
    df_out = df.copy()
    
    # Default weights (equal)
    if weights is None:
        weights = {
            'wiki': 0.25,
            'trends': 0.25,
            'youtube': 0.25,
            'chartmetric': 0.25
        }
    
    # Verify required columns exist
    required = ['WikiIdx', 'TrendsIdx', 'YouTubeIdx', 'ChartmetricIdx']
    missing = [col for col in required if col not in df_out.columns]
    
    if missing:
        warnings.warn(f"Missing buzz features: {missing}. Cannot compute composite score.")
        df_out['CompositeBuzzScore'] = 0.0
        return df_out
    
    # Compute weighted average
    df_out['CompositeBuzzScore'] = (
        df_out['WikiIdx'] * weights['wiki'] +
        df_out['TrendsIdx'] * weights['trends'] +
        df_out['YouTubeIdx'] * weights['youtube'] +
        df_out['ChartmetricIdx'] * weights['chartmetric']
    )
    
    return df_out


def get_feature_names() -> list:
    """
    Get the list of feature names created by this module.
    
    Returns:
        List of buzz feature column names
    """
    return ['WikiIdx', 'TrendsIdx', 'YouTubeIdx', 'ChartmetricIdx']
