"""
Social buzz/engagement feature engineering for ballet show predictions.

This module loads social media and online engagement metrics (Wikipedia, Google Trends, 
YouTube, Spotify) and merges them onto show data based on title matching.

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
        DataFrame with columns: title, wiki, trends, youtube, spotify, category, gender, source
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


def load_chartmetric_use_case_data() -> pd.DataFrame:
    """
    Load chartmetric use case data with motivation weights.
    
    This file contains the mapping of titles to their chartmetric scores along with
    a 'use' classification (USE, USE WITH CAUTION, DO NOT USE) and corresponding
    multiplier weights (1.0, 0.25, 0.0).
    
    Returns:
        DataFrame with columns: title, use, multiplier, score, chartmetric_search_term, notes
    """
    path = PRODUCTIONS_DIR / "use_case_chartmetrics.csv"
    if not path.exists():
        warnings.warn(f"Chartmetric use case data not found at {path}")
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


def compute_spotify_idx(value: float) -> float:
    """
    Compute Spotify engagement index.
    
    The raw value represents track popularity and streaming metrics (0-100 scale).
    
    Args:
        value: Raw Spotify score (0-100)
        
    Returns:
        SpotifyIdx normalized score (0-100)
    """
    if pd.isna(value):
        return 0.0
    return float(value)


def compute_chartmetric_idx(value: float) -> float:
    """
    Compute Chartmetric engagement index.
    
    The raw value represents music track popularity and streaming metrics (0-100 scale).
    
    Args:
        value: Raw Chartmetric score (0-100)
        
    Returns:
        ChartmetricIdx normalized score (0-100)
    """
    if pd.isna(value):
        return 0.0
    return float(value)


def compute_music_motivation_bonus(chartmetric_score: float, motivation_weight: float) -> float:
    """
    Compute music motivation bonus from chartmetric score and motivation weight.
    
    The motivation weight is derived from the 'use' classification:
    - "USE" = 1.0 (full music motivation signal)
    - "USE WITH CAUTION" = 0.25 (limited music motivation signal)
    - "DO NOT USE" = 0.0 (no music motivation signal)
    
    Args:
        chartmetric_score: Normalized Chartmetric score (0-100)
        motivation_weight: Multiplier based on use classification (0.0, 0.25, or 1.0)
        
    Returns:
        Music motivation bonus score (0-100)
    """
    if pd.isna(chartmetric_score) or pd.isna(motivation_weight):
        return 0.0
    return float(chartmetric_score) * float(motivation_weight)


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
    - SpotifyIdx: Spotify streaming popularity
    - ChartmetricIdx: Music popularity from Chartmetric (raw score)
    - MusicMotivationBonus: Chartmetric score weighted by motivation factor
    
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
        df_out['SpotifyIdx'] = 0.0
        df_out['ChartmetricIdx'] = 0.0
        df_out['MusicMotivationBonus'] = 0.0
        return df_out
    
    # Load chartmetric use case data
    chartmetric_use = load_chartmetric_use_case_data()
    
    # Normalize titles for matching
    df_out['_title_normalized'] = df_out[title_column].apply(normalize_title)
    
    # Prepare baselines data for merging
    # Handle both 'chartmetric' (current) and 'spotify' (legacy) column names
    base_cols = ['title_normalized', 'wiki', 'trends', 'youtube']
    if 'chartmetric' in baselines.columns:
        base_cols.append('chartmetric')
    if 'spotify' in baselines.columns:
        base_cols.append('spotify')
    baselines_subset = baselines[base_cols].copy()
    
    # Merge on normalized title
    df_merged = df_out.merge(
        baselines_subset,
        left_on='_title_normalized',
        right_on='title_normalized',
        how='left',
        suffixes=('', '_baselines')
    )
    
    # Merge chartmetric use case data for motivation weights
    if not chartmetric_use.empty:
        chartmetric_subset = chartmetric_use[['title_normalized', 'multiplier', 'use']].copy()
        df_merged = df_merged.merge(
            chartmetric_subset,
            left_on='_title_normalized',
            right_on='title_normalized',
            how='left',
            suffixes=('', '_chartmetric')
        )
    else:
        warnings.warn("No chartmetric use case data available. Setting motivation weights to 0.")
        df_merged['multiplier'] = 0.0
        df_merged['use'] = 'DO NOT USE'
    
    # Compute indices (fill missing with 0 - no buzz)
    df_merged['WikiIdx'] = df_merged['wiki'].apply(compute_wiki_idx).fillna(0.0)
    df_merged['TrendsIdx'] = df_merged['trends'].apply(compute_trends_idx).fillna(0.0)
    df_merged['YouTubeIdx'] = df_merged['youtube'].apply(compute_youtube_idx).fillna(0.0)
    
    # Handle spotify column (may not exist in all cases)
    if 'spotify' in df_merged.columns:
        df_merged['SpotifyIdx'] = df_merged['spotify'].apply(compute_spotify_idx).fillna(0.0)
    else:
        df_merged['SpotifyIdx'] = 0.0
    
    # Handle chartmetric column
    if 'chartmetric' in df_merged.columns:
        df_merged['ChartmetricIdx'] = df_merged['chartmetric'].apply(compute_chartmetric_idx).fillna(0.0)
    else:
        # Fallback to spotify if chartmetric doesn't exist (for backward compatibility)
        if 'spotify' in df_merged.columns:
            df_merged['ChartmetricIdx'] = df_merged['spotify'].apply(compute_chartmetric_idx).fillna(0.0)
        else:
            df_merged['ChartmetricIdx'] = 0.0
    
    # Compute music motivation bonus
    df_merged['multiplier'] = df_merged['multiplier'].fillna(0.0)
    df_merged['MusicMotivationBonus'] = df_merged.apply(
        lambda row: compute_music_motivation_bonus(row['ChartmetricIdx'], row['multiplier']),
        axis=1
    )
    
    # Select output columns (drop merge artifacts)
    output_cols = [col for col in df_out.columns if col != '_title_normalized'] + \
                  ['WikiIdx', 'TrendsIdx', 'YouTubeIdx', 'SpotifyIdx', 'ChartmetricIdx', 'MusicMotivationBonus']
    
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
        df: DataFrame with WikiIdx, TrendsIdx, YouTubeIdx, SpotifyIdx columns
        weights: Optional dictionary with weights for each index
                Default: {'wiki': 0.25, 'trends': 0.25, 'youtube': 0.25, 'spotify': 0.25}
        
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
            'spotify': 0.25
        }
    
    # Verify required columns exist
    required = ['WikiIdx', 'TrendsIdx', 'YouTubeIdx', 'SpotifyIdx']
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
        df_out['SpotifyIdx'] * weights['spotify']
    )
    
    return df_out


def get_feature_names() -> list:
    """
    Get the list of feature names created by this module.
    
    Returns:
        List of buzz feature column names
    """
    return ['WikiIdx', 'TrendsIdx', 'YouTubeIdx', 'SpotifyIdx', 'ChartmetricIdx', 'MusicMotivationBonus']
