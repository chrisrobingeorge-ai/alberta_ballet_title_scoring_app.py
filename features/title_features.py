"""
Title-based feature engineering for ballet show predictions.

This module extracts features from show titles to help predict ticket sales.
"""

import pandas as pd
from typing import List


def is_benchmark_classic(title: str) -> bool:
    """
    Check if a title is a benchmark classic ballet.
    
    Benchmark classics are well-known, high-performing ballets that consistently
    draw strong audiences. 'Cinderella' serves as Alberta Ballet's internal
    benchmark for a standard top-tier show.
    
    Args:
        title: Show title string
        
    Returns:
        True if title contains any benchmark classic keywords, False otherwise
    """
    if pd.isna(title):
        return False
    
    # Convert to lowercase for case-insensitive matching
    title_lower = str(title).lower()
    
    # List of benchmark classic keywords
    benchmark_keywords = [
        'cinderella',
        'swan lake',
        'sleeping beauty',
        'giselle',
        'romeo'  # Matches "Romeo and Juliet"
    ]
    
    return any(keyword in title_lower for keyword in benchmark_keywords)


def count_title_words(title: str) -> int:
    """
    Count the number of words in a show title.
    
    Args:
        title: Show title string
        
    Returns:
        Number of words in the title (0 if title is missing/null)
    """
    if pd.isna(title):
        return 0
    
    # Split on whitespace and count non-empty tokens
    words = str(title).strip().split()
    return len(words)


def add_title_features(df: pd.DataFrame, title_column: str = 'show_title') -> pd.DataFrame:
    """
    Add title-based features to a DataFrame.
    
    This function creates two new columns:
    - is_benchmark_classic: Binary indicator (1/0) for benchmark classic ballets
    - title_word_count: Number of words in the title
    
    Args:
        df: DataFrame containing show data
        title_column: Name of the column containing show titles (default: 'show_title')
        
    Returns:
        DataFrame with new feature columns added (original df is not modified)
        
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
    
    # Create a copy to avoid modifying the original DataFrame
    df_out = df.copy()
    
    # Generate features
    df_out['is_benchmark_classic'] = df_out[title_column].apply(is_benchmark_classic).astype(int)
    df_out['title_word_count'] = df_out[title_column].apply(count_title_words)
    
    return df_out


def get_feature_names() -> List[str]:
    """
    Get the list of feature names created by this module.
    
    Returns:
        List of feature column names
    """
    return ['is_benchmark_classic', 'title_word_count']
