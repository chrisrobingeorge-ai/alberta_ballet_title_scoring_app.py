"""
Title-based feature engineering for ballet show predictions.

This module extracts features from show titles to help predict ticket sales.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import os


def load_live_analytics_mapping(csv_path: str = None) -> Dict[str, float]:
    """
    Load the live analytics CSV and create a mapping of Category -> Addressable Market (cust_cnt).
    
    The CSV has a complex header structure:
    - Row 0: Empty/metadata
    - Row 1: Category names (e.g., pop_ip, family_classic, etc.)
    - Row 2: Contains 'cust_cnt' in first column and customer counts for each category
    
    Args:
        csv_path: Path to live_analytics.csv. If None, uses default path.
        
    Returns:
        Dictionary mapping category names to customer counts (addressable market size)
    """
    if csv_path is None:
        # Default path relative to this module
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, 'data', 'audiences', 'live_analytics.csv')
    
    if not os.path.exists(csv_path):
        # Return empty dict if file doesn't exist
        return {}
    
    try:
        # Read the CSV without header to handle complex structure
        df = pd.read_csv(csv_path, header=None)
        
        # Row 1 (index 1) contains category names
        # Row 2 (index 2) contains 'cust_cnt' in column 0, and counts in other columns
        category_row = df.iloc[1]
        count_row = df.iloc[2]
        
        # Verify we're on the right row by checking for 'cust_cnt'
        if count_row.iloc[1] != 'Customers':
            raise ValueError("Expected 'Customers' label in row 2, column 1")
        
        # Build mapping dictionary
        mapping = {}
        for col_idx in range(2, len(category_row)):
            category = category_row.iloc[col_idx]
            count_str = count_row.iloc[col_idx]
            
            # Skip empty categories or counts
            if pd.isna(category) or pd.isna(count_str) or category == '':
                continue
            
            # Clean category name (strip whitespace)
            category_clean = str(category).strip()
            
            # Parse count (remove commas and convert to float)
            try:
                count = float(str(count_str).replace(',', ''))
                mapping[category_clean] = count
            except (ValueError, AttributeError):
                # Skip if count cannot be parsed
                continue
        
        return mapping
        
    except Exception as e:
        # If anything goes wrong, return empty dict and log warning
        print(f"Warning: Could not load live_analytics.csv: {e}")
        return {}


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
    
    This function creates the following columns:
    - is_benchmark_classic: Binary indicator (1/0) for benchmark classic ballets
    - title_word_count: Number of words in the title
    - LA_AddressableMarket: Addressable market size from live analytics (customer count)
    - LA_AddressableMarket_Norm: Normalized market size (0-1 scale)
    
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
    
    # Generate basic title features
    df_out['is_benchmark_classic'] = df_out[title_column].apply(is_benchmark_classic).astype(int)
    df_out['title_word_count'] = df_out[title_column].apply(count_title_words)
    
    # Load live analytics market size mapping
    la_mapping = load_live_analytics_mapping()
    
    if la_mapping:
        # Calculate median market size for fallback
        market_sizes = list(la_mapping.values())
        median_market_size = np.median(market_sizes) if market_sizes else 0
        max_market_size = max(market_sizes) if market_sizes else 1
        
        # Add addressable market feature
        # Look up category in the mapping, fall back to median if not found
        def get_market_size(row):
            # Check if 'Category' column exists
            if 'Category' in row.index and not pd.isna(row['Category']):
                category = str(row['Category']).strip()
                return la_mapping.get(category, median_market_size)
            return median_market_size
        
        df_out['LA_AddressableMarket'] = df_out.apply(get_market_size, axis=1)
        
        # Normalize to 0-1 scale
        if max_market_size > 0:
            df_out['LA_AddressableMarket_Norm'] = df_out['LA_AddressableMarket'] / max_market_size
        else:
            df_out['LA_AddressableMarket_Norm'] = 0.0
    else:
        # If live analytics file not available, set to NaN or 0
        df_out['LA_AddressableMarket'] = np.nan
        df_out['LA_AddressableMarket_Norm'] = np.nan
    
    return df_out


def get_feature_names() -> List[str]:
    """
    Get the list of feature names created by this module.
    
    Returns:
        List of feature column names
    """
    return [
        'is_benchmark_classic', 
        'title_word_count',
        'LA_AddressableMarket',
        'LA_AddressableMarket_Norm'
    ]
