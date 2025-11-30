"""
Time-aware splitting utilities for preventing data leakage.

This module provides utilities for ensuring train/test splits are strictly
chronological, so that future data never leaks into predictions for past events.

Key guarantees:
1. All train dates are earlier than all test dates
2. No row index overlap between train and test sets
3. Train preprocessing (scalers, encoders) is fit only on training data
"""

from __future__ import annotations

from datetime import date
from typing import Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def assert_chronological_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_column: str = "end_date"
) -> None:
    """
    Assert that train/test split is strictly chronological.
    
    Raises AssertionError if:
    - Any train date >= any test date
    - There is row index overlap between train and test
    
    Args:
        train_df: Training data with date column
        test_df: Test data with date column  
        date_column: Name of the date column to check
        
    Raises:
        AssertionError: If chronological split is violated
    """
    # Ensure date column exists
    if date_column not in train_df.columns or date_column not in test_df.columns:
        raise AssertionError(
            f"Date column '{date_column}' not found in both train and test DataFrames. "
            f"Train columns: {list(train_df.columns)}, Test columns: {list(test_df.columns)}"
        )
    
    # Parse dates
    train_dates = pd.to_datetime(train_df[date_column], errors="coerce")
    test_dates = pd.to_datetime(test_df[date_column], errors="coerce")
    
    # Check for chronological ordering
    train_max = train_dates.max()
    test_min = test_dates.min()
    
    if pd.notna(train_max) and pd.notna(test_min):
        if train_max >= test_min:
            raise AssertionError(
                f"TIME LEAKAGE DETECTED: Train max date ({train_max}) >= test min date ({test_min}). "
                f"Train dates must be strictly earlier than test dates."
            )
    
    # Check for index overlap
    train_idx = set(train_df.index)
    test_idx = set(test_df.index)
    overlap = train_idx & test_idx
    
    if overlap:
        raise AssertionError(
            f"INDEX LEAKAGE DETECTED: {len(overlap)} overlapping indices between train and test. "
            f"First 5 overlapping: {list(overlap)[:5]}"
        )


def chronological_train_test_split(
    df: pd.DataFrame,
    date_column: str = "end_date",
    test_ratio: float = 0.2,
    min_test_rows: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically, using the most recent data as test.
    
    Unlike random train_test_split, this ensures:
    - All training data comes before all test data in time
    - No future leakage is possible
    
    Args:
        df: DataFrame with date column
        date_column: Name of date column to sort by
        test_ratio: Fraction of data to use for testing (default 0.2)
        min_test_rows: Minimum number of test rows required
        
    Returns:
        Tuple of (train_df, test_df)
        
    Raises:
        ValueError: If date column not found or not enough data
    """
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found. Columns: {list(df.columns)}")
    
    if len(df) < 2:
        raise ValueError(f"Need at least 2 rows for splitting, got {len(df)}")
    
    # Sort by date
    df_sorted = df.copy()
    df_sorted[date_column] = pd.to_datetime(df_sorted[date_column], errors="coerce")
    df_sorted = df_sorted.sort_values(date_column).reset_index(drop=True)
    
    # Calculate split point
    n_test = max(min_test_rows, int(len(df_sorted) * test_ratio))
    split_idx = len(df_sorted) - n_test
    
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    # Validate the split
    assert_chronological_split(train_df, test_df, date_column)
    
    return train_df, test_df


def rolling_origin_cv_splits(
    df: pd.DataFrame,
    date_column: str = "end_date",
    initial_train_period: str = "730D",
    horizon: str = "180D",
    step: str = "180D"
) -> Generator[Tuple[List[int], List[int]], None, None]:
    """
    Generate rolling-origin cross-validation splits for time series.
    
    This is a walk-forward validation where:
    - Training window expands (or slides) forward
    - Test window always comes after training window
    - No future leakage is possible
    
    Args:
        df: DataFrame with date column
        date_column: Name of date column
        initial_train_period: Initial training window size (pandas timedelta string)
        horizon: Forecast horizon / test window size
        step: Step size between folds
        
    Yields:
        Tuples of (train_indices, test_indices)
    """
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found")
    
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.sort_values(date_column).reset_index(drop=True)
    
    start_date = df[date_column].min()
    end_date = df[date_column].max()
    
    train_end = start_date + pd.Timedelta(initial_train_period)
    
    while train_end + pd.Timedelta(horizon) <= end_date:
        test_end = train_end + pd.Timedelta(horizon)
        
        train_idx = df.index[df[date_column] <= train_end].tolist()
        test_idx = df.index[(df[date_column] > train_end) & (df[date_column] <= test_end)].tolist()
        
        if train_idx and test_idx:
            yield train_idx, test_idx
        
        train_end += pd.Timedelta(step)


class TimeSeriesCVSplitter:
    """
    Time-series cross-validation splitter with chronological guarantees.
    
    Unlike sklearn's KFold with shuffle, this ensures:
    - Training folds always precede test folds in time
    - Assertions validate no time leakage
    
    Compatible with sklearn's cross-validation interface.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        date_column: str = "end_date"
    ):
        """
        Initialize time-series CV splitter.
        
        Args:
            n_splits: Number of cross-validation folds
            date_column: Name of the date column in the DataFrame
        """
        self.n_splits = n_splits
        self.date_column = date_column
        self._df = None
    
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices for each fold.
        
        Uses expanding window: fold 1 trains on first portion,
        fold 2 trains on first two portions, etc.
        
        Args:
            X: Features DataFrame (must have date column for sorting)
            y: Target (ignored, for sklearn compatibility)
            groups: Groups (ignored, for sklearn compatibility)
            
        Yields:
            Tuples of (train_indices, test_indices)
        """
        if self.date_column not in X.columns:
            # If no date column, fall back to sequential split (assumed sorted)
            n = len(X)
            fold_size = n // (self.n_splits + 1)
            
            for i in range(self.n_splits):
                train_end = (i + 1) * fold_size
                test_start = train_end
                test_end = test_start + fold_size
                
                train_idx = np.arange(train_end)
                test_idx = np.arange(test_start, min(test_end, n))
                
                if len(train_idx) > 0 and len(test_idx) > 0:
                    yield train_idx, test_idx
            return
        
        # Sort by date and get indices
        df = X.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column], errors="coerce")
        sorted_indices = df.sort_values(self.date_column).index.values
        
        n = len(sorted_indices)
        fold_size = n // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            train_end = (i + 1) * fold_size
            test_start = train_end
            test_end = test_start + fold_size
            
            train_idx = sorted_indices[:train_end]
            test_idx = sorted_indices[test_start:min(test_end, n)]
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                # Validate chronological ordering
                train_dates = df.loc[train_idx, self.date_column]
                test_dates = df.loc[test_idx, self.date_column]
                
                train_max = train_dates.max()
                test_min = test_dates.min()
                
                if pd.notna(train_max) and pd.notna(test_min) and train_max >= test_min:
                    raise AssertionError(
                        f"TIME LEAKAGE in fold {i+1}: "
                        f"train max date ({train_max}) >= test min date ({test_min})"
                    )
                
                yield train_idx, test_idx
    
    def get_n_splits(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None
    ) -> int:
        """Return the number of splits."""
        return self.n_splits


def assert_group_chronological_split(
    df: pd.DataFrame,
    train_mask: pd.Series,
    date_column: str = "end_date",
    group_column: str = "canonical_title"
) -> None:
    """
    Assert that within each group, train dates precede test dates.
    
    For grouped time series (e.g., multiple shows), this ensures
    that for each show, training data comes before test data.
    
    Args:
        df: Full DataFrame
        train_mask: Boolean mask indicating training rows
        date_column: Name of date column
        group_column: Name of group column
        
    Raises:
        AssertionError: If any group violates chronological ordering
    """
    if group_column not in df.columns:
        # If no group column, just check overall chronology
        train_df = df[train_mask]
        test_df = df[~train_mask]
        assert_chronological_split(train_df, test_df, date_column)
        return
    
    violations = []
    
    for group_name, group_df in df.groupby(group_column):
        train_group = group_df[train_mask.loc[group_df.index]]
        test_group = group_df[~train_mask.loc[group_df.index]]
        
        if train_group.empty or test_group.empty:
            continue
        
        if date_column not in group_df.columns:
            continue
        
        train_dates = pd.to_datetime(train_group[date_column], errors="coerce")
        test_dates = pd.to_datetime(test_group[date_column], errors="coerce")
        
        train_max = train_dates.max()
        test_min = test_dates.min()
        
        if pd.notna(train_max) and pd.notna(test_min) and train_max >= test_min:
            violations.append(
                f"{group_name}: train max={train_max}, test min={test_min}"
            )
    
    if violations:
        raise AssertionError(
            f"TIME LEAKAGE within groups! {len(violations)} groups have train dates >= test dates:\n"
            + "\n".join(violations[:10])  # Show first 10
        )


class GroupedCVSplitter:
    """
    Grouped cross-validation splitter using GroupKFold.
    
    This splitter ensures that groups (e.g., productions with the same title
    or from the same season) are never split across train and test sets within
    the same fold. This prevents optimistic bias from repeated titles.
    
    When a group_column is specified, all rows with the same group value will
    be placed in either train or test, never both.
    
    Compatible with sklearn's cross-validation interface.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        group_column: str = "show_title"
    ):
        """
        Initialize grouped CV splitter.
        
        Args:
            n_splits: Number of cross-validation folds
            group_column: Name of the column to use for grouping
                         (e.g., 'show_title', 'season', 'canonical_title')
        """
        self.n_splits = n_splits
        self.group_column = group_column
        self._gkf = GroupKFold(n_splits=n_splits)
    
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices for each fold.
        
        All rows with the same group value will be in either train or test,
        never both.
        
        Args:
            X: Features DataFrame (must have the group column)
            y: Target (ignored, for sklearn compatibility)
            groups: Optional pre-computed groups. If None, extracted from
                   X[group_column].
            
        Yields:
            Tuples of (train_indices, test_indices)
            
        Raises:
            ValueError: If group_column not found in X and groups not provided
        """
        if groups is None:
            if self.group_column not in X.columns:
                raise ValueError(
                    f"Group column '{self.group_column}' not found in DataFrame. "
                    f"Available columns: {list(X.columns)}"
                )
            groups = X[self.group_column].values
        
        # Use positional indices (0 to len-1) rather than DataFrame index
        # to match sklearn's expected behavior
        indices = np.arange(len(X))
        
        for train_idx, test_idx in self._gkf.split(indices, y, groups):
            yield train_idx, test_idx
    
    def get_n_splits(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None
    ) -> int:
        """Return the number of splits."""
        return self.n_splits
    
    @staticmethod
    def assert_no_group_leakage(
        X: pd.DataFrame,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        group_column: str
    ) -> None:
        """
        Assert that no group appears in both train and test sets.
        
        Args:
            X: Features DataFrame with group column
            train_idx: Training indices
            test_idx: Test indices
            group_column: Name of the group column
            
        Raises:
            AssertionError: If any group appears in both train and test
        """
        if group_column not in X.columns:
            raise ValueError(f"Group column '{group_column}' not found in DataFrame")
        
        train_groups = set(X.iloc[train_idx][group_column].unique())
        test_groups = set(X.iloc[test_idx][group_column].unique())
        
        overlap = train_groups & test_groups
        
        if overlap:
            raise AssertionError(
                f"GROUP LEAKAGE DETECTED: {len(overlap)} groups appear in both "
                f"train and test sets. Overlapping groups: {list(overlap)[:10]}"
            )
