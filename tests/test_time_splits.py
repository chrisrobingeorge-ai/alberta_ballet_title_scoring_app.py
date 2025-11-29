"""
Tests for time-aware splitting utilities.

These tests verify that:
1. Train/test splits are strictly chronological
2. No future data leaks into training
3. Assertions catch time leakage violations
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.time_splits import (
    assert_chronological_split,
    chronological_train_test_split,
    rolling_origin_cv_splits,
    TimeSeriesCVSplitter,
    assert_group_chronological_split,
)


# =============================================================================
# Test fixtures
# =============================================================================

@pytest.fixture
def sample_df():
    """Create sample DataFrame with dates for testing."""
    dates = pd.date_range("2018-01-01", periods=100, freq="7D")
    return pd.DataFrame({
        "end_date": dates,
        "title": [f"Show_{i % 10}" for i in range(100)],
        "canonical_title": [f"show_{i % 10}" for i in range(100)],
        "sales": np.random.randint(1000, 5000, 100),
        "wiki": np.random.randint(20, 80, 100),
    })


@pytest.fixture
def sample_df_no_dates():
    """Create sample DataFrame without date column."""
    return pd.DataFrame({
        "title": [f"Show_{i}" for i in range(50)],
        "sales": np.random.randint(1000, 5000, 50),
    })


# =============================================================================
# Tests for assert_chronological_split
# =============================================================================

def test_assert_chronological_split_valid(sample_df):
    """Test that valid chronological split passes assertion."""
    sorted_df = sample_df.sort_values("end_date")
    train = sorted_df.iloc[:80]
    test = sorted_df.iloc[80:]
    
    # Should not raise
    assert_chronological_split(train, test, "end_date")


def test_assert_chronological_split_invalid_time_overlap(sample_df):
    """Test that overlapping time periods raise AssertionError."""
    # Deliberately create overlapping train/test splits
    sorted_df = sample_df.sort_values("end_date")
    
    # Train includes later dates
    train = sorted_df.iloc[10:90]
    # Test includes earlier dates
    test = sorted_df.iloc[:20]
    
    with pytest.raises(AssertionError, match="TIME LEAKAGE DETECTED"):
        assert_chronological_split(train, test, "end_date")


def test_assert_chronological_split_invalid_index_overlap(sample_df):
    """Test that overlapping indices raise AssertionError."""
    sorted_df = sample_df.sort_values("end_date").reset_index(drop=True)
    
    # Create overlapping index ranges (rows 70-80 in both)
    train = sorted_df.iloc[:80].copy()
    test = sorted_df.iloc[70:].copy()
    
    # Dates are valid (train max < test min) but indices overlap
    # This tests the index overlap check specifically
    # Need to modify test dates to be after train dates
    test_copy = test.copy()
    test_copy["end_date"] = pd.date_range("2020-01-01", periods=len(test_copy), freq="7D")
    
    with pytest.raises(AssertionError, match="INDEX LEAKAGE DETECTED"):
        assert_chronological_split(train, test_copy, "end_date")


def test_assert_chronological_split_missing_date_column():
    """Test that missing date column raises AssertionError."""
    train = pd.DataFrame({"value": [1, 2, 3]})
    test = pd.DataFrame({"value": [4, 5, 6]})
    
    with pytest.raises(AssertionError, match="Date column.*not found"):
        assert_chronological_split(train, test, "end_date")


# =============================================================================
# Tests for chronological_train_test_split
# =============================================================================

def test_chronological_train_test_split_basic(sample_df):
    """Test basic chronological split functionality."""
    train, test = chronological_train_test_split(sample_df, "end_date", test_ratio=0.2)
    
    # Check sizes are approximately correct
    assert len(train) > 0
    assert len(test) > 0
    assert len(train) + len(test) == len(sample_df)
    
    # Check chronological ordering
    train_max_date = train["end_date"].max()
    test_min_date = test["end_date"].min()
    assert train_max_date < test_min_date


def test_chronological_train_test_split_no_date_column():
    """Test that missing date column raises ValueError."""
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
    
    with pytest.raises(ValueError, match="Date column.*not found"):
        chronological_train_test_split(df, "end_date")


def test_chronological_train_test_split_too_few_rows():
    """Test that insufficient data raises ValueError."""
    df = pd.DataFrame({"end_date": [pd.Timestamp("2020-01-01")]})
    
    with pytest.raises(ValueError, match="Need at least 2 rows"):
        chronological_train_test_split(df, "end_date")


def test_chronological_train_test_split_preserves_data(sample_df):
    """Test that split preserves all data without loss."""
    train, test = chronological_train_test_split(sample_df, "end_date", test_ratio=0.3)
    
    # All rows should be present
    assert len(train) + len(test) == len(sample_df)
    
    # All sales values should be preserved
    all_sales = set(sample_df["sales"])
    split_sales = set(train["sales"]) | set(test["sales"])
    assert all_sales == split_sales


# =============================================================================
# Tests for rolling_origin_cv_splits
# =============================================================================

def test_rolling_origin_cv_splits_generates_folds(sample_df):
    """Test that rolling origin CV generates valid folds."""
    folds = list(rolling_origin_cv_splits(
        sample_df, "end_date",
        initial_train_period="200D",
        horizon="100D",
        step="50D"
    ))
    
    assert len(folds) > 0
    
    for train_idx, test_idx in folds:
        assert len(train_idx) > 0
        assert len(test_idx) > 0
        
        # Verify no overlap
        assert len(set(train_idx) & set(test_idx)) == 0


def test_rolling_origin_cv_splits_chronological(sample_df):
    """Test that rolling origin splits maintain chronological order."""
    sorted_df = sample_df.sort_values("end_date").reset_index(drop=True)
    
    folds = list(rolling_origin_cv_splits(
        sorted_df, "end_date",
        initial_train_period="100D",
        horizon="50D",
        step="50D"
    ))
    
    for train_idx, test_idx in folds:
        train_max = sorted_df.loc[train_idx, "end_date"].max()
        test_min = sorted_df.loc[test_idx, "end_date"].min()
        
        assert train_max < test_min, f"Train max {train_max} >= test min {test_min}"


# =============================================================================
# Tests for TimeSeriesCVSplitter
# =============================================================================

def test_time_series_cv_splitter_basic(sample_df):
    """Test basic functionality of TimeSeriesCVSplitter."""
    splitter = TimeSeriesCVSplitter(n_splits=5, date_column="end_date")
    
    folds = list(splitter.split(sample_df))
    
    assert len(folds) == 5
    
    for train_idx, test_idx in folds:
        assert len(train_idx) > 0
        assert len(test_idx) > 0


def test_time_series_cv_splitter_chronological(sample_df):
    """Test that TimeSeriesCVSplitter maintains chronological order."""
    splitter = TimeSeriesCVSplitter(n_splits=3, date_column="end_date")
    
    for train_idx, test_idx in splitter.split(sample_df):
        train_dates = sample_df.loc[train_idx, "end_date"]
        test_dates = sample_df.loc[test_idx, "end_date"]
        
        assert train_dates.max() < test_dates.min()


def test_time_series_cv_splitter_no_date_column(sample_df_no_dates):
    """Test fallback behavior when date column is missing."""
    splitter = TimeSeriesCVSplitter(n_splits=3, date_column="end_date")
    
    # Should still work using sequential split
    folds = list(splitter.split(sample_df_no_dates))
    
    assert len(folds) >= 1
    
    for train_idx, test_idx in folds:
        assert len(train_idx) > 0
        assert len(test_idx) > 0
        # Sequential: all train indices < all test indices
        assert train_idx.max() < test_idx.min()


def test_time_series_cv_splitter_get_n_splits():
    """Test get_n_splits method."""
    splitter = TimeSeriesCVSplitter(n_splits=7)
    assert splitter.get_n_splits() == 7


# =============================================================================
# Tests for assert_group_chronological_split
# =============================================================================

def test_assert_group_chronological_split_valid(sample_df):
    """Test that valid group-wise chronological split passes."""
    sorted_df = sample_df.sort_values("end_date").reset_index(drop=True)
    
    # Create mask where first 80% is train
    train_mask = pd.Series([True] * 80 + [False] * 20, index=sorted_df.index)
    
    # Should not raise
    assert_group_chronological_split(
        sorted_df, train_mask,
        date_column="end_date",
        group_column="canonical_title"
    )


def test_assert_group_chronological_split_invalid():
    """Test that group-wise time overlap raises AssertionError."""
    # Create a DataFrame where one group has leakage
    df = pd.DataFrame({
        "end_date": pd.to_datetime([
            "2018-01-01", "2018-06-01",  # Group A - train
            "2019-01-01", "2019-06-01",  # Group A - should be test but we'll mark as train
            "2020-01-01", "2020-06-01",  # Group B - test
        ]),
        "canonical_title": ["A", "A", "A", "A", "B", "B"],
        "value": [1, 2, 3, 4, 5, 6],
    })
    
    # Mark 2019 data as train but 2018 data as test (leakage in group A)
    train_mask = pd.Series([False, False, True, True, False, False], index=df.index)
    
    with pytest.raises(AssertionError, match="TIME LEAKAGE within groups"):
        assert_group_chronological_split(
            df, train_mask,
            date_column="end_date",
            group_column="canonical_title"
        )


def test_assert_group_chronological_split_no_group_column(sample_df):
    """Test fallback to overall chronological check when no group column."""
    sorted_df = sample_df.sort_values("end_date").reset_index(drop=True)
    train_mask = pd.Series([True] * 80 + [False] * 20, index=sorted_df.index)
    
    # Should fall back to overall chronological check
    assert_group_chronological_split(
        sorted_df, train_mask,
        date_column="end_date",
        group_column="nonexistent_column"
    )


# =============================================================================
# Integration tests
# =============================================================================

def test_no_2018_2020_predicting_2019():
    """
    Specific test for the original concern:
    2018 and 2020 data should NOT predict 2019 tickets.
    """
    # Create sample data spanning 2018-2020
    dates = [
        "2018-03-01", "2018-06-01", "2018-11-01",  # 2018 shows
        "2019-02-01", "2019-05-01", "2019-10-01",  # 2019 shows
        "2020-03-01", "2020-07-01",                # 2020 shows
    ]
    df = pd.DataFrame({
        "end_date": pd.to_datetime(dates),
        "title": [f"Show_{i}" for i in range(8)],
        "sales": [1000, 1200, 800, 1500, 1100, 900, 1300, 1400],
    })
    
    # Correct split: train on 2018, test on 2019+
    train_2018 = df[df["end_date"].dt.year == 2018]
    test_2019_plus = df[df["end_date"].dt.year >= 2019]
    
    # This should pass
    assert_chronological_split(train_2018, test_2019_plus, "end_date")
    
    # Wrong split: train on 2018+2020, test on 2019 (leakage!)
    train_leaky = df[df["end_date"].dt.year.isin([2018, 2020])]
    test_2019 = df[df["end_date"].dt.year == 2019]
    
    # This should FAIL because 2020 data is in training but 2019 is in test
    with pytest.raises(AssertionError, match="TIME LEAKAGE"):
        assert_chronological_split(train_leaky, test_2019, "end_date")


def test_train_test_split_prevents_2020_in_train_2019_in_test():
    """Test that chronological split cannot have 2020 train and 2019 test."""
    dates = pd.date_range("2018-01-01", periods=36, freq="ME")  # 3 years monthly
    df = pd.DataFrame({
        "end_date": dates,
        "value": range(36),
    })
    
    train, test = chronological_train_test_split(df, "end_date", test_ratio=0.33)
    
    # Verify 2020 cannot be in train while 2019 is in test
    train_years = train["end_date"].dt.year.unique()
    test_years = test["end_date"].dt.year.unique()
    
    # The latest train year should be <= the earliest test year
    assert train["end_date"].max() < test["end_date"].min()


def test_cv_splitter_with_sklearn_interface(sample_df):
    """Test that TimeSeriesCVSplitter works with sklearn-style iteration."""
    from sklearn.linear_model import Ridge
    
    splitter = TimeSeriesCVSplitter(n_splits=3, date_column="end_date")
    
    X = sample_df[["wiki", "sales"]].copy()
    y = sample_df["sales"]
    
    # Verify it works in a cross-validation loop
    scores = []
    for train_idx, test_idx in splitter.split(sample_df):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = Ridge()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
    
    assert len(scores) == 3
