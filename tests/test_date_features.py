"""Tests for date-based feature derivation functions.

These tests verify the new date-based features added for forecasting:
- Year, month, day of week extraction
- Season indicators (spring, summer, autumn, winter)
- Holiday season flag
- Run duration computation
- Days to opening computation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from data.features import (
    derive_date_features,
    compute_days_to_opening,
    _extract_temporal_features,
    _compute_season,
)


class TestDeriveDateFeatures:
    """Tests for the derive_date_features function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample production data with dates."""
        return pd.DataFrame({
            'title': ['Nutcracker', 'Swan Lake', 'Giselle', 'Romeo & Juliet'],
            'start_date': ['2024-12-15', '2025-02-10', '2025-05-01', '2025-09-20'],
            'end_date': ['2024-12-23', '2025-02-18', '2025-05-08', '2025-09-28']
        })
    
    def test_adds_opening_year(self, sample_data):
        """Should add opening_year column."""
        result = derive_date_features(sample_data)
        
        assert 'opening_year' in result.columns
        assert result['opening_year'].iloc[0] == 2024
        assert result['opening_year'].iloc[1] == 2025
    
    def test_adds_opening_month(self, sample_data):
        """Should add opening_month column."""
        result = derive_date_features(sample_data)
        
        assert 'opening_month' in result.columns
        assert result['opening_month'].iloc[0] == 12  # December
        assert result['opening_month'].iloc[1] == 2   # February
        assert result['opening_month'].iloc[2] == 5   # May
        assert result['opening_month'].iloc[3] == 9   # September
    
    def test_adds_opening_day_of_week(self, sample_data):
        """Should add opening_day_of_week column (0=Monday, 6=Sunday)."""
        result = derive_date_features(sample_data)
        
        assert 'opening_day_of_week' in result.columns
        # 2024-12-15 is a Sunday (6)
        assert result['opening_day_of_week'].iloc[0] == 6
    
    def test_adds_opening_quarter(self, sample_data):
        """Should add opening_quarter column."""
        result = derive_date_features(sample_data)
        
        assert 'opening_quarter' in result.columns
        assert result['opening_quarter'].iloc[0] == 4  # December = Q4
        assert result['opening_quarter'].iloc[1] == 1  # February = Q1
        assert result['opening_quarter'].iloc[2] == 2  # May = Q2
        assert result['opening_quarter'].iloc[3] == 3  # September = Q3
    
    def test_adds_opening_season(self, sample_data):
        """Should add opening_season column with correct seasons."""
        result = derive_date_features(sample_data)
        
        assert 'opening_season' in result.columns
        assert result['opening_season'].iloc[0] == 'winter'  # December
        assert result['opening_season'].iloc[1] == 'winter'  # February
        assert result['opening_season'].iloc[2] == 'spring'  # May
        assert result['opening_season'].iloc[3] == 'autumn'  # September
    
    def test_adds_season_binary_flags(self, sample_data):
        """Should add binary season flags."""
        result = derive_date_features(sample_data)
        
        # Check all season flag columns exist
        for season in ['winter', 'spring', 'summer', 'autumn']:
            assert f'opening_is_{season}' in result.columns
        
        # December should be winter
        assert result['opening_is_winter'].iloc[0] == 1
        assert result['opening_is_spring'].iloc[0] == 0
        assert result['opening_is_summer'].iloc[0] == 0
        assert result['opening_is_autumn'].iloc[0] == 0
        
        # September should be autumn
        assert result['opening_is_autumn'].iloc[3] == 1
    
    def test_adds_holiday_season_flag(self, sample_data):
        """Should add holiday season flag (Nov, Dec, Jan)."""
        result = derive_date_features(sample_data)
        
        assert 'opening_is_holiday_season' in result.columns
        assert result['opening_is_holiday_season'].iloc[0] == 1  # December
        assert result['opening_is_holiday_season'].iloc[2] == 0  # May
    
    def test_adds_weekend_flag(self, sample_data):
        """Should add weekend opening flag."""
        result = derive_date_features(sample_data)
        
        assert 'opening_is_weekend' in result.columns
        # 2024-12-15 is a Sunday
        assert result['opening_is_weekend'].iloc[0] == 1
    
    def test_adds_run_duration_days(self, sample_data):
        """Should compute run duration in days."""
        result = derive_date_features(sample_data)
        
        assert 'run_duration_days' in result.columns
        # 2024-12-15 to 2024-12-23 = 8 days
        assert result['run_duration_days'].iloc[0] == 8
    
    def test_handles_missing_dates(self):
        """Should handle rows with missing dates."""
        df = pd.DataFrame({
            'title': ['Show A', 'Show B'],
            'start_date': ['2024-12-15', None],
            'end_date': ['2024-12-23', '2025-01-05']
        })
        
        result = derive_date_features(df)
        
        # First row should have valid features
        assert pd.notna(result['opening_year'].iloc[0])
        
        # Second row should have NaN for features derived from start_date
        assert pd.isna(result['opening_year'].iloc[1])
    
    def test_handles_string_dates(self):
        """Should parse string dates correctly."""
        df = pd.DataFrame({
            'title': ['Show'],
            'start_date': ['2024-12-15'],
            'end_date': ['2024-12-23']
        })
        
        result = derive_date_features(df)
        assert result['opening_year'].iloc[0] == 2024
    
    def test_handles_datetime_dates(self):
        """Should handle datetime objects directly."""
        df = pd.DataFrame({
            'title': ['Show'],
            'start_date': [datetime(2024, 12, 15)],
            'end_date': [datetime(2024, 12, 23)]
        })
        
        result = derive_date_features(df)
        assert result['opening_year'].iloc[0] == 2024
    
    def test_preserves_original_columns(self, sample_data):
        """Should preserve all original columns."""
        result = derive_date_features(sample_data)
        
        for col in sample_data.columns:
            assert col in result.columns
    
    def test_run_duration_clips_to_zero(self):
        """Run duration should be clipped to non-negative."""
        df = pd.DataFrame({
            'title': ['Bad Data'],
            'start_date': ['2024-12-23'],  # End before start
            'end_date': ['2024-12-15']
        })
        
        result = derive_date_features(df)
        assert result['run_duration_days'].iloc[0] >= 0


class TestComputeSeason:
    """Tests for the _compute_season helper function."""
    
    def test_winter_months(self):
        """December, January, February should be winter."""
        dates = pd.Series(pd.to_datetime(['2024-12-15', '2025-01-15', '2025-02-15']))
        result = _compute_season(dates)
        
        assert all(s == 'winter' for s in result)
    
    def test_spring_months(self):
        """March, April, May should be spring."""
        dates = pd.Series(pd.to_datetime(['2024-03-15', '2024-04-15', '2024-05-15']))
        result = _compute_season(dates)
        
        assert all(s == 'spring' for s in result)
    
    def test_summer_months(self):
        """June, July, August should be summer."""
        dates = pd.Series(pd.to_datetime(['2024-06-15', '2024-07-15', '2024-08-15']))
        result = _compute_season(dates)
        
        assert all(s == 'summer' for s in result)
    
    def test_autumn_months(self):
        """September, October, November should be autumn."""
        dates = pd.Series(pd.to_datetime(['2024-09-15', '2024-10-15', '2024-11-15']))
        result = _compute_season(dates)
        
        assert all(s == 'autumn' for s in result)
    
    def test_handles_nat(self):
        """Should handle NaT values."""
        dates = pd.Series([pd.NaT, datetime(2024, 6, 15)])
        result = _compute_season(dates)
        
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == 'summer'


class TestComputeDaysToOpening:
    """Tests for the compute_days_to_opening function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with start dates."""
        return pd.DataFrame({
            'title': ['Show A', 'Show B', 'Show C'],
            'start_date': ['2024-12-15', '2025-02-10', '2025-05-01']
        })
    
    def test_computes_days_to_opening(self, sample_data):
        """Should compute days until opening from reference date."""
        ref_date = pd.Timestamp('2024-11-01')
        result = compute_days_to_opening(sample_data, reference_date=ref_date)
        
        assert 'days_to_opening' in result.columns
        # 2024-12-15 - 2024-11-01 = 44 days
        assert result['days_to_opening'].iloc[0] == 44
    
    def test_handles_past_dates(self, sample_data):
        """Should return negative values for past dates."""
        ref_date = pd.Timestamp('2025-03-01')
        result = compute_days_to_opening(sample_data, reference_date=ref_date)
        
        # 2024-12-15 is before 2025-03-01, so days should be negative
        assert result['days_to_opening'].iloc[0] < 0
    
    def test_handles_missing_start_date(self):
        """Should handle missing start_date column."""
        df = pd.DataFrame({'title': ['Show']})
        result = compute_days_to_opening(df)
        
        assert 'days_to_opening' in result.columns
        assert pd.isna(result['days_to_opening'].iloc[0])
    
    def test_uses_current_date_as_default(self, sample_data):
        """Should use current date as reference if not provided."""
        result = compute_days_to_opening(sample_data)
        
        assert 'days_to_opening' in result.columns
        # Values should be computed (may be negative for past dates)


class TestExtractTemporalFeatures:
    """Tests for the _extract_temporal_features helper."""
    
    def test_adds_features_with_prefix(self):
        """Should add features with specified prefix."""
        df = pd.DataFrame({
            'show_date': ['2024-12-15']
        })
        
        result = _extract_temporal_features(df, 'show_date', prefix='opening')
        
        assert 'opening_year' in result.columns
        assert 'opening_month' in result.columns
        assert 'opening_day_of_week' in result.columns
    
    def test_adds_features_without_prefix(self):
        """Should add features without prefix when prefix is empty."""
        df = pd.DataFrame({
            'show_date': ['2024-12-15']
        })
        
        result = _extract_temporal_features(df, 'show_date', prefix='')
        
        assert 'year' in result.columns
        assert 'month' in result.columns
    
    def test_handles_missing_column(self):
        """Should return unchanged dataframe if date column missing."""
        df = pd.DataFrame({
            'title': ['Show']
        })
        
        result = _extract_temporal_features(df, 'nonexistent_date', prefix='opening')
        
        # Should return original columns only
        assert list(result.columns) == ['title']


class TestDateFeaturesIntegration:
    """Integration tests for date feature engineering."""
    
    def test_all_expected_columns_created(self):
        """Verify all expected date feature columns are created."""
        df = pd.DataFrame({
            'title': ['Nutcracker'],
            'start_date': ['2024-12-15'],
            'end_date': ['2024-12-23']
        })
        
        result = derive_date_features(df)
        
        expected_columns = [
            'opening_year', 'opening_month', 'opening_day_of_week',
            'opening_week_of_year', 'opening_quarter', 'opening_season',
            'opening_is_winter', 'opening_is_spring', 'opening_is_summer',
            'opening_is_autumn', 'opening_is_holiday_season', 'opening_is_weekend',
            'run_duration_days'
        ]
        
        for col in expected_columns:
            assert col in result.columns, f"Missing expected column: {col}"
    
    def test_features_are_correct_types(self):
        """Verify feature columns have correct data types."""
        df = pd.DataFrame({
            'title': ['Nutcracker'],
            'start_date': ['2024-12-15'],
            'end_date': ['2024-12-23']
        })
        
        result = derive_date_features(df)
        
        # Numeric features
        assert pd.api.types.is_numeric_dtype(result['opening_year'])
        assert pd.api.types.is_numeric_dtype(result['opening_month'])
        assert pd.api.types.is_numeric_dtype(result['run_duration_days'])
        
        # Binary features should be integers (0/1)
        binary_cols = ['opening_is_winter', 'opening_is_spring', 'opening_is_summer',
                       'opening_is_autumn', 'opening_is_holiday_season', 'opening_is_weekend']
        for col in binary_cols:
            assert result[col].iloc[0] in [0, 1], f"{col} should be 0 or 1"
    
    def test_season_flags_are_mutually_exclusive(self):
        """Only one season flag should be true at a time."""
        df = pd.DataFrame({
            'title': ['Show A', 'Show B', 'Show C', 'Show D'],
            'start_date': ['2024-01-15', '2024-04-15', '2024-07-15', '2024-10-15'],
            'end_date': ['2024-01-22', '2024-04-22', '2024-07-22', '2024-10-22']
        })
        
        result = derive_date_features(df)
        
        season_cols = ['opening_is_winter', 'opening_is_spring', 
                       'opening_is_summer', 'opening_is_autumn']
        
        for _, row in result.iterrows():
            # Sum of season flags should be exactly 1
            season_sum = sum(row[col] for col in season_cols)
            assert season_sum == 1, f"Expected exactly one season flag, got {season_sum}"
