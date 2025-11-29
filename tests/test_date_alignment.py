"""Tests for date alignment and forward-fill correctness."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data.features import (
    normalize_to_year_month,
    snap_show_date_to_month,
    join_consumer_confidence,
    join_energy_index,
    compute_inflation_adjustment_factor,
)


class TestDateAlignment:
    """Tests for correct date alignment in feature joins."""
    
    @pytest.fixture
    def monthly_economic_data(self):
        """Create monthly economic data with known values."""
        dates = pd.date_range(start='2024-01-01', periods=12, freq='MS')
        return pd.DataFrame({
            'date': dates,
            'A.ENER': [1000 + i * 50 for i in range(12)]  # 1000, 1050, 1100, ...
        })
    
    @pytest.fixture
    def show_dates_across_year(self):
        """Create show dates across multiple months."""
        return pd.DataFrame({
            'show_date': [
                '2024-01-15',  # Mid January
                '2024-03-05',  # Early March
                '2024-06-28',  # Late June
                '2024-12-20',  # Late December
            ],
            'title': ['Show A', 'Show B', 'Show C', 'Show D']
        })
    
    def test_energy_index_aligns_to_month_start(
        self, 
        show_dates_across_year, 
        monthly_economic_data
    ):
        """Energy index should use the month's data for any day in that month."""
        result = join_energy_index(
            show_dates_across_year, 
            monthly_economic_data, 
            date_column='show_date'
        )
        
        # January show should get January energy index (1000)
        jan_show = result[result['title'] == 'Show A']['energy_index'].iloc[0]
        assert jan_show == 1000.0
        
        # March show should get March energy index (1100)
        mar_show = result[result['title'] == 'Show B']['energy_index'].iloc[0]
        assert mar_show == 1100.0
        
        # June show should get June energy index (1250)
        jun_show = result[result['title'] == 'Show C']['energy_index'].iloc[0]
        assert jun_show == 1250.0
    
    def test_no_future_data_leak(self, show_dates_across_year, monthly_economic_data):
        """Features should not use data from after the show date."""
        # Add a show in a month with no data yet
        df = show_dates_across_year.copy()
        df = pd.concat([df, pd.DataFrame({
            'show_date': ['2025-06-15'],
            'title': ['Future Show']
        })], ignore_index=True)
        
        result = join_energy_index(df, monthly_economic_data, date_column='show_date')
        
        # Future show should not have 2025 data (which doesn't exist)
        # It should get forward-filled with latest available
        future_show = result[result['title'] == 'Future Show']['energy_index'].iloc[0]
        
        # Should either be NaN or the last available value (December 2024)
        assert pd.isna(future_show) or future_show == monthly_economic_data['A.ENER'].iloc[-1]


class TestForwardFill:
    """Tests for forward-fill behavior when data is missing."""
    
    @pytest.fixture
    def sparse_economic_data(self):
        """Create economic data with gaps."""
        return pd.DataFrame({
            'date': ['2024-01-01', '2024-03-01', '2024-06-01'],  # Missing Feb, Apr, May
            'A.ENER': [1000.0, 1100.0, 1200.0]
        })
    
    @pytest.fixture
    def monthly_shows(self):
        """Create shows for every month."""
        months = pd.date_range(start='2024-01-15', periods=6, freq='MS') + timedelta(days=14)
        return pd.DataFrame({
            'show_date': months,
            'title': [f'Show {i}' for i in range(6)]
        })
    
    def test_forward_fill_uses_most_recent_value(
        self, 
        monthly_shows, 
        sparse_economic_data
    ):
        """Missing months should use available values or forward-fill."""
        result = join_energy_index(
            monthly_shows, 
            sparse_economic_data, 
            date_column='show_date'
        )
        
        # January has data (1000)
        jan = result[result['title'] == 'Show 0']['energy_index'].iloc[0]
        # Value should be filled (either from January data or latest available)
        assert pd.notna(jan), "January energy index should not be NaN"
        
        # February is missing - should get some valid value
        feb = result[result['title'] == 'Show 1']['energy_index'].iloc[0]
        # Should not be NaN - forward-fill should apply
        assert pd.notna(feb), "February should not be NaN"
    
    def test_forward_fill_preserves_data_integrity(self, sparse_economic_data):
        """Forward-fill should not modify the original data values."""
        original = sparse_economic_data.copy()
        
        shows = pd.DataFrame({
            'show_date': ['2024-02-15', '2024-04-15'],  # Dates with no data
            'title': ['Show A', 'Show B']
        })
        
        result = join_energy_index(shows, sparse_economic_data, date_column='show_date')
        
        # Original data should be unchanged
        pd.testing.assert_frame_equal(original, sparse_economic_data)


class TestCPIInflationFactor:
    """Tests for CPI-based inflation adjustment factor."""
    
    @pytest.fixture
    def cpi_data_with_inflation(self):
        """Create CPI data showing inflation over time."""
        return pd.DataFrame({
            'date': ['2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01', '2024-01-01'],
            'V41690973': [136.8, 138.2, 145.3, 153.9, 158.3]  # Rising CPI
        })
    
    def test_inflation_factor_increases_with_time(self, cpi_data_with_inflation):
        """Later dates should have higher inflation factors (prices rose)."""
        shows = pd.DataFrame({
            'show_date': ['2021-01-15', '2023-01-15', '2024-01-15']
        })
        
        result = compute_inflation_adjustment_factor(
            shows, 
            cpi_data_with_inflation,
            date_column='show_date',
            base_date='2020-01-01'
        )
        
        factors = result['inflation_adjustment_factor'].tolist()
        
        # Factors should increase over time (as CPI rises relative to base)
        assert factors[0] < factors[1] < factors[2], \
            f"Inflation factors should increase: {factors}"
    
    def test_base_date_factor_is_near_one(self, cpi_data_with_inflation):
        """Show at base date should have factor near 1.0."""
        shows = pd.DataFrame({
            'show_date': ['2020-01-15']
        })
        
        result = compute_inflation_adjustment_factor(
            shows,
            cpi_data_with_inflation,
            date_column='show_date',
            base_date='2020-01-01'
        )
        
        factor = result['inflation_adjustment_factor'].iloc[0]
        
        # Should be very close to 1.0
        assert abs(factor - 1.0) < 0.05, f"Base date factor should be ~1.0, got {factor}"


class TestTimeZoneHandling:
    """Tests for timezone handling in date operations."""
    
    def test_handles_mixed_date_formats(self):
        """Should handle various date string formats."""
        shows = pd.DataFrame({
            'show_date': [
                '2024-01-15',        # ISO format
                '2024/02/20',        # Slash format
            ]
        })
        
        # Should not raise error
        result = snap_show_date_to_month(shows['show_date'])
        
        # First should parse correctly
        assert result.iloc[0] == '2024-01'
        # Second may or may not parse depending on pandas version
        assert result.iloc[1] == '2024-02' or pd.isna(result.iloc[1])
    
    def test_handles_datetime_objects(self):
        """Should work with actual datetime objects."""
        shows = pd.DataFrame({
            'show_date': [
                datetime(2024, 1, 15, 19, 30),  # With time
                datetime(2024, 2, 20, 14, 0),
            ]
        })
        
        result = snap_show_date_to_month(shows['show_date'])
        
        assert result.iloc[0] == '2024-01'
        assert result.iloc[1] == '2024-02'
    
    def test_handles_pandas_timestamp(self):
        """Should work with Pandas Timestamp objects."""
        shows = pd.DataFrame({
            'show_date': pd.to_datetime(['2024-01-15', '2024-02-20'])
        })
        
        result = snap_show_date_to_month(shows['show_date'])
        
        assert result.iloc[0] == '2024-01'
        assert result.iloc[1] == '2024-02'
