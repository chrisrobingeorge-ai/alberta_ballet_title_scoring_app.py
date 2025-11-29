"""Tests for feature engineering schema validation."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from data.features import (
    normalize_to_year_month,
    snap_show_date_to_month,
    join_consumer_confidence,
    join_energy_index,
    compute_inflation_adjustment_factor,
    add_city_segmentation_features,
    build_feature_store,
)


class TestDateNormalization:
    """Tests for date normalization functions."""
    
    def test_normalize_to_year_month_with_datetime(self):
        """Should convert datetime series to Period[M]."""
        dates = pd.Series([
            datetime(2024, 1, 15),
            datetime(2024, 2, 28),
            datetime(2024, 12, 1)
        ])
        result = normalize_to_year_month(dates)
        
        assert len(result) == 3
        # Check periods are correct
        assert str(result.iloc[0]) == '2024-01'
        assert str(result.iloc[1]) == '2024-02'
        assert str(result.iloc[2]) == '2024-12'
    
    def test_normalize_to_year_month_with_string_dates(self):
        """Should handle string date inputs."""
        dates = pd.Series(['2024-01-15', '2024-02-28', '2024-12-01'])
        result = normalize_to_year_month(dates)
        
        assert len(result) == 3
        assert str(result.iloc[0]) == '2024-01'
    
    def test_normalize_to_year_month_empty_series(self):
        """Should return empty series for empty input."""
        result = normalize_to_year_month(pd.Series(dtype='datetime64[ns]'))
        assert result.empty
    
    def test_snap_show_date_to_month(self):
        """Should convert dates to YYYY-MM string format."""
        dates = pd.Series([
            datetime(2024, 1, 15),
            datetime(2024, 6, 30),
            datetime(2025, 12, 25)
        ])
        result = snap_show_date_to_month(dates)
        
        assert result.iloc[0] == '2024-01'
        assert result.iloc[1] == '2024-06'
        assert result.iloc[2] == '2025-12'


class TestFeatureSchema:
    """Tests for feature engineering output schema."""
    
    @pytest.fixture
    def sample_show_data(self):
        """Create sample show/production data."""
        return pd.DataFrame({
            'show_title': ['Nutcracker', 'Swan Lake', 'Giselle'],
            'show_date': ['2024-12-15', '2025-02-10', '2025-05-01'],
            'city': ['Calgary', 'Edmonton', 'Calgary'],
            'venue_capacity': [2500, 2800, 2500],
            'ticket_price': [75.0, 85.0, 70.0]
        })
    
    @pytest.fixture
    def sample_nanos_data(self):
        """Create sample Nanos consumer confidence data."""
        return pd.DataFrame({
            'category': ['BNCCI', 'BNCCI', 'Demographics', 'Demographics'],
            'subcategory': ['Headline Index', 'Headline Index', 'Region', 'Region'],
            'metric': ['This week', 'Last week', 'Prairies', 'Ontario'],
            'year_or_period': ['2024-12-01', '2024-11-24', '2024-12-01', '2024-12-01'],
            'value': [52.5, 51.8, 54.2, 48.7],
            'unit': ['index', 'index', 'index', 'index']
        })
    
    @pytest.fixture
    def sample_commodity_data(self):
        """Create sample commodity price data."""
        return pd.DataFrame({
            'date': ['2024-01-01', '2024-02-01', '2024-03-01'],
            'A.BCPI': [600.0, 610.0, 605.0],
            'A.ENER': [1300.0, 1350.0, 1280.0],
            'A.MTLS': [750.0, 760.0, 755.0]
        })
    
    @pytest.fixture
    def sample_cpi_data(self):
        """Create sample CPI data."""
        return pd.DataFrame({
            'date': ['2024-01-01', '2024-02-01', '2024-03-01'],
            'V41690973': [161.3, 163.0, 163.5],
            'STATIC_TOTALCPICHANGE': [2.9, 2.6, 2.3]
        })
    
    def test_join_consumer_confidence_adds_expected_columns(self, sample_show_data, sample_nanos_data):
        """Consumer confidence join should add expected feature columns."""
        result = join_consumer_confidence(sample_show_data, sample_nanos_data)
        
        assert 'consumer_confidence_headline' in result.columns
        assert 'consumer_confidence_prairies' in result.columns
        assert len(result) == len(sample_show_data)
    
    def test_join_consumer_confidence_handles_empty_nanos(self, sample_show_data):
        """Should handle empty Nanos data gracefully."""
        result = join_consumer_confidence(sample_show_data, pd.DataFrame())
        
        # Should return original data unchanged
        assert len(result) == len(sample_show_data)
        assert set(sample_show_data.columns).issubset(set(result.columns))
    
    def test_join_energy_index_adds_expected_columns(self, sample_show_data, sample_commodity_data):
        """Energy index join should add energy_index column."""
        result = join_energy_index(sample_show_data, sample_commodity_data)
        
        assert 'energy_index' in result.columns
        assert len(result) == len(sample_show_data)
    
    def test_compute_inflation_factor_adds_expected_columns(self, sample_show_data, sample_cpi_data):
        """Inflation factor computation should add adjustment factor."""
        result = compute_inflation_adjustment_factor(sample_show_data, sample_cpi_data)
        
        assert 'inflation_adjustment_factor' in result.columns
        assert len(result) == len(sample_show_data)
    
    def test_add_city_segmentation_adds_expected_columns(self, sample_show_data):
        """City segmentation should add city binary features."""
        result = add_city_segmentation_features(sample_show_data)
        
        assert 'city_calgary' in result.columns
        assert 'city_edmonton' in result.columns
        assert 'city_population' in result.columns
        assert 'city_median_household_income' in result.columns
    
    def test_city_binary_features_are_correct(self, sample_show_data):
        """City binary features should correctly encode city values."""
        result = add_city_segmentation_features(sample_show_data)
        
        # First row is Calgary
        assert result.loc[0, 'city_calgary'] == 1
        assert result.loc[0, 'city_edmonton'] == 0
        
        # Second row is Edmonton
        assert result.loc[1, 'city_calgary'] == 0
        assert result.loc[1, 'city_edmonton'] == 1
    
    def test_build_feature_store_returns_all_features(
        self, 
        sample_show_data, 
        sample_nanos_data, 
        sample_commodity_data, 
        sample_cpi_data
    ):
        """Feature store should include all feature types."""
        result = build_feature_store(
            history_df=sample_show_data,
            nanos_confidence_df=sample_nanos_data,
            commodity_df=sample_commodity_data,
            cpi_df=sample_cpi_data,
            date_column='show_date',
            city_column='city'
        )
        
        # Check all expected features are present
        expected_features = [
            'consumer_confidence_headline',
            'consumer_confidence_prairies',
            'energy_index',
            'inflation_adjustment_factor',
            'city_calgary',
            'city_edmonton'
        ]
        
        for feature in expected_features:
            assert feature in result.columns, f"Missing feature: {feature}"
    
    def test_build_feature_store_handles_empty_history(self):
        """Should handle empty history data gracefully."""
        result = build_feature_store(pd.DataFrame())
        assert result.empty


class TestFeatureDataTypes:
    """Tests for correct data types in features."""
    
    def test_city_binary_features_are_integers(self):
        """City binary features should be integers (0/1)."""
        df = pd.DataFrame({
            'city': ['Calgary', 'Edmonton', 'Calgary']
        })
        result = add_city_segmentation_features(df)
        
        assert result['city_calgary'].dtype in [np.int64, np.int32, int]
        assert result['city_edmonton'].dtype in [np.int64, np.int32, int]
        assert set(result['city_calgary'].unique()).issubset({0, 1})
        assert set(result['city_edmonton'].unique()).issubset({0, 1})
    
    def test_confidence_values_are_numeric(self):
        """Consumer confidence values should be numeric."""
        show_df = pd.DataFrame({
            'show_date': ['2024-12-15'],
            'city': ['Calgary']
        })
        nanos_df = pd.DataFrame({
            'category': ['BNCCI'],
            'subcategory': ['Headline Index'],
            'metric': ['This week'],
            'year_or_period': ['2024-12-01'],
            'value': [52.5],
            'unit': ['index']
        })
        
        result = join_consumer_confidence(show_df, nanos_df)
        
        assert pd.api.types.is_numeric_dtype(result['consumer_confidence_headline'])
    
    def test_inflation_factor_is_positive(self):
        """Inflation adjustment factor should be positive."""
        show_df = pd.DataFrame({
            'show_date': ['2024-01-15', '2024-02-15']
        })
        cpi_df = pd.DataFrame({
            'date': ['2024-01-01', '2024-02-01'],
            'V41690973': [161.3, 163.0]
        })
        
        result = compute_inflation_adjustment_factor(show_df, cpi_df)
        
        # All non-null values should be positive
        valid_values = result['inflation_adjustment_factor'].dropna()
        if len(valid_values) > 0:
            assert (valid_values > 0).all()
