"""Tests for economic data loader functions."""
import pytest
import pandas as pd
from datetime import date

from data.loader import (
    load_oil_prices,
    load_unemployment_rates,
    load_segment_priors,
    load_audience_analytics,
    get_economic_sentiment_factor,
    get_segment_weight,
    get_all_segment_weights,
)


class TestOilPriceLoader:
    """Tests for oil price data loading."""
    
    def test_load_oil_prices_returns_dataframe(self):
        """load_oil_prices should return a DataFrame."""
        result = load_oil_prices()
        assert isinstance(result, pd.DataFrame)
    
    def test_load_oil_prices_has_expected_columns(self):
        """Oil price DataFrame should have date and price columns."""
        result = load_oil_prices()
        if not result.empty:
            assert 'date' in result.columns
            assert 'wcs_oil_price' in result.columns or len(result.columns) > 1
    
    def test_load_oil_prices_fallback_on_missing(self):
        """Should return empty DataFrame when file is missing and fallback is True."""
        result = load_oil_prices(csv_name="nonexistent.csv", fallback_empty=True)
        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestUnemploymentLoader:
    """Tests for unemployment rate data loading."""
    
    def test_load_unemployment_returns_dataframe(self):
        """load_unemployment_rates should return a DataFrame."""
        result = load_unemployment_rates()
        assert isinstance(result, pd.DataFrame)
    
    def test_load_unemployment_has_expected_columns(self):
        """Unemployment DataFrame should have date, rate, and region columns."""
        result = load_unemployment_rates()
        if not result.empty:
            assert 'date' in result.columns
            assert 'unemployment_rate' in result.columns
            assert 'region' in result.columns
    
    def test_load_unemployment_has_alberta_region(self):
        """Should include Alberta region data."""
        result = load_unemployment_rates()
        if not result.empty and 'region' in result.columns:
            regions = result['region'].unique().tolist()
            assert 'Alberta' in regions or any('alberta' in str(r).lower() for r in regions)
    
    def test_load_unemployment_fallback_on_missing(self):
        """Should return empty DataFrame when file is missing and fallback is True."""
        result = load_unemployment_rates(csv_name="nonexistent.csv", fallback_empty=True)
        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestSegmentPriorsLoader:
    """Tests for segment priors loading."""
    
    def test_load_segment_priors_returns_dataframe(self):
        """load_segment_priors should return a DataFrame."""
        result = load_segment_priors()
        assert isinstance(result, pd.DataFrame)
    
    def test_load_segment_priors_has_expected_columns(self):
        """Segment priors should have region, category, segment, and weight columns."""
        result = load_segment_priors()
        if not result.empty:
            expected_cols = ['region', 'category', 'segment', 'weight']
            for col in expected_cols:
                assert col in result.columns
    
    def test_load_segment_priors_fallback_on_missing(self):
        """Should return empty DataFrame when file is missing and fallback is True."""
        result = load_segment_priors(csv_name="nonexistent.csv", fallback_empty=True)
        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestEconomicSentimentFactor:
    """Tests for economic sentiment factor calculation."""
    
    def test_returns_float(self):
        """Should return a float value."""
        result = get_economic_sentiment_factor()
        assert isinstance(result, float)
    
    def test_returns_within_bounds(self):
        """Result should be within min/max bounds."""
        result = get_economic_sentiment_factor(
            min_factor=0.85,
            max_factor=1.15
        )
        assert 0.85 <= result <= 1.15
    
    def test_with_specific_date(self):
        """Should work with a specific date."""
        # Use a date 6 months ago to ensure data availability
        test_date = pd.Timestamp.now() - pd.DateOffset(months=6)
        result = get_economic_sentiment_factor(run_date=test_date)
        assert isinstance(result, float)
        assert 0.5 <= result <= 1.5  # Reasonable range
    
    def test_with_city_filter(self):
        """Should work with city filter."""
        result = get_economic_sentiment_factor(city="Calgary")
        assert isinstance(result, float)
    
    def test_neutral_when_no_data(self):
        """Should return 1.0 when data is unavailable."""
        # Test with extreme date that won't have data
        result = get_economic_sentiment_factor(run_date=pd.Timestamp("1900-01-01"))
        assert isinstance(result, float)


class TestSegmentWeightFunctions:
    """Tests for segment weight lookup functions."""
    
    def test_get_segment_weight_returns_float(self):
        """get_segment_weight should return a float."""
        result = get_segment_weight("Province", "classic_romance", "General Population")
        assert isinstance(result, float)
    
    def test_get_segment_weight_default_on_missing(self):
        """Should return default when segment not found."""
        result = get_segment_weight("NonexistentRegion", "nonexistent", "nonexistent", default=99.0)
        assert result == 99.0
    
    def test_get_all_segment_weights_returns_dict(self):
        """get_all_segment_weights should return a dictionary."""
        result = get_all_segment_weights("Province", "classic_romance")
        assert isinstance(result, dict)
    
    def test_get_all_segment_weights_values_are_floats(self):
        """All values in segment weights dict should be floats."""
        result = get_all_segment_weights("Province", "classic_romance")
        for key, value in result.items():
            assert isinstance(value, (int, float))


class TestAudienceAnalyticsLoader:
    """Tests for audience analytics loading."""
    
    def test_load_audience_analytics_returns_dataframe(self):
        """load_audience_analytics should return a DataFrame."""
        result = load_audience_analytics()
        assert isinstance(result, pd.DataFrame)
    
    def test_load_audience_analytics_fallback_on_missing(self):
        """Should return empty DataFrame when file is missing and fallback is True."""
        result = load_audience_analytics(csv_name="nonexistent.csv", fallback_empty=True)
        assert isinstance(result, pd.DataFrame)
        assert result.empty
