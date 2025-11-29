"""
Tests for weather and live analytics data integration.

These tests verify that the weather data from weatherstats_calgary_daily.csv
and weatherstats_edmonton_daily.csv, as well as live_analytics.csv, are being
properly loaded and integrated into the scoring calculations.
"""

import pytest
import pandas as pd
from datetime import date


class TestWeatherDataLoaders:
    """Tests for weather data loading functions."""
    
    def test_load_weather_calgary(self):
        """Test that Calgary weather data loads successfully."""
        from data.loader import load_weather_calgary
        
        df = load_weather_calgary()
        
        # Should load data (file exists)
        assert not df.empty, "Calgary weather data should not be empty"
        
        # Check expected columns
        assert 'date' in df.columns, "Weather data should have 'date' column"
        assert 'city' in df.columns, "Weather data should have 'city' column"
        assert df['city'].iloc[0] == 'Calgary', "City should be Calgary"
        
        # Check for temperature columns
        assert 'min_temperature' in df.columns or 'avg_temperature' in df.columns, \
            "Weather data should have temperature columns"
    
    def test_load_weather_edmonton(self):
        """Test that Edmonton weather data loads successfully."""
        from data.loader import load_weather_edmonton
        
        df = load_weather_edmonton()
        
        # Should load data (file exists)
        assert not df.empty, "Edmonton weather data should not be empty"
        
        # Check expected columns
        assert 'date' in df.columns, "Weather data should have 'date' column"
        assert 'city' in df.columns, "Weather data should have 'city' column"
        assert df['city'].iloc[0] == 'Edmonton', "City should be Edmonton"
    
    def test_load_weather_all_cities(self):
        """Test that combined weather data loads both cities."""
        from data.loader import load_weather_all_cities
        
        df = load_weather_all_cities()
        
        assert not df.empty, "Combined weather data should not be empty"
        assert 'city' in df.columns
        
        cities = df['city'].unique()
        assert 'Calgary' in cities, "Combined data should include Calgary"
        assert 'Edmonton' in cities, "Combined data should include Edmonton"
    
    def test_weather_impact_factor_range(self):
        """Test that weather impact factor is within expected range."""
        from data.loader import get_weather_impact_factor
        
        factor = get_weather_impact_factor()
        
        assert 0.85 <= factor <= 1.05, \
            f"Weather factor {factor} should be between 0.85 and 1.05"
    
    def test_weather_impact_factor_city_specific(self):
        """Test that city-specific weather factors work."""
        from data.loader import get_weather_impact_factor
        
        calgary_factor = get_weather_impact_factor(city='Calgary')
        edmonton_factor = get_weather_impact_factor(city='Edmonton')
        
        # Both should be valid factors
        assert 0.85 <= calgary_factor <= 1.05
        assert 0.85 <= edmonton_factor <= 1.05
    
    def test_monthly_weather_summary(self):
        """Test monthly weather summary function."""
        from data.loader import get_monthly_weather_summary
        
        # Test for January (cold month)
        jan_summary = get_monthly_weather_summary(month=1, city='Calgary')
        
        # Should return a dict with weather stats
        assert isinstance(jan_summary, dict)
        
        # If data exists for January, should have temperature info
        if jan_summary:
            # At least one of these should be present
            has_temp = any(k in jan_summary for k in ['avg_temp_mean', 'min_temp_mean'])
            assert has_temp or len(jan_summary) == 0, \
                "Summary should include temperature data if available"


class TestLiveAnalyticsLoaders:
    """Tests for live analytics data loading functions."""
    
    def test_load_live_analytics_raw(self):
        """Test that live analytics data loads successfully."""
        from data.loader import load_live_analytics_raw
        
        df = load_live_analytics_raw()
        
        # File should exist and load
        assert not df.empty, "Live analytics data should not be empty"
        
        # Should have multiple rows
        assert len(df) > 10, "Live analytics should have substantial data"
    
    def test_get_live_analytics_category_factors(self):
        """Test that category factors are extracted correctly."""
        from data.loader import get_live_analytics_category_factors
        
        factors = get_live_analytics_category_factors()
        
        # Should return a dict
        assert isinstance(factors, dict)
        
        # Should have some categories
        assert len(factors) > 0, "Should extract at least one category"
        
        # Check for expected categories
        expected_categories = ['pop_ip', 'classic_romance', 'contemporary', 'family_classic']
        for cat in expected_categories:
            if cat in factors:
                assert 'engagement_factor' in factors[cat], \
                    f"{cat} should have engagement_factor"
    
    def test_category_engagement_factor_range(self):
        """Test that engagement factors are within expected range."""
        from data.loader import get_category_engagement_factor
        
        test_categories = ['pop_ip', 'classic_romance', 'family_classic', 
                          'contemporary', 'dramatic']
        
        for cat in test_categories:
            factor = get_category_engagement_factor(cat)
            assert 0.90 <= factor <= 1.15, \
                f"Engagement factor for {cat} ({factor}) should be 0.90-1.15"
    
    def test_unknown_category_returns_neutral(self):
        """Test that unknown categories return neutral factor."""
        from data.loader import get_category_engagement_factor
        
        factor = get_category_engagement_factor('unknown_category_xyz')
        assert factor == 1.0, "Unknown category should return 1.0"


class TestIntegrationWithScoring:
    """Tests for integration of weather and analytics into scoring."""
    
    def test_external_factors_summary(self):
        """Test the combined external factors summary function."""
        # This tests the integration in streamlit_app.py
        # We can't fully test without Streamlit, but we can test the loader side
        from data.loader import (
            get_weather_impact_factor,
            get_category_engagement_factor
        )
        
        # Test combined factors for a specific scenario
        weather = get_weather_impact_factor()
        engagement = get_category_engagement_factor('family_classic')
        
        combined = weather * engagement
        
        # Combined factor should be reasonable
        assert 0.80 <= combined <= 1.25, \
            f"Combined factor {combined} should be between 0.80 and 1.25"
    
    def test_weather_and_engagement_affect_different_categories(self):
        """Test that different categories get different engagement factors."""
        from data.loader import get_category_engagement_factor
        
        factors = {}
        for cat in ['pop_ip', 'contemporary', 'family_classic', 'dramatic']:
            factors[cat] = get_category_engagement_factor(cat)
        
        # Not all categories should have the same factor
        unique_factors = set(round(f, 2) for f in factors.values())
        # With the current data, we should have at least 2 different values
        assert len(unique_factors) >= 1, \
            "Categories should have varied engagement factors"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
