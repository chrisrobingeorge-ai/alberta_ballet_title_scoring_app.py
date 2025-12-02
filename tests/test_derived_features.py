"""
Tests for derived feature engineering functions.

These tests verify that feature derivation functions correctly:
1. Create marketing features (lagged spend, total spend, spend per ticket)
2. Create weather features (temperature, precipitation, day-of-week effects)
3. Create economy features (unemployment rate, oil price, CPI)
4. Create baseline features (wiki/trends/youtube/spotify signals)
5. Handle missing data gracefully
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime


class TestDeriveMarketingFeatures:
    """Tests for derive_marketing_features function."""
    
    def test_creates_spend_per_ticket(self):
        """Should calculate marketing spend per ticket."""
        from data.features import derive_marketing_features
        
        df = pd.DataFrame({
            'show_title': ['Show A', 'Show B', 'Show C'],
            'start_date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01']),
            'marketing_spend': [10000, 20000, 15000],
            'single_tickets': [1000, 2000, 1500]
        })
        
        result = derive_marketing_features(df)
        
        assert 'marketing_spend_per_ticket' in result.columns
        assert result['marketing_spend_per_ticket'].iloc[0] == pytest.approx(10.0)
        assert result['marketing_spend_per_ticket'].iloc[1] == pytest.approx(10.0)
        assert result['marketing_spend_per_ticket'].iloc[2] == pytest.approx(10.0)
    
    def test_creates_lagged_spend(self):
        """Should create lagged marketing spend features."""
        from data.features import derive_marketing_features
        
        df = pd.DataFrame({
            'show_title': ['Show A', 'Show B', 'Show C'],
            'start_date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01']),
            'marketing_spend': [10000, 20000, 15000],
            'single_tickets': [1000, 2000, 1500]
        })
        
        result = derive_marketing_features(df, lag_periods=[1])
        
        assert 'marketing_spend_lag_1' in result.columns
        # First row should have NaN lag (no prior data)
        assert pd.isna(result['marketing_spend_lag_1'].iloc[0])
        # Second row should have first row's spend
        assert result['marketing_spend_lag_1'].iloc[1] == 10000
    
    def test_handles_missing_spend_column(self):
        """Should handle missing marketing spend column gracefully."""
        from data.features import derive_marketing_features
        
        df = pd.DataFrame({
            'show_title': ['Show A', 'Show B'],
            'start_date': pd.to_datetime(['2023-01-01', '2023-02-01']),
            'single_tickets': [1000, 2000]
        })
        
        result = derive_marketing_features(df)
        
        # Should create columns with NaN values
        assert 'marketing_spend_total' in result.columns
        assert 'marketing_spend_per_ticket' in result.columns
        assert result['marketing_spend_total'].isna().all()
    
    def test_handles_zero_tickets(self):
        """Should handle zero tickets without division error."""
        from data.features import derive_marketing_features
        
        df = pd.DataFrame({
            'show_title': ['Show A'],
            'start_date': pd.to_datetime(['2023-01-01']),
            'marketing_spend': [10000],
            'single_tickets': [0]  # Zero tickets
        })
        
        result = derive_marketing_features(df)
        
        # Should result in NaN, not infinity
        assert pd.isna(result['marketing_spend_per_ticket'].iloc[0])


class TestDeriveWeatherFeatures:
    """Tests for derive_weather_features function."""
    
    def test_creates_temperature_normalized(self):
        """Should normalize temperature to 0-1 range."""
        from data.features import derive_weather_features
        
        df = pd.DataFrame({
            'start_date': pd.to_datetime(['2023-01-15', '2023-07-15']),
            'weather_avg_temperature': [-20, 25]  # Winter and summer
        })
        
        result = derive_weather_features(df)
        
        assert 'weather_temp_normalized' in result.columns
        # -20C should normalize to about 0.27 ((−20 + 40) / 75)
        assert 0.2 < result['weather_temp_normalized'].iloc[0] < 0.4
        # 25C should normalize to about 0.87 ((25 + 40) / 75)
        assert 0.8 < result['weather_temp_normalized'].iloc[1] < 1.0
    
    def test_creates_extreme_cold_flag(self):
        """Should flag extreme cold conditions (<-20°C)."""
        from data.features import derive_weather_features
        
        df = pd.DataFrame({
            'start_date': pd.to_datetime(['2023-01-15', '2023-01-16']),
            'weather_min_temperature': [-25, -15]
        })
        
        result = derive_weather_features(df)
        
        assert 'weather_extreme_cold_flag' in result.columns
        assert result['weather_extreme_cold_flag'].iloc[0] == 1  # Below -20
        assert result['weather_extreme_cold_flag'].iloc[1] == 0  # Above -20
    
    def test_creates_day_of_week_features(self):
        """Should create day-of-week and weekend flags."""
        from data.features import derive_weather_features
        
        df = pd.DataFrame({
            'start_date': pd.to_datetime([
                '2023-01-16',  # Monday
                '2023-01-21'   # Saturday
            ])
        })
        
        result = derive_weather_features(df)
        
        assert 'weather_day_of_week' in result.columns
        assert 'weather_weekend_flag' in result.columns
        assert result['weather_day_of_week'].iloc[0] == 0  # Monday
        assert result['weather_weekend_flag'].iloc[0] == 0
        assert result['weather_day_of_week'].iloc[1] == 5  # Saturday
        assert result['weather_weekend_flag'].iloc[1] == 1
    
    def test_handles_missing_weather_data(self):
        """Should handle missing weather columns gracefully."""
        from data.features import derive_weather_features
        
        df = pd.DataFrame({
            'start_date': pd.to_datetime(['2023-01-15']),
            'single_tickets': [1000]
        })
        
        result = derive_weather_features(df)
        
        # Should create columns with default/NaN values
        assert 'weather_temp_normalized' in result.columns
        assert 'weather_extreme_cold_flag' in result.columns


class TestDeriveEconomyFeatures:
    """Tests for derive_economy_features function."""
    
    def test_joins_unemployment_rate(self):
        """Should join unemployment rate via temporal matching."""
        from data.features import derive_economy_features
        
        df = pd.DataFrame({
            'show_title': ['Show A', 'Show B'],
            'start_date': pd.to_datetime(['2023-01-15', '2023-06-15']),
            'city': ['Calgary', 'Edmonton']
        })
        
        unemployment = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-06-01']),
            'unemployment_rate': [5.5, 6.0],
            'region': ['Alberta', 'Alberta']
        })
        
        result = derive_economy_features(df, unemployment_df=unemployment)
        
        assert 'economy_unemployment_rate' in result.columns
        # Should have matched to nearest prior date
        assert pd.notna(result['economy_unemployment_rate'].iloc[0])
    
    def test_joins_oil_price(self):
        """Should join oil price via temporal matching."""
        from data.features import derive_economy_features
        
        df = pd.DataFrame({
            'show_title': ['Show A'],
            'start_date': pd.to_datetime(['2023-03-15']),
            'city': ['Calgary']
        })
        
        oil = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01']),
            'wcs_oil_price': [50.0, 55.0, 60.0]
        })
        
        result = derive_economy_features(df, oil_price_df=oil)
        
        assert 'economy_oil_price' in result.columns
        assert result['economy_oil_price'].iloc[0] == pytest.approx(60.0)
    
    def test_creates_oil_change_feature(self):
        """Should calculate 3-month oil price change."""
        from data.features import derive_economy_features
        
        df = pd.DataFrame({
            'show_title': ['Show A'],
            'start_date': pd.to_datetime(['2023-06-15']),
            'city': ['Calgary']
        })
        
        # Oil prices with enough history for 3-month lag
        oil = pd.DataFrame({
            'date': pd.to_datetime([
                '2023-01-01', '2023-02-01', '2023-03-01',
                '2023-04-01', '2023-05-01', '2023-06-01'
            ]),
            'wcs_oil_price': [50.0, 52.0, 54.0, 56.0, 58.0, 60.0]
        })
        
        result = derive_economy_features(df, oil_price_df=oil)
        
        assert 'economy_oil_change_3m' in result.columns
    
    def test_handles_missing_date_column(self):
        """Should handle missing date column gracefully."""
        from data.features import derive_economy_features
        
        df = pd.DataFrame({
            'show_title': ['Show A'],
            'city': ['Calgary']
        })
        
        result = derive_economy_features(df)
        
        assert 'economy_unemployment_rate' in result.columns
        assert result['economy_unemployment_rate'].isna().all()


class TestDeriveBaselineFeatures:
    """Tests for derive_baseline_features function."""
    
    def test_joins_baseline_signals(self):
        """Should join baseline signals by title."""
        from data.features import derive_baseline_features
        
        df = pd.DataFrame({
            'show_title': ['Nutcracker', 'Swan Lake'],
            'single_tickets': [5000, 4000]
        })
        
        baselines = pd.DataFrame({
            'title': ['Nutcracker', 'Swan Lake'],
            'wiki': [90, 85],
            'trends': [80, 75],
            'youtube': [95, 90],
            'spotify': [70, 65]
        })
        
        result = derive_baseline_features(df, baselines_df=baselines)
        
        assert 'baseline_wiki' in result.columns
        assert 'baseline_trends' in result.columns
        assert 'baseline_youtube' in result.columns
        assert 'baseline_spotify' in result.columns
        assert result['baseline_wiki'].iloc[0] == 90
    
    def test_creates_familiarity_index(self):
        """Should compute composite familiarity index."""
        from data.features import derive_baseline_features
        
        df = pd.DataFrame({
            'show_title': ['Nutcracker'],
            'single_tickets': [5000]
        })
        
        baselines = pd.DataFrame({
            'title': ['Nutcracker'],
            'wiki': [100],
            'trends': [100],
            'youtube': [100],
            'spotify': [100]
        })
        
        result = derive_baseline_features(df, baselines_df=baselines)
        
        assert 'baseline_familiarity_index' in result.columns
        # All 100s should give familiarity index of 100
        assert result['baseline_familiarity_index'].iloc[0] == pytest.approx(100.0)
    
    def test_creates_digital_presence_score(self):
        """Should compute digital presence score."""
        from data.features import derive_baseline_features
        
        df = pd.DataFrame({
            'show_title': ['Nutcracker'],
            'single_tickets': [5000]
        })
        
        baselines = pd.DataFrame({
            'title': ['Nutcracker'],
            'wiki': [80],
            'trends': [60],
            'youtube': [90],
            'spotify': [50]
        })
        
        result = derive_baseline_features(df, baselines_df=baselines)
        
        assert 'baseline_digital_presence' in result.columns
        # Expected: 0.25*80 + 0.25*60 + 0.30*90 + 0.20*50 = 72
        assert result['baseline_digital_presence'].iloc[0] == pytest.approx(72.0)
    
    def test_handles_missing_baselines(self):
        """Should handle missing baselines data gracefully."""
        from data.features import derive_baseline_features
        
        df = pd.DataFrame({
            'show_title': ['Unknown Show'],
            'single_tickets': [1000]
        })
        
        result = derive_baseline_features(df, baselines_df=None)
        
        assert 'baseline_wiki' in result.columns
        assert result['baseline_wiki'].isna().all()
    
    def test_case_insensitive_title_matching(self):
        """Should match titles case-insensitively."""
        from data.features import derive_baseline_features
        
        df = pd.DataFrame({
            'show_title': ['NUTCRACKER'],  # Uppercase
            'single_tickets': [5000]
        })
        
        baselines = pd.DataFrame({
            'title': ['Nutcracker'],  # Mixed case
            'wiki': [90],
            'trends': [80],
            'youtube': [95],
            'spotify': [70]
        })
        
        result = derive_baseline_features(df, baselines_df=baselines)
        
        assert result['baseline_wiki'].iloc[0] == 90


class TestDeriveAllExternalFeatures:
    """Tests for derive_all_external_features function."""
    
    def test_combines_all_feature_types(self):
        """Should create features from all external data sources."""
        from data.features import derive_all_external_features
        
        df = pd.DataFrame({
            'show_title': ['Nutcracker', 'Swan Lake'],
            'start_date': pd.to_datetime(['2023-12-01', '2023-02-14']),
            'city': ['Calgary', 'Edmonton'],
            'single_tickets': [5000, 4000],
            'marketing_spend': [50000, 40000]
        })
        
        baselines = pd.DataFrame({
            'title': ['Nutcracker', 'Swan Lake'],
            'wiki': [90, 85],
            'trends': [80, 75],
            'youtube': [95, 90],
            'spotify': [70, 65]
        })
        
        result = derive_all_external_features(
            df,
            baselines_df=baselines
        )
        
        # Should have marketing features
        assert 'marketing_spend_total' in result.columns
        assert 'marketing_spend_per_ticket' in result.columns
        
        # Should have weather features
        assert 'weather_day_of_week' in result.columns
        
        # Should have economy features
        assert 'economy_unemployment_rate' in result.columns
        
        # Should have baseline features
        assert 'baseline_wiki' in result.columns
        assert 'baseline_familiarity_index' in result.columns
    
    def test_handles_empty_dataframe(self):
        """Should handle empty DataFrame gracefully."""
        from data.features import derive_all_external_features
        
        df = pd.DataFrame()
        result = derive_all_external_features(df)
        
        assert result.empty


class TestFeatureDocumentation:
    """Tests to verify feature documentation is accurate."""
    
    def test_marketing_features_documented(self):
        """Verify marketing feature names match documentation."""
        from data.features import derive_marketing_features
        
        df = pd.DataFrame({
            'show_title': ['Show A'],
            'start_date': pd.to_datetime(['2023-01-01']),
            'marketing_spend': [10000],
            'single_tickets': [1000]
        })
        
        result = derive_marketing_features(df)
        
        # These should match feature inventory documentation
        documented_features = [
            'marketing_spend_total',
            'marketing_spend_per_ticket',
            'marketing_spend_lag_1'
        ]
        
        for feature in documented_features:
            assert feature in result.columns, f"Feature '{feature}' should be documented"
    
    def test_economy_features_documented(self):
        """Verify economy feature names match documentation."""
        from data.features import derive_economy_features
        
        df = pd.DataFrame({
            'show_title': ['Show A'],
            'start_date': pd.to_datetime(['2023-01-01']),
            'city': ['Calgary']
        })
        
        result = derive_economy_features(df)
        
        # These should match feature inventory documentation
        documented_features = [
            'economy_unemployment_rate',
            'economy_oil_price',
            'economy_cpi'
        ]
        
        for feature in documented_features:
            assert feature in result.columns, f"Feature '{feature}' should be documented"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
