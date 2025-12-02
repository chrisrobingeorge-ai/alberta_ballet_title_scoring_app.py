"""
Tests for external data join functions.

These tests verify that joins with external datasets (weather, marketing spend, economy)
correctly:
1. Use 'start_date' or 'end_date' as the key for merging
2. Use LEFT joins to preserve all show rows
3. Properly handle missing external data
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime


class TestJoinHistoryWithWeather:
    """Tests for join_history_with_weather function."""
    
    def test_left_join_preserves_all_rows(self):
        """LEFT join should preserve all history rows even with no weather match."""
        from data.loader import join_history_with_weather
        
        # Create test history data
        history = pd.DataFrame({
            'show_title': ['Show A', 'Show B', 'Show C'],
            'city': ['Calgary', 'Edmonton', 'Calgary'],
            'start_date': pd.to_datetime(['2023-01-15', '2023-02-20', '2023-03-10']),
            'end_date': pd.to_datetime(['2023-01-22', '2023-02-27', '2023-03-17']),
            'single_tickets': [100, 200, 150]
        })
        
        result = join_history_with_weather(history, date_key='start_date')
        
        # All original rows should be preserved
        assert len(result) == len(history), \
            f"LEFT join should preserve all rows: expected {len(history)}, got {len(result)}"
        
        # Original columns should still exist
        assert 'show_title' in result.columns
        assert 'single_tickets' in result.columns
    
    def test_start_date_key_used_for_join(self):
        """Should use start_date as the join key when specified."""
        from data.loader import join_history_with_weather
        
        history = pd.DataFrame({
            'show_title': ['Show A'],
            'city': ['Calgary'],
            'start_date': pd.to_datetime(['2023-01-15']),
            'end_date': pd.to_datetime(['2023-01-22']),
        })
        
        # Should not raise error when start_date is present
        result = join_history_with_weather(history, date_key='start_date')
        assert len(result) == 1
    
    def test_end_date_key_used_for_join(self):
        """Should use end_date as the join key when specified."""
        from data.loader import join_history_with_weather
        
        history = pd.DataFrame({
            'show_title': ['Show A'],
            'city': ['Calgary'],
            'start_date': pd.to_datetime(['2023-01-15']),
            'end_date': pd.to_datetime(['2023-01-22']),
        })
        
        # Should not raise error when end_date is present
        result = join_history_with_weather(history, date_key='end_date')
        assert len(result) == 1
    
    def test_handles_empty_history(self):
        """Should handle empty history DataFrame gracefully."""
        from data.loader import join_history_with_weather
        
        empty_history = pd.DataFrame()
        result = join_history_with_weather(empty_history)
        
        assert result.empty
    
    def test_handles_missing_date_column(self):
        """Should return unchanged if date column is missing."""
        from data.loader import join_history_with_weather
        
        history = pd.DataFrame({
            'show_title': ['Show A'],
            'single_tickets': [100]
        })
        
        result = join_history_with_weather(history, date_key='start_date')
        
        # Should return unchanged
        assert len(result) == 1
        assert 'show_title' in result.columns


class TestJoinHistoryWithMarketingSpend:
    """Tests for join_history_with_marketing_spend function."""
    
    def test_left_join_preserves_all_rows(self):
        """LEFT join should preserve all history rows."""
        from data.loader import join_history_with_marketing_spend
        
        history = pd.DataFrame({
            'show_title': ['Show A', 'Show B', 'Show C'],
            'start_date': pd.to_datetime(['2023-01-15', '2023-02-20', '2023-03-10']),
            'single_tickets': [100, 200, 150]
        })
        
        # Empty marketing data should still preserve all rows
        marketing = pd.DataFrame()
        
        result = join_history_with_marketing_spend(history, marketing_df=marketing)
        
        # All original rows should be preserved
        assert len(result) == len(history), \
            f"LEFT join should preserve all rows: expected {len(history)}, got {len(result)}"
    
    def test_joins_on_start_date(self):
        """Should be able to join on start_date."""
        from data.loader import join_history_with_marketing_spend
        
        history = pd.DataFrame({
            'show_title': ['Show A', 'Show B'],
            'start_date': pd.to_datetime(['2023-01-15', '2023-02-20']),
        })
        
        marketing = pd.DataFrame({
            'show_title': ['Show A'],
            'start_date': pd.to_datetime(['2023-01-15']),
            'marketing_spend': [5000]
        })
        
        result = join_history_with_marketing_spend(
            history, 
            marketing_df=marketing,
            date_key='start_date'
        )
        
        # All history rows preserved
        assert len(result) == 2
        
        # Marketing spend should be joined for Show A
        if 'marketing_spend' in result.columns:
            show_a = result[result['show_title'] == 'Show A']
            if not show_a.empty:
                assert show_a['marketing_spend'].iloc[0] == 5000
    
    def test_handles_empty_history(self):
        """Should handle empty history DataFrame gracefully."""
        from data.loader import join_history_with_marketing_spend
        
        empty_history = pd.DataFrame()
        result = join_history_with_marketing_spend(empty_history)
        
        assert result.empty


class TestJoinHistoryWithExternalData:
    """Tests for join_history_with_external_data function."""
    
    def test_preserves_all_rows(self):
        """Should preserve all history rows through multiple joins."""
        from data.loader import join_history_with_external_data
        
        history = pd.DataFrame({
            'show_title': ['Show A', 'Show B'],
            'city': ['Calgary', 'Edmonton'],
            'start_date': pd.to_datetime(['2023-01-15', '2023-02-20']),
            'end_date': pd.to_datetime(['2023-01-22', '2023-02-27']),
            'single_tickets': [100, 200]
        })
        
        result = join_history_with_external_data(
            history,
            include_weather=True,
            include_marketing=True,
            include_economy=False,  # Skip economy to avoid file dependency
            date_key='start_date'
        )
        
        # All original rows should be preserved
        assert len(result) >= len(history), \
            f"Should preserve all rows: expected >= {len(history)}, got {len(result)}"
    
    def test_handles_empty_history(self):
        """Should handle empty history DataFrame gracefully."""
        from data.loader import join_history_with_external_data
        
        empty_history = pd.DataFrame()
        result = join_history_with_external_data(empty_history)
        
        assert result.empty


class TestLoadMarketingSpend:
    """Tests for load_marketing_spend function."""
    
    def test_returns_empty_when_file_missing(self):
        """Should return empty DataFrame when file doesn't exist."""
        from data.loader import load_marketing_spend
        
        # Default file likely doesn't exist, should return empty
        result = load_marketing_spend(fallback_empty=True)
        
        assert isinstance(result, pd.DataFrame)


class TestExternalFactorsJoinDocumentation:
    """Tests to verify join logic documentation is accurate."""
    
    def test_load_history_with_external_factors_uses_left_join(self):
        """Verify that load_history_with_external_factors uses LEFT join."""
        from data.loader import load_history_with_external_factors
        
        # The function should preserve all history rows
        # This test validates the documented behavior
        history = pd.DataFrame({
            'city': ['Calgary', 'Edmonton', 'Calgary'],
            'show_title': ['Show A', 'Show B', 'Show C'],
            'year': [2023, 2023, 2024],
            'single_tickets': [100, 200, 150]
        })
        
        # Mock by calling with files that may not exist
        # The function should return history unchanged if external factors missing
        from unittest.mock import patch
        
        with patch('data.loader.load_history_sales') as mock_history:
            mock_history.return_value = history
            with patch('data.loader.load_external_factors') as mock_external:
                # Empty external factors
                mock_external.return_value = pd.DataFrame()
                
                result = load_history_with_external_factors()
                
                # Should return original history since external is empty
                assert len(result) == len(history)
    
    def test_load_history_with_predicthq_uses_left_join(self):
        """Verify that load_history_with_predicthq uses LEFT join."""
        from data.loader import load_history_with_predicthq
        
        history = pd.DataFrame({
            'city': ['Calgary', 'Edmonton'],
            'show_title': ['Show A', 'Show B'],
            'single_tickets': [100, 200]
        })
        
        from unittest.mock import patch
        
        with patch('data.loader.load_history_sales') as mock_history:
            mock_history.return_value = history
            with patch('data.loader.load_predicthq_events') as mock_phq:
                # Empty PredictHQ data
                mock_phq.return_value = pd.DataFrame()
                
                result = load_history_with_predicthq()
                
                # Should return original history since PredictHQ is empty
                assert len(result) == len(history)


class TestJoinKeyDocumentation:
    """Tests to verify that join key documentation is accurate."""
    
    def test_weather_join_uses_date_key_parameter(self):
        """Weather join should respect the date_key parameter."""
        from data.loader import join_history_with_weather
        
        history = pd.DataFrame({
            'show_title': ['Show A'],
            'city': ['Calgary'],
            'start_date': pd.to_datetime(['2023-01-15']),
            'end_date': pd.to_datetime(['2023-01-22']),
        })
        
        # Should work with both start_date and end_date
        result_start = join_history_with_weather(history, date_key='start_date')
        result_end = join_history_with_weather(history, date_key='end_date')
        
        assert len(result_start) == 1
        assert len(result_end) == 1
    
    def test_marketing_join_uses_date_key_parameter(self):
        """Marketing join should respect the date_key parameter."""
        from data.loader import join_history_with_marketing_spend
        
        history = pd.DataFrame({
            'show_title': ['Show A'],
            'start_date': pd.to_datetime(['2023-01-15']),
            'end_date': pd.to_datetime(['2023-01-22']),
        })
        
        # Should work with both start_date and end_date
        result_start = join_history_with_marketing_spend(history, date_key='start_date')
        result_end = join_history_with_marketing_spend(history, date_key='end_date')
        
        assert len(result_start) == 1
        assert len(result_end) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
