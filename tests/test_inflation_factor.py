"""Tests for inflation adjustment factor computation."""
import pytest
import pandas as pd
import numpy as np

from data.features import compute_inflation_adjustment_factor


class TestInflationFactorDeterminism:
    """Tests for deterministic inflation factor calculation."""
    
    @pytest.fixture
    def cpi_data(self):
        """Standard CPI data for testing."""
        return pd.DataFrame({
            'date': [
                '2020-01-01', '2020-06-01', '2020-12-01',
                '2021-01-01', '2021-06-01', '2021-12-01',
                '2022-01-01', '2022-06-01', '2022-12-01',
                '2023-01-01', '2023-06-01', '2023-12-01',
                '2024-01-01', '2024-06-01'
            ],
            'V41690973': [
                136.8, 137.2, 137.4,  # 2020
                138.2, 141.4, 144.0,  # 2021
                145.3, 152.9, 153.1,  # 2022
                153.9, 157.2, 158.3,  # 2023
                158.3, 161.4          # 2024
            ]
        })
    
    def test_same_input_produces_same_output(self, cpi_data):
        """Same input should always produce identical output."""
        shows = pd.DataFrame({
            'show_date': ['2023-06-15', '2024-01-20']
        })
        
        # Run multiple times
        results = []
        for _ in range(3):
            result = compute_inflation_adjustment_factor(
                shows.copy(),
                cpi_data.copy(),
                date_column='show_date',
                base_date='2020-01-01'
            )
            results.append(result['inflation_adjustment_factor'].tolist())
        
        # All runs should produce identical results
        assert results[0] == results[1] == results[2], \
            f"Non-deterministic results: {results}"
    
    def test_order_independence(self, cpi_data):
        """Result should not depend on row order."""
        shows1 = pd.DataFrame({
            'show_date': ['2023-01-15', '2024-01-15']
        })
        shows2 = pd.DataFrame({
            'show_date': ['2024-01-15', '2023-01-15']  # Reversed order
        })
        
        result1 = compute_inflation_adjustment_factor(
            shows1, cpi_data, date_column='show_date', base_date='2020-01-01'
        )
        result2 = compute_inflation_adjustment_factor(
            shows2, cpi_data, date_column='show_date', base_date='2020-01-01'
        )
        
        # First show in result1 should match second show in result2
        factor_2023_run1 = result1['inflation_adjustment_factor'].iloc[0]
        factor_2023_run2 = result2['inflation_adjustment_factor'].iloc[1]
        
        assert factor_2023_run1 == factor_2023_run2, \
            f"Order-dependent results: {factor_2023_run1} vs {factor_2023_run2}"


class TestInflationMonotonicity:
    """Tests for monotonicity when CPI rises."""
    
    @pytest.fixture
    def strictly_rising_cpi(self):
        """CPI data that strictly increases over time."""
        dates = pd.date_range(start='2020-01-01', periods=48, freq='MS')
        # CPI increasing by about 0.5 per month
        cpi_values = [136.8 + i * 0.5 for i in range(48)]
        return pd.DataFrame({
            'date': dates,
            'V41690973': cpi_values
        })
    
    def test_inflation_factor_monotonic_with_rising_cpi(self, strictly_rising_cpi):
        """When CPI only rises, inflation factor should be monotonically increasing."""
        # Create shows at regular intervals
        show_dates = pd.date_range(start='2020-03-15', periods=12, freq='4MS')
        shows = pd.DataFrame({
            'show_date': show_dates
        })
        
        result = compute_inflation_adjustment_factor(
            shows,
            strictly_rising_cpi,
            date_column='show_date',
            base_date='2020-01-01'
        )
        
        factors = result['inflation_adjustment_factor'].tolist()
        
        # Each factor should be >= previous
        for i in range(1, len(factors)):
            assert factors[i] >= factors[i-1], \
                f"Non-monotonic at position {i}: {factors[i-1]} -> {factors[i]}"
    
    def test_higher_cpi_means_higher_factor(self, strictly_rising_cpi):
        """Shows in periods with higher CPI should have higher factors."""
        early_show = pd.DataFrame({'show_date': ['2020-06-15']})
        late_show = pd.DataFrame({'show_date': ['2023-06-15']})
        
        result_early = compute_inflation_adjustment_factor(
            early_show, strictly_rising_cpi, date_column='show_date', base_date='2020-01-01'
        )
        result_late = compute_inflation_adjustment_factor(
            late_show, strictly_rising_cpi, date_column='show_date', base_date='2020-01-01'
        )
        
        factor_early = result_early['inflation_adjustment_factor'].iloc[0]
        factor_late = result_late['inflation_adjustment_factor'].iloc[0]
        
        assert factor_late > factor_early, \
            f"Later show should have higher factor: {factor_early} vs {factor_late}"


class TestInflationFactorFormula:
    """Tests for correct formula implementation."""
    
    def test_formula_calculation(self):
        """Verify the formula: factor = current_cpi / base_cpi."""
        cpi_data = pd.DataFrame({
            'date': ['2020-01-01', '2024-01-01'],
            'V41690973': [100.0, 120.0]  # Simple values for easy math
        })
        
        shows = pd.DataFrame({'show_date': ['2024-01-15']})
        
        result = compute_inflation_adjustment_factor(
            shows, cpi_data, date_column='show_date', base_date='2020-01-01'
        )
        
        factor = result['inflation_adjustment_factor'].iloc[0]
        expected = 120.0 / 100.0  # = 1.2
        
        assert abs(factor - expected) < 0.001, \
            f"Expected factor {expected}, got {factor}"
    
    def test_factor_at_base_date_is_one(self):
        """Factor at base date should be exactly 1.0."""
        cpi_data = pd.DataFrame({
            'date': ['2020-01-01', '2020-02-01'],
            'V41690973': [100.0, 100.5]
        })
        
        shows = pd.DataFrame({'show_date': ['2020-01-15']})
        
        result = compute_inflation_adjustment_factor(
            shows, cpi_data, date_column='show_date', base_date='2020-01-01'
        )
        
        factor = result['inflation_adjustment_factor'].iloc[0]
        
        # Should be 1.0 (same CPI as base)
        assert abs(factor - 1.0) < 0.01, \
            f"Expected factor 1.0 at base date, got {factor}"
    
    def test_deflation_produces_factor_below_one(self):
        """If CPI drops (deflation), factor should be below 1."""
        cpi_data = pd.DataFrame({
            'date': ['2020-01-01', '2021-01-01'],
            'V41690973': [100.0, 95.0]  # CPI dropped
        })
        
        shows = pd.DataFrame({'show_date': ['2021-01-15']})
        
        result = compute_inflation_adjustment_factor(
            shows, cpi_data, date_column='show_date', base_date='2020-01-01'
        )
        
        factor = result['inflation_adjustment_factor'].iloc[0]
        
        assert factor < 1.0, \
            f"Deflation should produce factor < 1, got {factor}"


class TestEdgeCases:
    """Tests for edge cases in inflation factor computation."""
    
    def test_handles_empty_shows(self):
        """Should handle empty show DataFrame."""
        cpi_data = pd.DataFrame({
            'date': ['2020-01-01'],
            'V41690973': [100.0]
        })
        
        result = compute_inflation_adjustment_factor(
            pd.DataFrame({'show_date': []}),
            cpi_data,
            date_column='show_date'
        )
        
        assert 'inflation_adjustment_factor' in result.columns
        assert len(result) == 0
    
    def test_handles_empty_cpi(self):
        """Should handle empty CPI DataFrame."""
        shows = pd.DataFrame({'show_date': ['2024-01-15']})
        
        result = compute_inflation_adjustment_factor(
            shows,
            pd.DataFrame(),
            date_column='show_date'
        )
        
        # Should return original DataFrame when no CPI data
        # The function returns the input unchanged when CPI is empty
        assert len(result) == len(shows)
    
    def test_handles_missing_date_column(self):
        """Should handle missing date column gracefully."""
        shows = pd.DataFrame({'title': ['Show A']})  # No date column
        cpi_data = pd.DataFrame({
            'date': ['2020-01-01'],
            'V41690973': [100.0]
        })
        
        result = compute_inflation_adjustment_factor(
            shows,
            cpi_data,
            date_column='show_date'  # This column doesn't exist
        )
        
        # Should have default factor
        assert 'inflation_adjustment_factor' in result.columns
    
    def test_handles_null_dates(self):
        """Should handle null/NaN dates in show data."""
        shows = pd.DataFrame({
            'show_date': ['2024-01-15', None, '2024-03-20']
        })
        cpi_data = pd.DataFrame({
            'date': ['2024-01-01', '2024-03-01'],
            'V41690973': [100.0, 101.0]
        })
        
        result = compute_inflation_adjustment_factor(
            shows, cpi_data, date_column='show_date', base_date='2024-01-01'
        )
        
        # Should not crash
        assert len(result) == 3
        # First and third should have valid factors
        assert pd.notna(result['inflation_adjustment_factor'].iloc[0])
        assert pd.notna(result['inflation_adjustment_factor'].iloc[2])
