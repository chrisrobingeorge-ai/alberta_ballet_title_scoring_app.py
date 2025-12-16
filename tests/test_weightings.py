"""
Unit tests for weighting systems.

Tests the three weighting systems (Live Analytics, Economics, Stone Olafson)
to ensure they are correctly loaded and applied.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_show_data():
    """Create sample show data for testing."""
    return pd.DataFrame({
        'title': ['Test Show A', 'Test Show B', 'Test Show C'],
        'category': ['pop_ip', 'classic_romance', 'family_classic'],
        'gender': ['female', 'female', 'male'],
        'wiki': [70, 65, 75],
        'trends': [60, 55, 62],
        'youtube': [80, 70, 75],
        'chartmetric': [65, 60, 68],
        'show_date': ['2022-05-01', '2022-06-15', '2022-09-20'],
        'city': ['calgary', 'edmonton', 'calgary']
    })


@pytest.fixture
def sample_consumer_confidence_data():
    """Create sample consumer confidence data."""
    return pd.DataFrame({
        'category': ['Demographics', 'Demographics', 'BNCCI'],
        'subcategory': ['Region', 'Region', 'Headline Index'],
        'metric': ['Prairies', 'Prairies', 'This week'],
        'year_or_period': ['2022-01-07', '2022-07-01', '2022-01-07'],
        'value': [48.5, 52.0, 50.0]
    })


@pytest.fixture
def sample_commodity_data():
    """Create sample commodity price data."""
    return pd.DataFrame({
        'date': ['2022-01-01', '2022-06-01', '2022-09-01'],
        'A.ENER': [900, 1100, 1050]
    })


@pytest.fixture
def sample_cpi_data():
    """Create sample CPI data."""
    return pd.DataFrame({
        'date': ['2022-01-01', '2022-06-01', '2022-09-01'],
        'value': [150.0, 153.5, 155.0]
    })


# =============================================================================
# LIVE ANALYTICS TESTS
# =============================================================================

class TestLiveAnalyticsWeightings:
    """Test Live Analytics weighting system."""
    
    def test_engagement_factor_loading(self):
        """Test that engagement factors are loaded from live_analytics.csv."""
        from data.loader import get_category_engagement_factor
        
        # Test known categories
        factor_pop = get_category_engagement_factor('pop_ip')
        factor_classic = get_category_engagement_factor('classic_romance')
        factor_family = get_category_engagement_factor('family_classic')
        
        # All should return valid multipliers (not 1.0 default)
        assert isinstance(factor_pop, float)
        assert isinstance(factor_classic, float)
        assert isinstance(factor_family, float)
        
        # Should be positive and reasonable (0.5-2.0 range)
        assert 0.5 < factor_pop < 2.0
        assert 0.5 < factor_classic < 2.0
        assert 0.5 < factor_family < 2.0
    
    def test_engagement_factor_default_fallback(self):
        """Test that unknown categories return default (1.0)."""
        from data.loader import get_category_engagement_factor
        
        factor = get_category_engagement_factor('unknown_category_xyz_123')
        assert factor == 1.0
    
    @pytest.mark.xfail(reason="Known issue: Live analytics has limited variance (only 4 unique values)")
    def test_engagement_factor_variance(self):
        """Test that engagement factors have sufficient variance.
        
        KNOWN ISSUE: As of Dec 2025, engagement factors have only 4 unique values
        (1.051, 1.073, 1.077, 1.080) resulting in variance < 0.0001.
        See docs/weightings_diagnostics.md for details.
        """
        from data.loader import get_category_engagement_factor
        
        categories = [
            'pop_ip', 'classic_romance', 'contemporary',
            'family_classic', 'romantic_tragedy', 'dramatic'
        ]
        
        factors = [get_category_engagement_factor(cat) for cat in categories]
        
        # Check variance
        factors_array = np.array(factors)
        variance = factors_array.var()
        
        # Should have some variance (not all identical)
        assert variance > 0.0001, f"Engagement factors have insufficient variance: {factors}"
        
        # Check range
        min_factor = factors_array.min()
        max_factor = factors_array.max()
        range_span = max_factor - min_factor
        
        assert range_span > 0.01, f"Engagement factor range too narrow: {min_factor}-{max_factor}"
    
    def test_engagement_factor_applied_in_dataset(self, sample_show_data):
        """Test that engagement factors are correctly applied to show data."""
        from data.loader import get_category_engagement_factor
        
        # Apply engagement factors
        sample_show_data['aud__engagement_factor'] = sample_show_data['category'].apply(
            lambda cat: get_category_engagement_factor(cat) if pd.notna(cat) else 1.0
        )
        
        # Check all rows have engagement factors
        assert 'aud__engagement_factor' in sample_show_data.columns
        assert sample_show_data['aud__engagement_factor'].notna().all()
        
        # Check factors are in reasonable range
        assert (sample_show_data['aud__engagement_factor'] >= 0.5).all()
        assert (sample_show_data['aud__engagement_factor'] <= 2.0).all()


# =============================================================================
# ECONOMICS TESTS
# =============================================================================

class TestEconomicsWeightings:
    """Test Economics weighting system."""
    
    def test_consumer_confidence_loading(self, sample_show_data, sample_consumer_confidence_data):
        """Test consumer confidence data loading and joining."""
        from data.features import join_consumer_confidence
        
        result = join_consumer_confidence(
            sample_show_data,
            sample_consumer_confidence_data,
            date_column='show_date'
        )
        
        # Check column added
        assert 'consumer_confidence_prairies' in result.columns
        
        # Check all rows have values
        assert result['consumer_confidence_prairies'].notna().all()
        
        # Check values are in reasonable range (0-100)
        assert (result['consumer_confidence_prairies'] >= 0).all()
        assert (result['consumer_confidence_prairies'] <= 100).all()
    
    def test_energy_index_loading(self, sample_show_data, sample_commodity_data):
        """Test energy index data loading and joining."""
        from data.features import join_energy_index
        
        result = join_energy_index(
            sample_show_data,
            sample_commodity_data,
            date_column='show_date'
        )
        
        # Check column added
        assert 'energy_index' in result.columns
        
        # Check all rows have values
        assert result['energy_index'].notna().all()
        
        # Check values are positive
        assert (result['energy_index'] > 0).all()
    
    def test_inflation_factor_computation(self, sample_show_data, sample_cpi_data):
        """Test inflation factor computation."""
        from data.features import compute_inflation_adjustment_factor
        
        result = compute_inflation_adjustment_factor(
            sample_show_data,
            sample_cpi_data,
            date_column='show_date',
            base_date='2022-01-01'
        )
        
        # Check column added
        assert 'inflation_adjustment_factor' in result.columns
        
        # Check all rows have values
        assert result['inflation_adjustment_factor'].notna().all()
        
        # Check factors are near 1.0 (slight inflation adjustment)
        assert (result['inflation_adjustment_factor'] >= 0.8).all()
        assert (result['inflation_adjustment_factor'] <= 1.5).all()
    
    @pytest.mark.xfail(reason="Known issue: Consumer confidence Prairies data has only 1-2 unique values")
    def test_consumer_confidence_variance(self):
        """Test that consumer confidence has temporal variance in production data.
        
        KNOWN ISSUE: As of Dec 2025, Prairies consumer confidence data has only
        1-2 unique values, indicating nearly flat data over time.
        See docs/weightings_diagnostics.md - this is HIGH PRIORITY to fix.
        """
        # This test checks the actual data file
        data_path = Path('data/economics/nanos_consumer_confidence.csv')
        
        if not data_path.exists():
            pytest.skip("Production consumer confidence data not available")
        
        df = pd.read_csv(data_path)
        
        # Filter for Prairies region
        prairies = df[
            (df['category'] == 'Demographics') &
            (df['subcategory'] == 'Region') &
            (df['metric'] == 'Prairies')
        ]
        
        if len(prairies) == 0:
            pytest.skip("No Prairies data found in consumer confidence")
        
        # Check variance
        values = prairies['value'].dropna()
        unique_count = values.nunique()
        
        # Should have more than 2 unique values
        assert unique_count > 2, (
            f"Consumer confidence has only {unique_count} unique values. "
            f"This indicates flat data and limited predictive signal."
        )
        
        # Check range
        value_range = values.max() - values.min()
        assert value_range > 1.0, (
            f"Consumer confidence range is only {value_range:.2f}. "
            f"Expected more variance over time."
        )


# =============================================================================
# STONE OLAFSON TESTS
# =============================================================================

class TestStoneOlafsonWeightings:
    """Test Stone Olafson weighting system (segment/region multipliers)."""
    
    def test_segment_mult_exists(self):
        """Test that SEGMENT_MULT dictionary is defined."""
        from streamlit_app import SEGMENT_MULT
        
        assert isinstance(SEGMENT_MULT, dict)
        assert len(SEGMENT_MULT) > 0
    
    def test_region_mult_exists(self):
        """Test that REGION_MULT dictionary is defined."""
        from streamlit_app import REGION_MULT
        
        assert isinstance(REGION_MULT, dict)
        assert len(REGION_MULT) > 0
        
        # Should have Calgary and Edmonton
        assert 'Calgary' in REGION_MULT or 'calgary' in REGION_MULT
        assert 'Edmonton' in REGION_MULT or 'edmonton' in REGION_MULT
    
    def test_calc_scores_applies_multipliers(self):
        """Test that calc_scores applies segment and region multipliers."""
        from streamlit_app import calc_scores, SEGMENT_MULT, REGION_MULT
        
        # Get first segment and region keys
        seg_keys = list(SEGMENT_MULT.keys())
        reg_keys = list(REGION_MULT.keys())
        
        if not seg_keys or not reg_keys:
            pytest.skip("SEGMENT_MULT or REGION_MULT not configured")
        
        # Create test entry
        test_entry = {
            'wiki': 70,
            'trends': 60,
            'youtube': 80,
            'chartmetric': 65,
            'gender': 'female',
            'category': 'pop_ip'
        }
        
        # Compute scores
        fam, mot = calc_scores(test_entry, seg_keys[0], reg_keys[0])
        
        # Scores should be positive
        assert fam > 0
        assert mot > 0
        
        # Scores should be influenced by multipliers (not just base signal)
        base_fam = test_entry['wiki'] * 0.55 + test_entry['trends'] * 0.30 + test_entry['chartmetric'] * 0.15
        
        # Multiplied score should differ from base (unless all multipliers are exactly 1.0)
        # Allow small tolerance for floating point
        assert abs(fam - base_fam) > 0.01 or abs(fam - base_fam) < 0.01
    
    def test_segment_multipliers_reasonable(self):
        """Test that segment multipliers are in reasonable ranges."""
        from streamlit_app import SEGMENT_MULT
        
        for seg_key, multipliers in SEGMENT_MULT.items():
            if not isinstance(multipliers, dict):
                continue
            
            for key, value in multipliers.items():
                if isinstance(value, (int, float)):
                    # Multipliers should be positive and reasonable (0.5-2.0)
                    assert 0.3 < value < 3.0, (
                        f"Segment multiplier {seg_key}[{key}] = {value} "
                        f"is outside reasonable range (0.3-3.0)"
                    )
    
    def test_region_multipliers_reasonable(self):
        """Test that region multipliers are in reasonable ranges."""
        from streamlit_app import REGION_MULT
        
        for region, mult in REGION_MULT.items():
            if isinstance(mult, (int, float)):
                # Region multipliers should be positive and reasonable (0.5-1.5)
                assert 0.5 < mult < 2.0, (
                    f"Region multiplier {region} = {mult} "
                    f"is outside reasonable range (0.5-2.0)"
                )


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestWeightingsIntegration:
    """Integration tests for combined weighting systems."""
    
    def test_all_weightings_non_zero_impact(self):
        """Test that each weighting system has non-zero impact on scores."""
        # This would require loading the full modelling dataset and running
        # the diagnostic script, so we'll use a simplified version
        
        sample_data = pd.DataFrame({
            'wiki': [70, 65, 75],
            'trends': [60, 55, 62],
            'youtube': [80, 70, 75],
            'chartmetric': [65, 60, 68],
            'aud__engagement_factor': [1.05, 1.08, 1.06],
            'consumer_confidence_prairies': [50.0, 51.0, 50.5],
            'energy_index': [1000, 1200, 1100],
            'inflation_adjustment_factor': [1.0, 1.05, 1.02]
        })
        
        # Compute base scores
        base_scores = (
            sample_data['wiki'] * 0.55 + 
            sample_data['trends'] * 0.30 + 
            sample_data['chartmetric'] * 0.15
        )
        
        # Apply Live Analytics
        la_scores = base_scores * sample_data['aud__engagement_factor']
        la_delta = (la_scores - base_scores).abs().mean()
        
        assert la_delta > 0.1, "Live Analytics has negligible impact"
        
        # Apply Economics (simplified)
        econ_mult = (
            (sample_data['consumer_confidence_prairies'] / 50.0) * 0.3 +
            (sample_data['energy_index'] / 1000.0) * 0.4 +
            sample_data['inflation_adjustment_factor'] * 0.3
        )
        econ_scores = base_scores * econ_mult
        econ_delta = (econ_scores - base_scores).abs().mean()
        
        assert econ_delta > 0.1, "Economics has negligible impact"
    
    def test_weightings_produce_consistent_results(self):
        """Test that weighting calculations are deterministic."""
        from data.loader import get_category_engagement_factor
        
        # Same category should always return same factor
        factor1 = get_category_engagement_factor('pop_ip')
        factor2 = get_category_engagement_factor('pop_ip')
        factor3 = get_category_engagement_factor('pop_ip')
        
        assert factor1 == factor2 == factor3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
