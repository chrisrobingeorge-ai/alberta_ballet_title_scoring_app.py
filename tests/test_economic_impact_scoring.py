"""Tests for economic impact scoring functionality."""
import pytest
import pandas as pd
import numpy as np

from ml.scoring import (
    load_economic_baselines,
    compute_economic_impact_score,
    compute_city_economic_summary,
    score_with_economic_impact,
    DEFAULT_ECONOMIC_BASELINES,
)

# Tolerance for neutral score assertion
# Neutral inputs should produce scores within this range of zero
NEUTRAL_SCORE_TOLERANCE = 15


class TestEconomicBaselinesConfig:
    """Tests for economic baselines configuration."""
    
    def test_load_economic_baselines_returns_dict(self):
        """load_economic_baselines should return a dictionary."""
        result = load_economic_baselines()
        assert isinstance(result, dict)
    
    def test_baselines_have_required_keys(self):
        """Baselines should have consumer_confidence, energy_index, cpi_base."""
        result = load_economic_baselines()
        
        expected_keys = ['consumer_confidence', 'energy_index', 'cpi_base']
        for key in expected_keys:
            assert key in result, f"Missing baseline key: {key}"
    
    def test_consumer_confidence_has_required_fields(self):
        """Consumer confidence baseline should have baseline, thresholds."""
        result = load_economic_baselines()
        
        cc = result.get('consumer_confidence', {})
        assert 'baseline' in cc, "Missing consumer_confidence baseline"
        assert 'good_threshold' in cc, "Missing good_threshold"
        assert 'poor_threshold' in cc, "Missing poor_threshold"
    
    def test_default_baselines_available(self):
        """DEFAULT_ECONOMIC_BASELINES should be importable."""
        assert DEFAULT_ECONOMIC_BASELINES is not None
        assert isinstance(DEFAULT_ECONOMIC_BASELINES, dict)


class TestEconomicImpactScoreComputation:
    """Tests for economic impact score computation."""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature data with economic indicators."""
        return pd.DataFrame({
            'title': ['Show A', 'Show B', 'Show C'],
            'city': ['Calgary', 'Edmonton', 'Calgary'],
            'consumer_confidence_headline': [58.0, 48.0, 55.0],
            'consumer_confidence_prairies': [60.0, 50.0, 55.0],
            'energy_index': [1200.0, 600.0, 800.0],
            'inflation_adjustment_factor': [1.05, 1.15, 1.00]
        })
    
    @pytest.fixture
    def neutral_features(self):
        """Create features with neutral/baseline values."""
        return pd.DataFrame({
            'title': ['Neutral Show'],
            'city': ['Calgary'],
            'consumer_confidence_headline': [55.0],  # baseline
            'consumer_confidence_prairies': [55.0],  # baseline
            'energy_index': [800.0],  # baseline
            'inflation_adjustment_factor': [1.0]  # baseline
        })
    
    def test_compute_economic_impact_adds_score_column(self, sample_features):
        """Should add economic_impact_score column."""
        result = compute_economic_impact_score(sample_features)
        
        assert 'economic_impact_score' in result.columns
        assert len(result) == len(sample_features)
    
    def test_includes_component_scores_when_requested(self, sample_features):
        """Should include component scores when include_components=True."""
        result = compute_economic_impact_score(
            sample_features, 
            include_components=True
        )
        
        assert 'econ_impact_consumer_confidence' in result.columns
        assert 'econ_impact_energy' in result.columns
        assert 'econ_impact_inflation' in result.columns
    
    def test_excludes_component_scores_when_not_requested(self, sample_features):
        """Should not include component scores when include_components=False."""
        result = compute_economic_impact_score(
            sample_features,
            include_components=False
        )
        
        assert 'econ_impact_consumer_confidence' not in result.columns
        assert 'econ_impact_energy' not in result.columns
        assert 'econ_impact_inflation' not in result.columns
    
    def test_neutral_inputs_produce_near_zero_score(self, neutral_features):
        """Baseline values should produce score near zero."""
        result = compute_economic_impact_score(neutral_features)
        
        score = result['economic_impact_score'].iloc[0]
        
        # Should be close to zero (within reasonable tolerance)
        assert abs(score) < NEUTRAL_SCORE_TOLERANCE, f"Neutral inputs should give near-zero score, got {score}"
    
    def test_positive_indicators_produce_positive_score(self):
        """Strong economic indicators should produce positive score."""
        strong_economy = pd.DataFrame({
            'title': ['Good Times Show'],
            'consumer_confidence_headline': [65.0],  # High confidence
            'consumer_confidence_prairies': [65.0],  # High regional confidence
            'energy_index': [1500.0],  # High energy prices
            'inflation_adjustment_factor': [0.98]  # Low inflation
        })
        
        result = compute_economic_impact_score(strong_economy)
        score = result['economic_impact_score'].iloc[0]
        
        assert score > 0, f"Strong economy should give positive score, got {score}"
    
    def test_negative_indicators_produce_negative_score(self):
        """Weak economic indicators should produce negative score."""
        weak_economy = pd.DataFrame({
            'title': ['Hard Times Show'],
            'consumer_confidence_headline': [40.0],  # Low confidence
            'consumer_confidence_prairies': [40.0],  # Low regional confidence
            'energy_index': [400.0],  # Low energy prices
            'inflation_adjustment_factor': [1.20]  # High inflation
        })
        
        result = compute_economic_impact_score(weak_economy)
        score = result['economic_impact_score'].iloc[0]
        
        assert score < 0, f"Weak economy should give negative score, got {score}"
    
    def test_score_is_bounded(self, sample_features):
        """Economic impact score should be bounded to reasonable range."""
        result = compute_economic_impact_score(sample_features)
        
        for score in result['economic_impact_score']:
            assert -100 <= score <= 100, f"Score out of bounds: {score}"
    
    def test_handles_missing_columns_gracefully(self):
        """Should handle missing economic columns without crashing."""
        incomplete_data = pd.DataFrame({
            'title': ['Show A'],
            'city': ['Calgary']
            # No economic columns
        })
        
        result = compute_economic_impact_score(incomplete_data)
        
        assert 'economic_impact_score' in result.columns
        # Score should be 0 (neutral) when no data available
        assert result['economic_impact_score'].iloc[0] == 0.0


class TestCityEconomicSummary:
    """Tests for city-level economic summary computation."""
    
    @pytest.fixture
    def multi_city_features(self):
        """Create features with multiple shows per city."""
        return pd.DataFrame({
            'title': ['Show 1', 'Show 2', 'Show 3', 'Show 4'],
            'city': ['Calgary', 'Calgary', 'Edmonton', 'Edmonton'],
            'consumer_confidence_prairies': [58.0, 60.0, 52.0, 50.0],
            'energy_index': [1000.0, 1100.0, 900.0, 950.0],
            'inflation_adjustment_factor': [1.02, 1.03, 1.05, 1.06]
        })
    
    def test_summary_aggregates_by_city(self, multi_city_features):
        """Should create summary with one row per city."""
        scored = compute_economic_impact_score(multi_city_features)
        summary = compute_city_economic_summary(scored)
        
        assert len(summary) == 2  # Calgary and Edmonton
    
    def test_summary_includes_mean_score(self, multi_city_features):
        """Summary should include mean economic impact score."""
        scored = compute_economic_impact_score(multi_city_features)
        summary = compute_city_economic_summary(scored)
        
        mean_cols = [c for c in summary.columns if 'mean' in c]
        assert len(mean_cols) > 0
    
    def test_summary_includes_min_max(self, multi_city_features):
        """Summary should include min and max scores."""
        scored = compute_economic_impact_score(multi_city_features)
        summary = compute_city_economic_summary(scored)
        
        min_cols = [c for c in summary.columns if 'min' in c]
        max_cols = [c for c in summary.columns if 'max' in c]
        assert len(min_cols) > 0
        assert len(max_cols) > 0


class TestScoringWithEconomicImpact:
    """Tests for combined scoring with economic impact."""
    
    @pytest.fixture
    def scorable_features(self):
        """Create features that could be used for model scoring."""
        return pd.DataFrame({
            'title': ['Show A', 'Show B'],
            'city': ['Calgary', 'Edmonton'],
            'consumer_confidence_prairies': [55.0, 52.0],
            'energy_index': [850.0, 750.0],
            'inflation_adjustment_factor': [1.02, 1.05],
            # Additional features a model might use
            'venue_capacity': [2500, 2800],
            'days_out': [60, 45],
            'is_premiere': [1, 0]
        })
    
    def test_score_with_economic_returns_predictions(self, scorable_features):
        """Should include forecast column in output."""
        result = score_with_economic_impact(
            scorable_features,
            model=None,  # Will try to load or skip model
            include_economic=True
        )
        
        assert 'forecast_single_tickets' in result.columns
    
    def test_score_with_economic_adds_impact_score(self, scorable_features):
        """Should include economic impact score."""
        result = score_with_economic_impact(
            scorable_features,
            model=None,
            include_economic=True
        )
        
        assert 'economic_impact_score' in result.columns
    
    def test_can_skip_economic_scoring(self, scorable_features):
        """Should skip economic scoring when include_economic=False."""
        result = score_with_economic_impact(
            scorable_features,
            model=None,
            include_economic=False
        )
        
        assert 'economic_impact_score' not in result.columns


class TestScoreInterpretation:
    """Tests for score value interpretation."""
    
    def test_score_direction_makes_sense(self):
        """Verify that score direction matches economic intuition."""
        # Better confidence = better score
        high_conf = pd.DataFrame({
            'consumer_confidence_prairies': [70.0],
            'energy_index': [800.0],
            'inflation_adjustment_factor': [1.0]
        })
        low_conf = pd.DataFrame({
            'consumer_confidence_prairies': [40.0],
            'energy_index': [800.0],
            'inflation_adjustment_factor': [1.0]
        })
        
        high_result = compute_economic_impact_score(high_conf)
        low_result = compute_economic_impact_score(low_conf)
        
        assert high_result['economic_impact_score'].iloc[0] > \
               low_result['economic_impact_score'].iloc[0], \
            "Higher confidence should give higher score"
    
    def test_higher_inflation_gives_lower_score(self):
        """Higher inflation should produce lower (more negative) score."""
        low_inflation = pd.DataFrame({
            'consumer_confidence_prairies': [55.0],
            'energy_index': [800.0],
            'inflation_adjustment_factor': [0.98]
        })
        high_inflation = pd.DataFrame({
            'consumer_confidence_prairies': [55.0],
            'energy_index': [800.0],
            'inflation_adjustment_factor': [1.15]
        })
        
        low_result = compute_economic_impact_score(low_inflation)
        high_result = compute_economic_impact_score(high_inflation)
        
        assert low_result['economic_impact_score'].iloc[0] > \
               high_result['economic_impact_score'].iloc[0], \
            "Lower inflation should give higher score"
