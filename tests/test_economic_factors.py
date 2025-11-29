"""Tests for economic factors module with BoC integration."""
import pytest
from unittest.mock import patch, MagicMock
from datetime import date

from utils.economic_factors import (
    load_boc_config,
    get_boc_config,
    is_boc_live_enabled,
    fetch_boc_indicators,
    compute_boc_economic_sentiment,
    get_boc_indicator_display,
    get_combined_economic_sentiment,
    _compute_z_score,
)


class TestComputeZScore:
    """Tests for z-score computation."""
    
    def test_positive_direction_above_baseline(self):
        """Positive direction: above baseline should give positive z."""
        z = _compute_z_score(value=130, baseline=100, std=30, direction="positive")
        assert z == pytest.approx(1.0, abs=0.01)
    
    def test_positive_direction_below_baseline(self):
        """Positive direction: below baseline should give negative z."""
        z = _compute_z_score(value=70, baseline=100, std=30, direction="positive")
        assert z == pytest.approx(-1.0, abs=0.01)
    
    def test_negative_direction_above_baseline(self):
        """Negative direction: above baseline should give negative z."""
        z = _compute_z_score(value=5, baseline=3, std=2, direction="negative")
        assert z == pytest.approx(-1.0, abs=0.01)
    
    def test_negative_direction_below_baseline(self):
        """Negative direction: below baseline should give positive z."""
        z = _compute_z_score(value=1, baseline=3, std=2, direction="negative")
        assert z == pytest.approx(1.0, abs=0.01)
    
    def test_at_baseline_returns_zero(self):
        """Value at baseline should give z=0."""
        z = _compute_z_score(value=100, baseline=100, std=30, direction="positive")
        assert z == pytest.approx(0.0, abs=0.01)
    
    def test_zero_std_uses_default(self):
        """Zero std should not cause division error."""
        z = _compute_z_score(value=5, baseline=3, std=0, direction="positive")
        assert z == pytest.approx(2.0, abs=0.01)


class TestLoadBocConfig:
    """Tests for config loading."""
    
    def test_loads_config_from_file(self):
        """Should load config from YAML file."""
        config = load_boc_config()
        assert isinstance(config, dict)
    
    def test_config_has_required_keys(self):
        """Config should have required structure."""
        config = load_boc_config()
        # Either loaded from file or defaults
        assert "use_boc_live_data" in config or "fallback_mode" in config or "sentiment_calculation" in config
    
    def test_missing_file_uses_defaults(self):
        """Should use defaults when file is missing."""
        config = load_boc_config(config_path="/nonexistent/path.yaml")
        assert isinstance(config, dict)
        assert "fallback_mode" in config


class TestIsBocLiveEnabled:
    """Tests for BoC enable check."""
    
    @patch('utils.economic_factors.get_boc_config')
    @patch('utils.economic_factors.BOC_CLIENT_AVAILABLE', True)
    def test_enabled_when_config_true(self, mock_config):
        """Should return True when config enables it and client available."""
        mock_config.return_value = {"use_boc_live_data": True}
        assert is_boc_live_enabled() == True
    
    @patch('utils.economic_factors.get_boc_config')
    @patch('utils.economic_factors.BOC_CLIENT_AVAILABLE', True)
    def test_disabled_when_config_false(self, mock_config):
        """Should return False when config disables it."""
        mock_config.return_value = {"use_boc_live_data": False}
        assert is_boc_live_enabled() == False
    
    @patch('utils.economic_factors.get_boc_config')
    @patch('utils.economic_factors.BOC_CLIENT_AVAILABLE', False)
    def test_disabled_when_client_unavailable(self, mock_config):
        """Should return False when client not available."""
        mock_config.return_value = {"use_boc_live_data": True}
        assert is_boc_live_enabled() == False


class TestFetchBocIndicators:
    """Tests for indicator fetching."""
    
    @patch('utils.economic_factors.is_boc_live_enabled')
    @patch('utils.economic_factors.BOC_CLIENT_AVAILABLE', False)
    def test_returns_empty_when_client_unavailable(self, mock_enabled):
        """Should return empty dict when client unavailable."""
        mock_enabled.return_value = False
        values, success = fetch_boc_indicators()
        assert values == {}
        assert success == False
    
    @patch('utils.economic_factors.get_latest_boc_values')
    @patch('utils.economic_factors.get_boc_config')
    @patch('utils.economic_factors.BOC_CLIENT_AVAILABLE', True)
    def test_fetches_configured_series(self, mock_config, mock_fetch):
        """Should fetch all configured series."""
        mock_config.return_value = {
            "boc_series": {
                "policy_rate": {"id": "B114039"},
                "bcpi_energy": {"id": "A.ENER"},
            }
        }
        mock_fetch.return_value = {"B114039": 5.0, "A.ENER": 130.0}
        
        values, success = fetch_boc_indicators()
        
        assert success == True
        assert values["policy_rate"] == 5.0
        assert values["bcpi_energy"] == 130.0


class TestComputeBocEconomicSentiment:
    """Tests for sentiment computation."""
    
    @patch('utils.economic_factors.is_boc_live_enabled')
    def test_returns_fallback_when_disabled(self, mock_enabled):
        """Should return fallback when BoC disabled."""
        mock_enabled.return_value = False
        
        factor, details = compute_boc_economic_sentiment()
        
        assert isinstance(factor, float)
        assert 0.5 <= factor <= 1.5
        assert details.get("fallback_used") == True
    
    @patch('utils.economic_factors.fetch_boc_indicators')
    @patch('utils.economic_factors.is_boc_live_enabled')
    def test_returns_fallback_when_fetch_fails(self, mock_enabled, mock_fetch):
        """Should return fallback when fetch fails."""
        mock_enabled.return_value = True
        mock_fetch.return_value = ({}, False)
        
        factor, details = compute_boc_economic_sentiment()
        
        assert isinstance(factor, float)
        assert details.get("fallback_used") == True
    
    @patch('utils.economic_factors.fetch_boc_indicators')
    @patch('utils.economic_factors.get_boc_config')
    @patch('utils.economic_factors.is_boc_live_enabled')
    def test_computes_sentiment_from_indicators(self, mock_enabled, mock_config, mock_fetch):
        """Should compute sentiment from fetched indicators."""
        mock_enabled.return_value = True
        mock_config.return_value = {
            "boc_series": {
                "bcpi_energy": {
                    "id": "A.ENER",
                    "baseline": 100.0,
                    "weight": 1.0,
                    "direction": "positive",
                }
            },
            "sentiment_calculation": {
                "min_factor": 0.85,
                "max_factor": 1.15,
                "neutral": 1.0,
                "sensitivity": 0.10,
                "historical_stats": {
                    "bcpi_energy": {"mean": 100.0, "std": 30.0}
                }
            }
        }
        # Energy index above baseline = favorable
        mock_fetch.return_value = ({"bcpi_energy": 130.0}, True)
        
        factor, details = compute_boc_economic_sentiment()
        
        assert factor > 1.0  # Favorable
        assert details["boc_available"] == True
        assert "bcpi_energy" in details["indicators"]
    
    @patch('utils.economic_factors.fetch_boc_indicators')
    @patch('utils.economic_factors.get_boc_config')
    @patch('utils.economic_factors.is_boc_live_enabled')
    def test_clamps_to_bounds(self, mock_enabled, mock_config, mock_fetch):
        """Should clamp factor to min/max bounds."""
        mock_enabled.return_value = True
        mock_config.return_value = {
            "boc_series": {
                "test": {
                    "id": "TEST",
                    "baseline": 0.0,
                    "weight": 1.0,
                    "direction": "positive",
                }
            },
            "sentiment_calculation": {
                "min_factor": 0.85,
                "max_factor": 1.15,
                "neutral": 1.0,
                "sensitivity": 0.50,  # Very high sensitivity
                "historical_stats": {
                    "test": {"mean": 0.0, "std": 1.0}
                }
            }
        }
        # Extreme value that would give huge z-score
        mock_fetch.return_value = ({"test": 100.0}, True)
        
        factor, details = compute_boc_economic_sentiment()
        
        assert factor <= 1.15  # Should be clamped


class TestGetBocIndicatorDisplay:
    """Tests for UI display helper."""
    
    @patch('utils.economic_factors.is_boc_live_enabled')
    def test_returns_unavailable_when_disabled(self, mock_enabled):
        """Should indicate unavailable when disabled."""
        mock_enabled.return_value = False
        
        result = get_boc_indicator_display()
        
        assert result["available"] == False
    
    @patch('utils.economic_factors.compute_boc_economic_sentiment')
    @patch('utils.economic_factors.is_boc_live_enabled')
    def test_returns_formatted_indicators(self, mock_enabled, mock_sentiment):
        """Should return formatted indicator data."""
        mock_enabled.return_value = True
        mock_sentiment.return_value = (1.05, {
            "boc_available": True,
            "source": "boc_live",
            "indicators": {
                "policy_rate": {"value": 5.0, "z_score": 0.5},
            }
        })
        
        result = get_boc_indicator_display()
        
        assert result["available"] == True
        assert result["sentiment_factor"] == 1.05


class TestGetCombinedEconomicSentiment:
    """Tests for combined sentiment calculation."""
    
    @patch('utils.economic_factors.compute_boc_economic_sentiment')
    @patch('utils.economic_factors.is_boc_live_enabled')
    def test_uses_boc_when_available(self, mock_enabled, mock_boc):
        """Should include BoC factor when available."""
        mock_enabled.return_value = True
        mock_boc.return_value = (1.05, {"boc_available": True})
        
        factor, details = get_combined_economic_sentiment(boc_weight=0.5)
        
        assert details["boc_factor"] == 1.05
    
    @patch('utils.economic_factors.is_boc_live_enabled')
    def test_uses_historical_when_boc_disabled(self, mock_enabled):
        """Should use only historical when BoC disabled."""
        mock_enabled.return_value = False
        
        factor, details = get_combined_economic_sentiment()
        
        assert details["boc_factor"] is None
        assert isinstance(factor, float)


class TestIntegration:
    """Integration tests."""
    
    def test_full_sentiment_computation_flow(self):
        """Test complete flow from config to sentiment."""
        # This tests the actual integration without mocks
        # Will use fallback if BoC disabled/unavailable
        factor, details = compute_boc_economic_sentiment()
        
        assert isinstance(factor, float)
        assert 0.5 <= factor <= 1.5
        assert "source" in details
    
    def test_display_helper_returns_valid_structure(self):
        """Test display helper returns valid structure."""
        result = get_boc_indicator_display()
        
        assert "available" in result
        if result["available"]:
            assert "indicators" in result
            assert "sentiment_factor" in result
