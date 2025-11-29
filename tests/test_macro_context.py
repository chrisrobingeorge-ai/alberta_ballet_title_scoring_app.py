"""Tests for macro context snapshot functions in economic_factors module."""
import pytest
from unittest.mock import patch, MagicMock


class TestIsBocGroupContextEnabled:
    """Tests for is_boc_group_context_enabled function."""
    
    @patch('utils.economic_factors.is_boc_live_enabled')
    @patch('utils.economic_factors.get_boc_config')
    @patch('utils.economic_factors.BOC_GROUPS_AVAILABLE', True)
    def test_enabled_when_all_conditions_met(self, mock_config, mock_live_enabled):
        """Should return True when config, client, and live are enabled."""
        from utils.economic_factors import is_boc_group_context_enabled
        mock_config.return_value = {"enable_boc_group_context": True}
        mock_live_enabled.return_value = True
        
        assert is_boc_group_context_enabled() == True
    
    @patch('utils.economic_factors.is_boc_live_enabled')
    @patch('utils.economic_factors.get_boc_config')
    @patch('utils.economic_factors.BOC_GROUPS_AVAILABLE', True)
    def test_disabled_when_config_false(self, mock_config, mock_live_enabled):
        """Should return False when config disables it."""
        from utils.economic_factors import is_boc_group_context_enabled
        mock_config.return_value = {"enable_boc_group_context": False}
        mock_live_enabled.return_value = True
        
        assert is_boc_group_context_enabled() == False
    
    @patch('utils.economic_factors.is_boc_live_enabled')
    @patch('utils.economic_factors.get_boc_config')
    @patch('utils.economic_factors.BOC_GROUPS_AVAILABLE', False)
    def test_disabled_when_groups_unavailable(self, mock_config, mock_live_enabled):
        """Should return False when groups client unavailable."""
        from utils.economic_factors import is_boc_group_context_enabled
        mock_config.return_value = {"enable_boc_group_context": True}
        mock_live_enabled.return_value = True
        
        assert is_boc_group_context_enabled() == False
    
    @patch('utils.economic_factors.is_boc_live_enabled')
    @patch('utils.economic_factors.get_boc_config')
    @patch('utils.economic_factors.BOC_GROUPS_AVAILABLE', True)
    def test_disabled_when_live_disabled(self, mock_config, mock_live_enabled):
        """Should return False when BoC live data is disabled."""
        from utils.economic_factors import is_boc_group_context_enabled
        mock_config.return_value = {"enable_boc_group_context": True}
        mock_live_enabled.return_value = False
        
        assert is_boc_group_context_enabled() == False


class TestGetMacroContextSnapshot:
    """Tests for get_macro_context_snapshot function."""
    
    @patch('utils.economic_factors.is_boc_group_context_enabled')
    def test_returns_unavailable_when_disabled(self, mock_enabled):
        """Should indicate unavailable when group context is disabled."""
        from utils.economic_factors import get_macro_context_snapshot
        mock_enabled.return_value = False
        
        result = get_macro_context_snapshot()
        
        assert result["available"] == False
        assert "message" in result
    
    @patch('utils.economic_factors.get_latest_group_observation')
    @patch('utils.economic_factors.get_group_metadata')
    @patch('utils.economic_factors.get_boc_config')
    @patch('utils.economic_factors.is_boc_group_context_enabled')
    def test_returns_snapshot_when_data_available(
        self, mock_enabled, mock_config, mock_metadata, mock_group_obs
    ):
        """Should return snapshot with available data."""
        from utils.economic_factors import get_macro_context_snapshot
        
        mock_enabled.return_value = True
        mock_config.return_value = {
            "boc_groups": {
                "bcpi_monthly": "BCPI_MONTHLY",
            },
            "macro_context_groups": ["bcpi_monthly"],
            "group_display": {
                "highlight_series": {
                    "M.BCPI": "BCPI Total (Monthly)",
                    "M.ENER": "BCPI Energy (Monthly)",
                },
                "formats": {
                    "M.BCPI": "{:.2f}",
                    "M.ENER": "{:.2f}",
                },
            },
        }
        mock_group_obs.return_value = {
            "M.BCPI": 145.67,
            "M.ENER": 180.23,
            "M.BCNE": 120.45,
        }
        mock_metadata.return_value = {
            "label": "Monthly BCPI",
            "description": "Monthly Bank of Canada commodity price index.",
        }
        
        result = get_macro_context_snapshot()
        
        assert result["available"] == True
        assert "bcpi_monthly" in result["groups"]
        assert result["groups"]["bcpi_monthly"]["series_count"] == 3
        assert len(result["highlighted"]) >= 2  # M.BCPI and M.ENER
    
    @patch('utils.economic_factors.get_latest_group_observation')
    @patch('utils.economic_factors.get_boc_config')
    @patch('utils.economic_factors.is_boc_group_context_enabled')
    def test_handles_group_fetch_failure(self, mock_enabled, mock_config, mock_group_obs):
        """Should handle failures gracefully without impacting scoring."""
        from utils.economic_factors import get_macro_context_snapshot
        
        mock_enabled.return_value = True
        mock_config.return_value = {
            "boc_groups": {
                "bcpi_monthly": "BCPI_MONTHLY",
            },
            "macro_context_groups": ["bcpi_monthly"],
            "group_display": {},
        }
        mock_group_obs.side_effect = Exception("Network error")
        
        result = get_macro_context_snapshot()
        
        # Should return partial result, not raise exception
        assert result["available"] == False
        assert len(result["errors"]) > 0
    
    @patch('utils.economic_factors.get_boc_config')
    @patch('utils.economic_factors.is_boc_group_context_enabled')
    def test_returns_unavailable_when_no_groups_configured(self, mock_enabled, mock_config):
        """Should indicate unavailable when no groups configured."""
        from utils.economic_factors import get_macro_context_snapshot
        
        mock_enabled.return_value = True
        mock_config.return_value = {
            "boc_groups": {},
            "macro_context_groups": [],
            "group_display": {},
        }
        
        result = get_macro_context_snapshot()
        
        assert result["available"] == False
        assert "message" in result
    
    @patch('utils.economic_factors.get_latest_group_observation')
    @patch('utils.economic_factors.get_group_metadata')
    @patch('utils.economic_factors.get_boc_config')
    @patch('utils.economic_factors.is_boc_group_context_enabled')
    def test_includes_timestamp(self, mock_enabled, mock_config, mock_metadata, mock_group_obs):
        """Should include fetched_at timestamp."""
        from utils.economic_factors import get_macro_context_snapshot
        
        mock_enabled.return_value = True
        mock_config.return_value = {
            "boc_groups": {"test": "TEST_GROUP"},
            "macro_context_groups": ["test"],
            "group_display": {},
        }
        mock_group_obs.return_value = {"A": 1.0}
        mock_metadata.return_value = None
        
        result = get_macro_context_snapshot()
        
        assert result["fetched_at"] is not None


class TestGetMacroContextDisplay:
    """Tests for get_macro_context_display function."""
    
    @patch('utils.economic_factors.get_macro_context_snapshot')
    def test_returns_unavailable_when_snapshot_unavailable(self, mock_snapshot):
        """Should return unavailable when snapshot unavailable."""
        from utils.economic_factors import get_macro_context_display
        
        mock_snapshot.return_value = {
            "available": False,
            "message": "Test message",
        }
        
        result = get_macro_context_display()
        
        assert result["available"] == False
        assert result["message"] == "Test message"
    
    @patch('utils.economic_factors.get_macro_context_snapshot')
    def test_formats_sections_for_display(self, mock_snapshot):
        """Should format snapshot into display sections."""
        from utils.economic_factors import get_macro_context_display
        
        mock_snapshot.return_value = {
            "available": True,
            "fetched_at": "2024-01-15T12:00:00+00:00",
            "groups": {
                "bcpi_monthly": {
                    "group_id": "BCPI_MONTHLY",
                    "label": "Monthly BCPI",
                    "description": "Monthly commodity index",
                    "series": {"M.BCPI": 145.67},
                    "series_count": 1,
                },
            },
            "highlighted": [
                {
                    "series_id": "M.BCPI",
                    "label": "BCPI Total",
                    "value": 145.67,
                    "formatted_value": "145.67",
                    "group_key": "bcpi_monthly",
                    "group_id": "BCPI_MONTHLY",
                }
            ],
        }
        
        result = get_macro_context_display()
        
        assert result["available"] == True
        assert len(result["sections"]) == 1
        assert result["sections"][0]["group_key"] == "bcpi_monthly"
        assert len(result["sections"][0]["items"]) == 1


class TestIntegrationMacroContext:
    """Integration tests for macro context functions."""
    
    def test_get_macro_context_snapshot_runs_without_error(self):
        """Test that get_macro_context_snapshot runs without error."""
        from utils.economic_factors import get_macro_context_snapshot
        
        # This will run with actual config, testing the integration
        result = get_macro_context_snapshot()
        
        # Should return a valid structure regardless of data availability
        assert "available" in result
        assert "groups" in result
        assert "highlighted" in result
    
    def test_get_macro_context_display_runs_without_error(self):
        """Test that get_macro_context_display runs without error."""
        from utils.economic_factors import get_macro_context_display
        
        result = get_macro_context_display()
        
        # Should return a valid structure
        assert "available" in result
        assert "sections" in result
