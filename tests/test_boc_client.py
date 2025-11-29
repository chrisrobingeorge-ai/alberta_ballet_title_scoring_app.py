"""Tests for Bank of Canada Valet API client."""
import pytest
from unittest.mock import patch, MagicMock
import json

from utils.boc_client import (
    get_latest_boc_value,
    get_latest_boc_values,
    clear_cache,
    get_cache_info,
    BocApiError,
    BocDataUnavailableError,
    _parse_observation_value,
    _fetch_from_api,
    SERIES_POLICY_RATE,
    SERIES_GOV_BOND_5YR,
    SERIES_BCPI_ENERGY,
)


class TestParseObservationValue:
    """Tests for observation value parsing."""
    
    def test_parse_numeric_float_value(self):
        """Should parse numeric float values."""
        observations = [{"date": "2024-01-15", "B114039": 5.0}]
        result = _parse_observation_value(observations, "B114039")
        assert result == 5.0
    
    def test_parse_numeric_int_value(self):
        """Should parse numeric int values."""
        observations = [{"date": "2024-01-15", "B114039": 5}]
        result = _parse_observation_value(observations, "B114039")
        assert result == 5.0
    
    def test_parse_string_value(self):
        """Should parse string numeric values."""
        observations = [{"date": "2024-01-15", "B114039": "5.25"}]
        result = _parse_observation_value(observations, "B114039")
        assert result == 5.25
    
    def test_parse_empty_observations(self):
        """Should return None for empty observations."""
        result = _parse_observation_value([], "B114039")
        assert result is None
    
    def test_parse_missing_series(self):
        """Should return None when series key is missing."""
        observations = [{"date": "2024-01-15", "OTHER": 5.0}]
        result = _parse_observation_value(observations, "B114039")
        assert result is None
    
    def test_parse_null_value(self):
        """Should return None for null values."""
        observations = [{"date": "2024-01-15", "B114039": None}]
        result = _parse_observation_value(observations, "B114039")
        assert result is None
    
    def test_parse_na_string(self):
        """Should return None for 'NA' string values."""
        observations = [{"date": "2024-01-15", "B114039": "NA"}]
        result = _parse_observation_value(observations, "B114039")
        assert result is None
    
    def test_parse_uses_latest_observation(self):
        """Should use the last observation (most recent)."""
        observations = [
            {"date": "2024-01-14", "B114039": 4.5},
            {"date": "2024-01-15", "B114039": 5.0},
        ]
        result = _parse_observation_value(observations, "B114039")
        assert result == 5.0


class TestFetchFromApi:
    """Tests for API fetching with mocked HTTP."""
    
    @patch('utils.boc_client.requests.get')
    def test_successful_fetch(self, mock_get):
        """Should return value on successful API call."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "observations": [{"date": "2024-01-15", "B114039": 5.0}]
        }
        mock_get.return_value = mock_response
        
        result = _fetch_from_api("B114039")
        
        assert result == 5.0
        mock_get.assert_called_once()
    
    @patch('utils.boc_client.requests.get')
    def test_timeout_raises_error(self, mock_get):
        """Should raise BocApiError on timeout."""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout()
        
        with pytest.raises(BocApiError, match="Timeout"):
            _fetch_from_api("B114039")
    
    @patch('utils.boc_client.requests.get')
    def test_404_raises_data_unavailable(self, mock_get):
        """Should raise BocDataUnavailableError on 404."""
        import requests
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_get.return_value = mock_response
        
        with pytest.raises(BocDataUnavailableError, match="not found"):
            _fetch_from_api("INVALID_SERIES")
    
    @patch('utils.boc_client.requests.get')
    def test_empty_observations_returns_none(self, mock_get):
        """Should return None when observations are empty."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"observations": []}
        mock_get.return_value = mock_response
        
        result = _fetch_from_api("B114039")
        
        assert result is None
    
    @patch('utils.boc_client.requests.get')
    def test_invalid_json_raises_error(self, mock_get):
        """Should raise BocApiError on invalid JSON."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response
        
        with pytest.raises(BocApiError, match="Invalid JSON"):
            _fetch_from_api("B114039")


class TestGetLatestBocValue:
    """Tests for get_latest_boc_value function."""
    
    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()
    
    @patch('utils.boc_client._fetch_from_api')
    def test_returns_value_from_api(self, mock_fetch):
        """Should return value fetched from API."""
        mock_fetch.return_value = 5.0
        
        result = get_latest_boc_value("B114039")
        
        assert result == 5.0
    
    @patch('utils.boc_client._fetch_from_api')
    def test_caches_value(self, mock_fetch):
        """Should cache value after first fetch."""
        mock_fetch.return_value = 5.0
        
        # First call - fetches from API
        result1 = get_latest_boc_value("B114039")
        assert result1 == 5.0
        assert mock_fetch.call_count == 1
        
        # Second call - uses cache
        result2 = get_latest_boc_value("B114039")
        assert result2 == 5.0
        assert mock_fetch.call_count == 1  # Not called again
    
    @patch('utils.boc_client._fetch_from_api')
    def test_skips_cache_when_disabled(self, mock_fetch):
        """Should skip cache when use_cache=False."""
        mock_fetch.return_value = 5.0
        
        result1 = get_latest_boc_value("B114039", use_cache=False)
        result2 = get_latest_boc_value("B114039", use_cache=False)
        
        assert result1 == 5.0
        assert result2 == 5.0
        assert mock_fetch.call_count == 2
    
    @patch('utils.boc_client._fetch_from_api')
    def test_returns_none_on_api_error(self, mock_fetch):
        """Should return None when API fails."""
        mock_fetch.side_effect = BocApiError("Test error")
        
        result = get_latest_boc_value("B114039")
        
        assert result is None


class TestGetLatestBocValues:
    """Tests for get_latest_boc_values function."""
    
    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()
    
    @patch('utils.boc_client._fetch_from_api')
    def test_fetches_multiple_series(self, mock_fetch):
        """Should fetch values for multiple series."""
        mock_fetch.side_effect = [5.0, 3.5, 130.0]
        
        series = ["B114039", "BD.CDN.5YR.DQ.YLD", "A.ENER"]
        result = get_latest_boc_values(series)
        
        assert result["B114039"] == 5.0
        assert result["BD.CDN.5YR.DQ.YLD"] == 3.5
        assert result["A.ENER"] == 130.0
    
    @patch('utils.boc_client._fetch_from_api')
    def test_handles_partial_failure(self, mock_fetch):
        """Should handle partial failures gracefully."""
        mock_fetch.side_effect = [5.0, BocApiError("Failed"), 130.0]
        
        series = ["B114039", "BD.CDN.5YR.DQ.YLD", "A.ENER"]
        result = get_latest_boc_values(series)
        
        assert result["B114039"] == 5.0
        assert result["BD.CDN.5YR.DQ.YLD"] is None
        assert result["A.ENER"] == 130.0


class TestCacheFunctions:
    """Tests for cache management functions."""
    
    def test_clear_cache_clears_all(self):
        """clear_cache should remove all cached values."""
        # Populate cache
        from utils.boc_client import _cache
        _cache.set("TEST1", 1.0)
        _cache.set("TEST2", 2.0)
        
        clear_cache()
        
        info = get_cache_info()
        assert info["size"] == 0
    
    def test_get_cache_info_returns_stats(self):
        """get_cache_info should return cache statistics."""
        clear_cache()
        
        from utils.boc_client import _cache
        _cache.set("TEST", 1.0)
        
        info = get_cache_info()
        
        assert info["size"] == 1
        assert "TEST" in info["series"]


class TestSeriesConstants:
    """Tests that series constants are defined correctly."""
    
    def test_policy_rate_constant(self):
        """SERIES_POLICY_RATE should be the target rate series."""
        assert SERIES_POLICY_RATE == "B114039"
    
    def test_bond_yield_constants(self):
        """Bond yield constants should be defined."""
        assert "BD.CDN" in SERIES_GOV_BOND_5YR
        assert "YLD" in SERIES_GOV_BOND_5YR
    
    def test_bcpi_energy_constant(self):
        """BCPI Energy constant should be defined."""
        assert SERIES_BCPI_ENERGY == "A.ENER"


class TestIntegration:
    """Integration tests with mocked HTTP layer."""
    
    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()
    
    @patch('utils.boc_client.requests.get')
    def test_end_to_end_fetch(self, mock_get):
        """Test complete fetch flow from API to result."""
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "observations": [
                {"date": "2024-01-14", "B114039": 4.75},
                {"date": "2024-01-15", "B114039": 5.0},
            ]
        }
        mock_get.return_value = mock_response
        
        result = get_latest_boc_value(SERIES_POLICY_RATE)
        
        assert result == 5.0
        
        # Verify caching works
        cache_info = get_cache_info()
        assert cache_info["size"] == 1
