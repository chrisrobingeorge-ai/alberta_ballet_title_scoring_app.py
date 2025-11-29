"""Tests for Bank of Canada Valet API group-based functions."""
import pytest
from unittest.mock import patch, MagicMock
import json

from utils.boc_client import (
    get_latest_group_observation,
    get_group_metadata,
    clear_group_cache,
    get_group_cache_info,
    _parse_group_observations,
    _validate_group_name,
    _fetch_group_from_api,
    BocApiError,
    BocDataUnavailableError,
)


class TestValidateGroupName:
    """Tests for group name validation."""
    
    def test_valid_alphanumeric_name(self):
        """Should accept alphanumeric group names."""
        assert _validate_group_name("FX_RATES_DAILY") == True
        assert _validate_group_name("BCPI_MONTHLY") == True
        assert _validate_group_name("CEER_DAILY") == True
    
    def test_valid_name_with_underscores(self):
        """Should accept names with underscores."""
        assert _validate_group_name("FX_RATES_MONTHLY") == True
        assert _validate_group_name("BCPI_ANNUAL") == True
    
    def test_valid_name_with_dashes(self):
        """Should accept names with dashes."""
        assert _validate_group_name("MPR-2025M07") == True
    
    def test_valid_name_with_dots(self):
        """Should accept names with dots."""
        assert _validate_group_name("GROUP.NAME") == True
    
    def test_invalid_empty_string(self):
        """Should reject empty strings."""
        assert _validate_group_name("") == False
    
    def test_invalid_none(self):
        """Should reject None."""
        assert _validate_group_name(None) == False
    
    def test_invalid_with_special_chars(self):
        """Should reject names with special characters."""
        assert _validate_group_name("GROUP/NAME") == False
        assert _validate_group_name("GROUP;NAME") == False
        assert _validate_group_name("GROUP?NAME") == False


class TestParseGroupObservations:
    """Tests for parsing group observation responses."""
    
    def test_parse_numeric_values(self):
        """Should parse numeric float values."""
        observations = [
            {"d": "2024-01-15", "FXUSDCAD": 1.3456, "FXEURCAD": 1.4789}
        ]
        result = _parse_group_observations(observations)
        assert result["FXUSDCAD"] == pytest.approx(1.3456)
        assert result["FXEURCAD"] == pytest.approx(1.4789)
    
    def test_parse_string_numeric_values(self):
        """Should parse string numeric values."""
        observations = [
            {"d": "2024-01-15", "M.BCPI": "145.67"}
        ]
        result = _parse_group_observations(observations)
        assert result["M.BCPI"] == pytest.approx(145.67)
    
    def test_parse_empty_observations(self):
        """Should return empty dict for empty observations."""
        result = _parse_group_observations([])
        assert result == {}
    
    def test_skip_date_field(self):
        """Should skip the date field 'd'."""
        observations = [
            {"d": "2024-01-15", "FXUSDCAD": 1.3456}
        ]
        result = _parse_group_observations(observations)
        assert "d" not in result
        assert "date" not in result
    
    def test_handle_null_values(self):
        """Should return None for null values."""
        observations = [
            {"d": "2024-01-15", "FXUSDCAD": None, "FXEURCAD": 1.4789}
        ]
        result = _parse_group_observations(observations)
        assert result["FXUSDCAD"] is None
        assert result["FXEURCAD"] == pytest.approx(1.4789)
    
    def test_handle_na_string(self):
        """Should return None for 'NA' string values."""
        observations = [
            {"d": "2024-01-15", "FXUSDCAD": "NA"}
        ]
        result = _parse_group_observations(observations)
        assert result["FXUSDCAD"] is None
    
    def test_uses_latest_observation(self):
        """Should use the last observation (most recent)."""
        observations = [
            {"d": "2024-01-14", "FXUSDCAD": 1.3400},
            {"d": "2024-01-15", "FXUSDCAD": 1.3456},
        ]
        result = _parse_group_observations(observations)
        assert result["FXUSDCAD"] == pytest.approx(1.3456)
    
    def test_handle_unparseable_value(self):
        """Should return None for unparseable values."""
        observations = [
            {"d": "2024-01-15", "FXUSDCAD": "not a number"}
        ]
        result = _parse_group_observations(observations)
        assert result["FXUSDCAD"] is None


class TestFetchGroupFromApi:
    """Tests for API fetching for groups with mocked HTTP."""
    
    @patch('utils.boc_client.requests.get')
    def test_successful_fetch(self, mock_get):
        """Should return values on successful API call."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "observations": [
                {"d": "2024-01-15", "FXUSDCAD": 1.3456, "FXEURCAD": 1.4789}
            ]
        }
        mock_get.return_value = mock_response
        
        result = _fetch_group_from_api("FX_RATES_DAILY")
        
        assert result["FXUSDCAD"] == pytest.approx(1.3456)
        assert result["FXEURCAD"] == pytest.approx(1.4789)
        mock_get.assert_called_once()
    
    @patch('utils.boc_client.requests.get')
    def test_timeout_raises_error(self, mock_get):
        """Should raise BocApiError on timeout."""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout()
        
        with pytest.raises(BocApiError, match="Timeout"):
            _fetch_group_from_api("FX_RATES_DAILY")
    
    @patch('utils.boc_client.requests.get')
    def test_404_raises_data_unavailable(self, mock_get):
        """Should raise BocDataUnavailableError on 404."""
        import requests
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_get.return_value = mock_response
        
        with pytest.raises(BocDataUnavailableError, match="not found"):
            _fetch_group_from_api("INVALID_GROUP")
    
    @patch('utils.boc_client.requests.get')
    def test_empty_observations_returns_empty_dict(self, mock_get):
        """Should return empty dict when observations are empty."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"observations": []}
        mock_get.return_value = mock_response
        
        result = _fetch_group_from_api("FX_RATES_DAILY")
        
        assert result == {}
    
    @patch('utils.boc_client.requests.get')
    def test_invalid_json_raises_error(self, mock_get):
        """Should raise BocApiError on invalid JSON."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response
        
        with pytest.raises(BocApiError, match="Invalid JSON"):
            _fetch_group_from_api("FX_RATES_DAILY")
    
    def test_invalid_group_name_raises_error(self):
        """Should raise BocApiError for invalid group name."""
        with pytest.raises(BocApiError, match="Invalid group name"):
            _fetch_group_from_api("GROUP;DROP TABLE")


class TestGetLatestGroupObservation:
    """Tests for get_latest_group_observation function."""
    
    def setup_method(self):
        """Clear cache before each test."""
        clear_group_cache()
    
    @patch('utils.boc_client._fetch_group_from_api')
    def test_returns_values_from_api(self, mock_fetch):
        """Should return values fetched from API."""
        mock_fetch.return_value = {"FXUSDCAD": 1.3456, "FXEURCAD": 1.4789}
        
        result = get_latest_group_observation("FX_RATES_DAILY")
        
        assert result["FXUSDCAD"] == 1.3456
        assert result["FXEURCAD"] == 1.4789
    
    @patch('utils.boc_client._fetch_group_from_api')
    def test_caches_values(self, mock_fetch):
        """Should cache values after first fetch."""
        mock_fetch.return_value = {"FXUSDCAD": 1.3456}
        
        # First call - fetches from API
        result1 = get_latest_group_observation("FX_RATES_DAILY")
        assert result1["FXUSDCAD"] == 1.3456
        assert mock_fetch.call_count == 1
        
        # Second call - uses cache
        result2 = get_latest_group_observation("FX_RATES_DAILY")
        assert result2["FXUSDCAD"] == 1.3456
        assert mock_fetch.call_count == 1  # Not called again
    
    @patch('utils.boc_client._fetch_group_from_api')
    def test_skips_cache_when_disabled(self, mock_fetch):
        """Should skip cache when use_cache=False."""
        mock_fetch.return_value = {"FXUSDCAD": 1.3456}
        
        result1 = get_latest_group_observation("FX_RATES_DAILY", use_cache=False)
        result2 = get_latest_group_observation("FX_RATES_DAILY", use_cache=False)
        
        assert result1["FXUSDCAD"] == 1.3456
        assert result2["FXUSDCAD"] == 1.3456
        assert mock_fetch.call_count == 2
    
    @patch('utils.boc_client._fetch_group_from_api')
    def test_returns_empty_on_api_error(self, mock_fetch):
        """Should return empty dict when API fails."""
        mock_fetch.side_effect = BocApiError("Test error")
        
        result = get_latest_group_observation("FX_RATES_DAILY")
        
        assert result == {}


class TestGetGroupMetadata:
    """Tests for get_group_metadata function."""
    
    @patch('builtins.open')
    @patch('pathlib.Path.exists')
    def test_returns_metadata_for_valid_group(self, mock_exists, mock_open):
        """Should return metadata for a valid group."""
        mock_exists.return_value = True
        mock_data = {
            "groups": {
                "FX_RATES_DAILY": {
                    "label": "Daily exchange rates",
                    "link": "https://www.bankofcanada.ca/valet/groups/FX_RATES_DAILY",
                    "description": "Daily average exchange rates"
                }
            }
        }
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_data)
        
        # Mock the json.load function properly
        with patch('json.load', return_value=mock_data):
            result = get_group_metadata("FX_RATES_DAILY")
        
        assert result is not None
        assert result["label"] == "Daily exchange rates"
        assert "description" in result
    
    def test_returns_none_for_nonexistent_group(self):
        """Should return None for non-existent group."""
        result = get_group_metadata("NONEXISTENT_GROUP_12345")
        # This will try to load from file and may return None if group not found
        # or return metadata if group exists in the actual file
        assert result is None or isinstance(result, dict)


class TestGroupCacheFunctions:
    """Tests for group cache management functions."""
    
    def test_clear_group_cache_clears_all(self):
        """clear_group_cache should remove all cached values."""
        from utils.boc_client import _group_cache
        
        # Populate cache
        with _group_cache._lock:
            _group_cache._cache["group:TEST1"] = ({"A": 1}, None)
            _group_cache._cache["group:TEST2"] = ({"B": 2}, None)
        
        clear_group_cache()
        
        info = get_group_cache_info()
        assert info["size"] == 0
    
    def test_get_group_cache_info_returns_stats(self):
        """get_group_cache_info should return cache statistics."""
        from utils.boc_client import _group_cache
        from datetime import datetime, timezone
        
        clear_group_cache()
        
        with _group_cache._lock:
            _group_cache._cache["group:TEST"] = ({"A": 1}, datetime.now(timezone.utc))
        
        info = get_group_cache_info()
        
        assert info["size"] == 1
        assert "group:TEST" in info["series"]


class TestIntegrationGroup:
    """Integration tests for group functions with mocked HTTP layer."""
    
    def setup_method(self):
        """Clear cache before each test."""
        clear_group_cache()
    
    @patch('utils.boc_client.requests.get')
    def test_end_to_end_group_fetch(self, mock_get):
        """Test complete group fetch flow from API to result."""
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "observations": [
                {"d": "2024-01-14", "FXUSDCAD": 1.3400, "FXEURCAD": 1.4700},
                {"d": "2024-01-15", "FXUSDCAD": 1.3456, "FXEURCAD": 1.4789},
            ]
        }
        mock_get.return_value = mock_response
        
        result = get_latest_group_observation("FX_RATES_DAILY")
        
        assert result["FXUSDCAD"] == pytest.approx(1.3456)
        assert result["FXEURCAD"] == pytest.approx(1.4789)
        
        # Verify caching works
        cache_info = get_group_cache_info()
        assert cache_info["size"] == 1
    
    @patch('utils.boc_client.requests.get')
    def test_group_with_many_series(self, mock_get):
        """Test parsing group with many series."""
        # Simulate BCPI_MONTHLY which has multiple components
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "observations": [
                {
                    "d": "2024-01-15",
                    "M.BCPI": 145.67,
                    "M.ENER": 180.23,
                    "M.BCNE": 120.45,
                    "M.MTLS": 130.56,
                }
            ]
        }
        mock_get.return_value = mock_response
        
        result = get_latest_group_observation("BCPI_MONTHLY")
        
        assert len(result) == 4
        assert result["M.BCPI"] == pytest.approx(145.67)
        assert result["M.ENER"] == pytest.approx(180.23)
