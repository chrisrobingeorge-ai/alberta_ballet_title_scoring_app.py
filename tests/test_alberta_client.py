"""Tests for Alberta Economic Dashboard API client."""
import pytest
from unittest.mock import patch, MagicMock
import json

from utils.alberta_client import (
    get_alberta_indicator,
    get_alberta_economic_indicators,
    get_alberta_indicators_by_category,
    clear_cache,
    get_cache_info,
    get_indicator_metadata,
    get_all_indicator_keys,
    get_indicators_by_category_grouped,
    AlbertaApiError,
    AlbertaDataUnavailableError,
    _parse_alberta_response,
    _fetch_indicator,
    ALBERTA_INDICATORS,
)


class TestParseAlbertaResponse:
    """Tests for Alberta API response parsing."""
    
    def test_parse_numeric_value(self):
        """Should parse numeric float values."""
        data = [
            {"Date": "2024-01-15", "Value": 7.5},
            {"Date": "2024-01-14", "Value": 7.4},
        ]
        latest_date, latest_value = _parse_alberta_response(data)
        assert latest_date == "2024-01-15"
        assert latest_value == 7.5
    
    def test_parse_string_value(self):
        """Should parse string numeric values."""
        data = [
            {"Date": "2024-01-15", "Value": "1,234.56"},
        ]
        latest_date, latest_value = _parse_alberta_response(data)
        assert latest_date == "2024-01-15"
        assert latest_value == 1234.56
    
    def test_parse_empty_data(self):
        """Should return None for empty data."""
        latest_date, latest_value = _parse_alberta_response([])
        assert latest_date is None
        assert latest_value is None
    
    def test_parse_missing_value(self):
        """Should handle missing value field."""
        data = [
            {"Date": "2024-01-15"},
        ]
        latest_date, latest_value = _parse_alberta_response(data)
        assert latest_date is None
        assert latest_value is None
    
    def test_parse_selects_latest_date(self):
        """Should select observation with latest date."""
        data = [
            {"Date": "2024-01-14", "Value": 7.4},
            {"Date": "2024-01-16", "Value": 7.6},  # Latest
            {"Date": "2024-01-15", "Value": 7.5},
        ]
        latest_date, latest_value = _parse_alberta_response(data)
        assert latest_date == "2024-01-16"
        assert latest_value == 7.6
    
    def test_parse_lowercase_keys(self):
        """Should handle lowercase date/value keys."""
        data = [
            {"date": "2024-01-15", "value": 65.2},
        ]
        latest_date, latest_value = _parse_alberta_response(data)
        assert latest_date == "2024-01-15"
        assert latest_value == 65.2
    
    def test_parse_integer_value(self):
        """Should parse integer values as float."""
        data = [
            {"Date": "2024-01-15", "Value": 2500000},
        ]
        latest_date, latest_value = _parse_alberta_response(data)
        assert latest_value == 2500000.0


class TestFetchIndicator:
    """Tests for API fetching with mocked HTTP."""
    
    @patch('utils.alberta_client.requests.get')
    def test_successful_fetch(self, mock_get):
        """Should return value on successful API call."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"Date": "2024-01-15", "Value": 7.5}
        ]
        mock_get.return_value = mock_response
        
        latest_date, value = _fetch_indicator("test-code-123")
        
        assert value == 7.5
        assert latest_date == "2024-01-15"
        mock_get.assert_called_once()
    
    @patch('utils.alberta_client.requests.get')
    def test_timeout_raises_error(self, mock_get):
        """Should raise AlbertaApiError on timeout."""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout()
        
        with pytest.raises(AlbertaApiError, match="Timeout"):
            _fetch_indicator("test-code-123")
    
    @patch('utils.alberta_client.requests.get')
    def test_404_raises_data_unavailable(self, mock_get):
        """Should raise AlbertaDataUnavailableError on 404."""
        import requests
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_get.return_value = mock_response
        
        with pytest.raises(AlbertaDataUnavailableError, match="not found"):
            _fetch_indicator("invalid-code")
    
    @patch('utils.alberta_client.requests.get')
    def test_empty_response_returns_none(self, mock_get):
        """Should return None when response is empty list."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response
        
        latest_date, value = _fetch_indicator("test-code-123")
        
        assert value is None
        assert latest_date is None
    
    @patch('utils.alberta_client.requests.get')
    def test_invalid_json_raises_error(self, mock_get):
        """Should raise AlbertaApiError on invalid JSON."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response
        
        with pytest.raises(AlbertaApiError, match="Invalid JSON"):
            _fetch_indicator("test-code-123")


class TestGetAlbertaIndicator:
    """Tests for get_alberta_indicator function."""
    
    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()
    
    @patch('utils.alberta_client._fetch_indicator')
    def test_returns_value_from_api(self, mock_fetch):
        """Should return value fetched from API."""
        mock_fetch.return_value = ("2024-01-15", 7.5)
        
        result = get_alberta_indicator("ab_unemployment_rate")
        
        assert result == 7.5
    
    @patch('utils.alberta_client._fetch_indicator')
    def test_caches_value(self, mock_fetch):
        """Should cache value after first fetch."""
        mock_fetch.return_value = ("2024-01-15", 7.5)
        
        # First call - fetches from API
        result1 = get_alberta_indicator("ab_unemployment_rate")
        assert result1 == 7.5
        assert mock_fetch.call_count == 1
        
        # Second call - uses cache
        result2 = get_alberta_indicator("ab_unemployment_rate")
        assert result2 == 7.5
        assert mock_fetch.call_count == 1  # Not called again
    
    @patch('utils.alberta_client._fetch_indicator')
    def test_skips_cache_when_disabled(self, mock_fetch):
        """Should skip cache when use_cache=False."""
        mock_fetch.return_value = ("2024-01-15", 7.5)
        
        result1 = get_alberta_indicator("ab_unemployment_rate", use_cache=False)
        result2 = get_alberta_indicator("ab_unemployment_rate", use_cache=False)
        
        assert result1 == 7.5
        assert result2 == 7.5
        assert mock_fetch.call_count == 2
    
    @patch('utils.alberta_client._fetch_indicator')
    def test_returns_none_on_api_error(self, mock_fetch):
        """Should return None when API fails."""
        mock_fetch.side_effect = AlbertaApiError("Test error")
        
        result = get_alberta_indicator("ab_unemployment_rate")
        
        assert result is None
    
    def test_returns_none_for_unknown_key(self):
        """Should return None for unknown indicator key."""
        result = get_alberta_indicator("unknown_indicator")
        assert result is None


class TestGetAlbertaEconomicIndicators:
    """Tests for get_alberta_economic_indicators function."""
    
    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()
    
    @patch('utils.alberta_client._fetch_indicator')
    def test_fetches_all_indicators(self, mock_fetch):
        """Should fetch values for all 12 indicators."""
        mock_fetch.return_value = ("2024-01-15", 100.0)
        
        results = get_alberta_economic_indicators()
        
        # Should have all 12 indicators
        assert len(results) == 12
        assert "ab_unemployment_rate" in results
        assert "ab_wcs_oil_price" in results
        assert "ab_population_quarterly" in results
    
    @patch('utils.alberta_client._fetch_indicator')
    def test_handles_partial_failure(self, mock_fetch):
        """Should handle partial failures gracefully."""
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 3 == 0:
                raise AlbertaApiError("Simulated failure")
            return ("2024-01-15", 100.0)
        
        mock_fetch.side_effect = side_effect
        
        results = get_alberta_economic_indicators()
        
        # Should have some None values due to failures
        assert len(results) == 12
        successful = sum(1 for v in results.values() if v is not None)
        assert successful > 0
        assert successful < 12


class TestCacheFunctions:
    """Tests for cache management functions."""
    
    def test_clear_cache_clears_all(self):
        """clear_cache should remove all cached values."""
        from utils.alberta_client import _cache
        _cache.set("TEST1", 1.0)
        _cache.set("TEST2", 2.0)
        
        clear_cache()
        
        info = get_cache_info()
        assert info["size"] == 0
    
    def test_get_cache_info_returns_stats(self):
        """get_cache_info should return cache statistics."""
        clear_cache()
        
        from utils.alberta_client import _cache
        _cache.set("TEST", 1.0)
        
        info = get_cache_info()
        
        assert info["size"] == 1
        assert "TEST" in info["keys"]


class TestIndicatorMetadata:
    """Tests for indicator metadata functions."""
    
    def test_get_indicator_metadata_returns_dict(self):
        """Should return metadata for valid indicator."""
        metadata = get_indicator_metadata("ab_unemployment_rate")
        
        assert metadata is not None
        assert "api_code" in metadata
        assert "description" in metadata
        assert "category" in metadata
    
    def test_get_indicator_metadata_returns_none_for_unknown(self):
        """Should return None for unknown indicator."""
        metadata = get_indicator_metadata("unknown_indicator")
        assert metadata is None
    
    def test_get_all_indicator_keys(self):
        """Should return all 12 indicator keys."""
        keys = get_all_indicator_keys()
        
        assert len(keys) == 12
        assert "ab_unemployment_rate" in keys
        assert "ab_wcs_oil_price" in keys
    
    def test_get_indicators_by_category_grouped(self):
        """Should group indicators by category."""
        grouped = get_indicators_by_category_grouped()
        
        assert "labour" in grouped
        assert "energy" in grouped
        assert "consumer" in grouped
        assert "population" in grouped
        assert "prices" in grouped
        
        # Labour should have 5 indicators
        assert len(grouped["labour"]) == 5
        assert "ab_unemployment_rate" in grouped["labour"]


class TestGetAlbertaIndicatorsByCategory:
    """Tests for category-based indicator fetching."""
    
    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()
    
    @patch('utils.alberta_client._fetch_indicator')
    def test_fetches_labour_indicators(self, mock_fetch):
        """Should fetch only labour category indicators."""
        mock_fetch.return_value = ("2024-01-15", 100.0)
        
        results = get_alberta_indicators_by_category("labour")
        
        # Should have 5 labour indicators
        assert len(results) == 5
        assert "ab_unemployment_rate" in results
        assert "ab_employment_rate" in results
        assert "ab_wcs_oil_price" not in results
    
    @patch('utils.alberta_client._fetch_indicator')
    def test_fetches_energy_indicators(self, mock_fetch):
        """Should fetch only energy category indicators."""
        mock_fetch.return_value = ("2024-01-15", 50.0)
        
        results = get_alberta_indicators_by_category("energy")
        
        # Should have 1 energy indicator
        assert len(results) == 1
        assert "ab_wcs_oil_price" in results


class TestAlbertaIndicatorsDefinitions:
    """Tests that indicator definitions are correct."""
    
    def test_all_indicators_have_required_fields(self):
        """All indicators should have api_code, description, category."""
        for key, config in ALBERTA_INDICATORS.items():
            assert "api_code" in config, f"{key} missing api_code"
            assert "description" in config, f"{key} missing description"
            assert "category" in config, f"{key} missing category"
    
    def test_indicator_keys_start_with_ab(self):
        """All indicator keys should start with ab_."""
        for key in ALBERTA_INDICATORS:
            assert key.startswith("ab_"), f"{key} does not start with ab_"
    
    def test_api_codes_are_valid_uuids(self):
        """API codes should look like UUIDs."""
        import re
        uuid_pattern = r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'
        
        for key, config in ALBERTA_INDICATORS.items():
            api_code = config["api_code"]
            assert re.match(uuid_pattern, api_code), f"{key} has invalid UUID: {api_code}"
    
    def test_exactly_12_indicators(self):
        """Should have exactly 12 indicators as specified."""
        assert len(ALBERTA_INDICATORS) == 12


class TestIntegration:
    """Integration tests with mocked HTTP layer."""
    
    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()
    
    @patch('utils.alberta_client.requests.get')
    def test_end_to_end_fetch_single(self, mock_get):
        """Test complete fetch flow for a single indicator."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"Date": "2024-01-14", "Value": 7.4},
            {"Date": "2024-01-15", "Value": 7.5},
        ]
        mock_get.return_value = mock_response
        
        result = get_alberta_indicator("ab_unemployment_rate")
        
        assert result == 7.5
        
        # Verify caching works
        cache_info = get_cache_info()
        assert cache_info["size"] == 1
    
    @patch('utils.alberta_client.requests.get')
    def test_end_to_end_fetch_all(self, mock_get):
        """Test complete fetch flow for all indicators."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"Date": "2024-01-15", "Value": 100.0},
        ]
        mock_get.return_value = mock_response
        
        results = get_alberta_economic_indicators()
        
        assert len(results) == 12
        assert all(v == 100.0 for v in results.values())
