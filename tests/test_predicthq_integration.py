"""
Tests for PredictHQ integration.

Tests the PredictHQ API client and data loader functions.
"""

import pytest
from datetime import date, datetime
from unittest.mock import patch, MagicMock
import pandas as pd

from integrations.predicthq import (
    PredictHQClient,
    PredictHQEvent,
    PredictHQFeatures,
    PredictHQError,
    PredictHQAuthError,
    get_predicthq_features_dict,
    ALBERTA_LOCATIONS,
    PHQ_RELEVANT_CATEGORIES,
)
from data.loader import load_predicthq_events, load_history_with_predicthq


class TestPredictHQClient:
    """Tests for PredictHQClient initialization."""
    
    def test_client_requires_api_key(self):
        """Test that client raises error without API key."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(PredictHQAuthError) as exc_info:
                PredictHQClient()
            assert "PREDICTHQ_API_KEY" in str(exc_info.value)
    
    def test_client_accepts_api_key_parameter(self):
        """Test that client accepts API key as parameter."""
        client = PredictHQClient(api_key="test_key")
        assert client.api_key == "test_key"
    
    def test_client_reads_from_env(self):
        """Test that client reads API key from environment."""
        with patch.dict('os.environ', {'PREDICTHQ_API_KEY': 'env_test_key'}):
            client = PredictHQClient()
            assert client.api_key == "env_test_key"


class TestPredictHQEvent:
    """Tests for PredictHQEvent dataclass."""
    
    def test_event_creation(self):
        """Test creating a PredictHQEvent."""
        event = PredictHQEvent(
            event_id="abc123",
            title="Test Event",
            category="concerts",
            rank=75,
            phq_attendance=5000,
        )
        assert event.event_id == "abc123"
        assert event.title == "Test Event"
        assert event.category == "concerts"
        assert event.rank == 75
        assert event.phq_attendance == 5000
    
    def test_event_defaults(self):
        """Test default values for PredictHQEvent."""
        event = PredictHQEvent(event_id="test", title="Test", category="sports")
        assert event.labels == []
        assert event.rank == 0
        assert event.phq_attendance is None
        assert event.raw_data == {}


class TestPredictHQFeatures:
    """Tests for PredictHQFeatures dataclass."""
    
    def test_features_creation(self):
        """Test creating PredictHQFeatures."""
        features = PredictHQFeatures(
            city="Calgary",
            start_date="2024-12-15",
            end_date="2024-12-22",
            phq_attendance_sum=25000,
            phq_event_count=5,
        )
        assert features.city == "Calgary"
        assert features.phq_attendance_sum == 25000
        assert features.phq_event_count == 5
    
    def test_features_defaults(self):
        """Test default values for PredictHQFeatures."""
        features = PredictHQFeatures(
            city="Edmonton",
            start_date="2024-01-01",
            end_date="2024-01-07"
        )
        assert features.phq_attendance_sum == 0
        assert features.phq_attendance_sports == 0
        assert features.phq_event_count == 0
        assert features.phq_rank_max == 0
        assert features.phq_holidays_flag is False
        assert features.phq_severe_weather_flag is False


class TestGetPredicthqFeaturesDict:
    """Tests for get_predicthq_features_dict helper function."""
    
    def test_converts_to_dict(self):
        """Test conversion of features to dictionary."""
        features = PredictHQFeatures(
            city="Calgary",
            start_date="2024-12-15",
            end_date="2024-12-22",
            phq_attendance_sum=25000,
            phq_event_count=5,
            phq_rank_max=85,
            phq_holidays_flag=True,
        )
        
        result = get_predicthq_features_dict(features)
        
        assert isinstance(result, dict)
        assert result["city"] == "Calgary"
        assert result["phq_attendance_sum"] == 25000
        assert result["phq_event_count"] == 5
        assert result["phq_rank_max"] == 85
        assert result["phq_holidays_flag"] == 1  # Converted to int
    
    def test_all_keys_present(self):
        """Test that all expected keys are in the dictionary."""
        features = PredictHQFeatures(
            city="Edmonton",
            start_date="2024-01-01",
            end_date="2024-01-07"
        )
        
        result = get_predicthq_features_dict(features)
        
        expected_keys = [
            "city", "phq_start_date", "phq_end_date",
            "phq_attendance_sum", "phq_attendance_sports",
            "phq_attendance_concerts", "phq_attendance_performing_arts",
            "phq_event_count", "phq_rank_max", "phq_rank_avg",
            "phq_holidays_flag", "phq_severe_weather_flag",
            "phq_event_spend", "phq_demand_impact_score",
        ]
        
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"


class TestAlbertaLocations:
    """Tests for Alberta location constants."""
    
    def test_calgary_location(self):
        """Test Calgary location is defined correctly."""
        assert "Calgary" in ALBERTA_LOCATIONS
        calgary = ALBERTA_LOCATIONS["Calgary"]
        assert "lat" in calgary
        assert "lon" in calgary
        assert "radius" in calgary
        # Verify approximate coordinates
        assert 50 < calgary["lat"] < 52
        assert -115 < calgary["lon"] < -113
    
    def test_edmonton_location(self):
        """Test Edmonton location is defined correctly."""
        assert "Edmonton" in ALBERTA_LOCATIONS
        edmonton = ALBERTA_LOCATIONS["Edmonton"]
        assert "lat" in edmonton
        assert "lon" in edmonton
        assert "radius" in edmonton
        # Verify approximate coordinates
        assert 53 < edmonton["lat"] < 54
        assert -114 < edmonton["lon"] < -113


class TestRelevantCategories:
    """Tests for event category constants."""
    
    def test_relevant_categories_defined(self):
        """Test that relevant categories are defined."""
        assert len(PHQ_RELEVANT_CATEGORIES) > 0
        assert "concerts" in PHQ_RELEVANT_CATEGORIES
        assert "sports" in PHQ_RELEVANT_CATEGORIES
        assert "performing-arts" in PHQ_RELEVANT_CATEGORIES


class TestPredictHQLoaders:
    """Tests for PredictHQ data loader functions."""
    
    def test_load_predicthq_events_fallback_empty(self):
        """Test that loader returns empty DataFrame when file doesn't exist."""
        df = load_predicthq_events("nonexistent_file.csv", fallback_empty=True)
        assert isinstance(df, pd.DataFrame)
        assert df.empty
    
    def test_load_history_with_predicthq_no_predicthq_data(self):
        """Test that loader returns history only when no PredictHQ data."""
        # Should return history without merging since predicthq file doesn't exist
        result = load_history_with_predicthq(
            predicthq_csv="nonexistent.csv"
        )
        # Result should be the history data (not empty if history exists)
        assert isinstance(result, pd.DataFrame)


class TestPredictHQClientMocked:
    """Tests for PredictHQClient with mocked API responses."""
    
    @patch('integrations.predicthq.requests.request')
    def test_search_events_returns_list(self, mock_request):
        """Test that search_events returns a list of events."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "count": 2,
            "results": [
                {
                    "id": "event1",
                    "title": "NHL Game",
                    "category": "sports",
                    "rank": 80,
                    "phq_attendance": 15000,
                },
                {
                    "id": "event2",
                    "title": "Concert",
                    "category": "concerts",
                    "rank": 60,
                    "phq_attendance": 3000,
                },
            ],
            "next": None,
        }
        mock_request.return_value = mock_response
        
        client = PredictHQClient(api_key="test_key")
        events = client.search_events(city="Calgary", start_date="2024-12-15")
        
        assert isinstance(events, list)
        assert len(events) == 2
        assert events[0].title == "NHL Game"
        assert events[0].category == "sports"
        assert events[1].title == "Concert"
    
    @patch('integrations.predicthq.requests.request')
    def test_get_features_for_run(self, mock_request):
        """Test that get_features_for_run aggregates features correctly."""
        # Setup mock responses for different API calls
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        # Return different results for different calls
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            
            if call_count[0] == 1:
                # Regular events
                mock_resp.json.return_value = {
                    "count": 2,
                    "results": [
                        {"id": "e1", "title": "NHL", "category": "sports", "rank": 80, "phq_attendance": 15000},
                        {"id": "e2", "title": "Concert", "category": "concerts", "rank": 60, "phq_attendance": 3000},
                    ],
                    "next": None,
                }
            elif call_count[0] == 2:
                # Holiday events
                mock_resp.json.return_value = {
                    "count": 1,
                    "results": [
                        {"id": "h1", "title": "Christmas", "category": "public-holidays", "rank": 90},
                    ],
                    "next": None,
                }
            else:
                # Weather events (empty)
                mock_resp.json.return_value = {
                    "count": 0,
                    "results": [],
                    "next": None,
                }
            return mock_resp
        
        mock_request.side_effect = side_effect
        
        client = PredictHQClient(api_key="test_key")
        features = client.get_features_for_run(
            city="Calgary",
            start_date="2024-12-15",
            end_date="2024-12-22"
        )
        
        assert features.city == "Calgary"
        assert features.phq_attendance_sum == 18000  # 15000 + 3000
        assert features.phq_attendance_sports == 15000
        assert features.phq_attendance_concerts == 3000
        assert features.phq_holidays_flag is True
        assert features.phq_severe_weather_flag is False


class TestPredictHQErrorHandling:
    """Tests for error handling in PredictHQ client."""
    
    @patch('integrations.predicthq.requests.request')
    def test_auth_error_on_401(self, mock_request):
        """Test that 401 response raises PredictHQAuthError."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_request.return_value = mock_response
        
        client = PredictHQClient(api_key="invalid_key")
        
        with pytest.raises(PredictHQAuthError):
            client.search_events(city="Calgary")
    
    @patch('integrations.predicthq.requests.request')
    def test_auth_error_on_403(self, mock_request):
        """Test that 403 response raises PredictHQAuthError."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_request.return_value = mock_response
        
        client = PredictHQClient(api_key="limited_key")
        
        with pytest.raises(PredictHQAuthError):
            client.search_events(city="Calgary")
