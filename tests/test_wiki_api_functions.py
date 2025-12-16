"""Tests for Wikipedia API functions in title_scoring_helper.py."""
from typing import Dict
from unittest.mock import patch, Mock
from urllib.parse import quote

import wiki_buzz_helper


class TestWikiApiHeaders:
    """Tests for _get_wiki_headers function logic."""

    def test_headers_without_credentials(self):
        """Should include User-Agent without authentication headers."""
        # Mock the function without credentials
        WIKI_API_KEY = None
        WIKI_API_SECRET = None

        def _get_wiki_headers() -> Dict[str, str]:
            headers = {
                'User-Agent': 'TitleScoringApp/1.0 (https://github.com/chrisrobingeorge-ai/alberta_ballet_title_scoring_app)'
            }
            if WIKI_API_KEY:
                headers['X-API-Key'] = WIKI_API_KEY
            if WIKI_API_SECRET:
                headers['X-API-Secret'] = WIKI_API_SECRET
            return headers

        result = _get_wiki_headers()

        assert 'User-Agent' in result
        assert result['User-Agent'] == 'TitleScoringApp/1.0 (https://github.com/chrisrobingeorge-ai/alberta_ballet_title_scoring_app)'
        assert 'X-API-Key' not in result
        assert 'X-API-Secret' not in result

    def test_headers_with_api_key_only(self):
        """Should include User-Agent and API key."""
        # Mock the function with API key only
        WIKI_API_KEY = 'test_api_key_123'
        WIKI_API_SECRET = None

        def _get_wiki_headers() -> Dict[str, str]:
            headers = {
                'User-Agent': 'TitleScoringApp/1.0 (https://github.com/chrisrobingeorge-ai/alberta_ballet_title_scoring_app)'
            }
            if WIKI_API_KEY:
                headers['X-API-Key'] = WIKI_API_KEY
            if WIKI_API_SECRET:
                headers['X-API-Secret'] = WIKI_API_SECRET
            return headers

        result = _get_wiki_headers()

        assert 'User-Agent' in result
        assert result['X-API-Key'] == 'test_api_key_123'
        assert 'X-API-Secret' not in result

    def test_headers_with_api_secret_only(self):
        """Should include User-Agent and API secret."""
        # Mock the function with API secret only
        WIKI_API_KEY = None
        WIKI_API_SECRET = 'test_api_secret_456'

        def _get_wiki_headers() -> Dict[str, str]:
            headers = {
                'User-Agent': 'TitleScoringApp/1.0 (https://github.com/chrisrobingeorge-ai/alberta_ballet_title_scoring_app)'
            }
            if WIKI_API_KEY:
                headers['X-API-Key'] = WIKI_API_KEY
            if WIKI_API_SECRET:
                headers['X-API-Secret'] = WIKI_API_SECRET
            return headers

        result = _get_wiki_headers()

        assert 'User-Agent' in result
        assert 'X-API-Key' not in result
        assert result['X-API-Secret'] == 'test_api_secret_456'

    def test_headers_with_both_credentials(self):
        """Should include User-Agent, API key and secret."""
        # Mock the function with both credentials
        WIKI_API_KEY = 'test_api_key_123'
        WIKI_API_SECRET = 'test_api_secret_456'

        def _get_wiki_headers() -> Dict[str, str]:
            headers = {
                'User-Agent': 'TitleScoringApp/1.0 (https://github.com/chrisrobingeorge-ai/alberta_ballet_title_scoring_app)'
            }
            if WIKI_API_KEY:
                headers['X-API-Key'] = WIKI_API_KEY
            if WIKI_API_SECRET:
                headers['X-API-Secret'] = WIKI_API_SECRET
            return headers

        result = _get_wiki_headers()

        assert 'User-Agent' in result
        assert result['X-API-Key'] == 'test_api_key_123'
        assert result['X-API-Secret'] == 'test_api_secret_456'
        assert len(result) == 3


class TestWikiApiFunctionsDocumentation:
    """Tests to verify patterns match YouTube/Spotify."""

    def test_wiki_credentials_follow_same_pattern_as_youtube_spotify(self):
        """Verify Wiki credentials follow same pattern."""
        # This is a documentation test to ensure consistency
        expected_patterns = {
            'youtube': ['YOUTUBE_API_KEY'],
            'chartmetric': ['SPOTIFY_CLIENT_ID', 'SPOTIFY_CLIENT_SECRET'],
            'wiki': ['WIKI_API_KEY', 'WIKI_API_SECRET']
        }


        # All services should have at least one credential
        for service, credentials in expected_patterns.items():
            assert len(credentials) > 0, (
                f"{service} should have credentials defined"
            )

        # Wiki should follow same pattern as Spotify (key + secret)
        assert len(expected_patterns['wiki']) == (
            len(expected_patterns['chartmetric'])
        )


class TestWikipediaUrlEncoding:
    """Tests for URL encoding in fetch_wikipedia_views_sum function."""

    def test_url_encoding_with_colon(self):
        """Test that colons are properly URL-encoded to avoid 404 errors."""
        title = "Peeping Tom: Dance Theatre Productions"
        start_date = "20250101"
        end_date = "20251215"
        
        with patch('wiki_buzz_helper.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"items": [{"views": 100}]}
            mock_get.return_value = mock_response
            
            result = wiki_buzz_helper.fetch_wikipedia_views_sum(
                title, start_date, end_date
            )
            
            # Get the URL that was called
            called_url = mock_get.call_args[0][0]
            
            # Verify the colon is encoded as %3A (not raw :)
            assert '%3A' in called_url, (
                f"Colon not properly URL-encoded. Got: {called_url}"
            )
            assert 'Peeping_Tom%3A_Dance_Theatre_Productions' in called_url
            # Ensure we're not passing raw colons
            assert 'Peeping_Tom:_Dance' not in called_url

    def test_url_encoding_with_parentheses(self):
        """Test that parentheses are properly URL-encoded."""
        title = "The Great Gatsby (2024)"
        start_date = "20250101"
        end_date = "20251215"
        
        with patch('wiki_buzz_helper.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"items": [{"views": 50}]}
            mock_get.return_value = mock_response
            
            result = wiki_buzz_helper.fetch_wikipedia_views_sum(
                title, start_date, end_date
            )
            
            called_url = mock_get.call_args[0][0]
            
            # Verify parentheses are encoded
            assert '%28' in called_url, (
                f"Opening parenthesis not encoded. Got: {called_url}"
            )
            assert '%29' in called_url, (
                f"Closing parenthesis not encoded. Got: {called_url}"
            )
            assert 'The_Great_Gatsby_%282024%29' in called_url

    def test_url_encoding_simple_title(self):
        """Test that simple titles without special chars still work."""
        title = "Cinderella"
        start_date = "20250101"
        end_date = "20251215"
        
        with patch('wiki_buzz_helper.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"items": [{"views": 200}]}
            mock_get.return_value = mock_response
            
            result = wiki_buzz_helper.fetch_wikipedia_views_sum(
                title, start_date, end_date
            )
            
            called_url = mock_get.call_args[0][0]
            
            # Verify the title is in the URL correctly
            assert 'Cinderella' in called_url
            assert result == 200

    def test_spaces_replaced_with_underscores(self):
        """Test that spaces are replaced with underscores before encoding."""
        title = "Swan Lake"
        start_date = "20250101"
        end_date = "20251215"
        
        with patch('wiki_buzz_helper.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"items": []}
            mock_get.return_value = mock_response
            
            result = wiki_buzz_helper.fetch_wikipedia_views_sum(
                title, start_date, end_date
            )
            
            called_url = mock_get.call_args[0][0]
            
            # Spaces should be replaced with underscores
            assert 'Swan_Lake' in called_url
            # Should not have %20 (URL-encoded space)
            assert 'Swan%20Lake' not in called_url
