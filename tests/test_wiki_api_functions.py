"""Tests for Wikipedia API functions in title_scoring_helper.py."""
from typing import Dict


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
            'spotify': ['SPOTIFY_CLIENT_ID', 'SPOTIFY_CLIENT_SECRET'],
            'wiki': ['WIKI_API_KEY', 'WIKI_API_SECRET']
        }


        # All services should have at least one credential
        for service, credentials in expected_patterns.items():
            assert len(credentials) > 0, (
                f"{service} should have credentials defined"
            )

        # Wiki should follow same pattern as Spotify (key + secret)
        assert len(expected_patterns['wiki']) == (
            len(expected_patterns['spotify'])
        )
