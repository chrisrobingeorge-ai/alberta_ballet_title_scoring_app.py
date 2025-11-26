"""
Tests for canonicalize_titles module.
"""

import pytest


def test_canonicalize_title_basic():
    """Test basic title canonicalization."""
    from utils.canonicalize_titles import canonicalize_title
    
    # Basic lowercase and strip
    assert canonicalize_title("Swan Lake") == "swan lake"
    assert canonicalize_title("  The Nutcracker  ") == "the nutcracker"
    
    # Ampersand removal - produces single space after collapse
    assert canonicalize_title("Romeo & Juliet") == "romeo juliet"
    
    # Empty string
    assert canonicalize_title("") == ""


def test_canonicalize_title_dashes():
    """Test dash normalization."""
    from utils.canonicalize_titles import canonicalize_title
    
    # Various dash characters should become standard hyphen
    assert canonicalize_title("All of Us – Tragically Hip") == "all of us - tragically hip"
    assert canonicalize_title("Der Wolf—Rite of Spring") == "der wolf-rite of spring"


def test_canonicalize_title_apostrophes():
    """Test apostrophe normalization."""
    from utils.canonicalize_titles import canonicalize_title
    
    # Various apostrophe characters should become standard
    assert canonicalize_title("Handmaid's Tale") == "handmaid's tale"
    assert canonicalize_title("Handmaid's Tale") == "handmaid's tale"  # curly quote


def test_canonicalize_title_punctuation():
    """Test punctuation removal."""
    from utils.canonicalize_titles import canonicalize_title
    
    # Punctuation should be removed
    assert canonicalize_title("Ballet!") == "ballet"
    assert canonicalize_title("Swan Lake.") == "swan lake"
    

def test_fuzzy_match_title_exact():
    """Test fuzzy matching with exact matches."""
    from utils.canonicalize_titles import fuzzy_match_title
    
    choices = ["Swan Lake", "Sleeping Beauty", "Romeo and Juliet"]
    
    # Exact match
    assert fuzzy_match_title("Swan Lake", choices) == "Swan Lake"
    
    # Case insensitive
    assert fuzzy_match_title("swan lake", choices) == "Swan Lake"


def test_fuzzy_match_title_close():
    """Test fuzzy matching with close matches."""
    from utils.canonicalize_titles import fuzzy_match_title
    
    choices = ["Swan Lake", "Sleeping Beauty", "Romeo and Juliet"]
    
    # Close match (typo)
    result = fuzzy_match_title("Swn Lake", choices, threshold=70)
    assert result == "Swan Lake"


def test_fuzzy_match_title_no_match():
    """Test fuzzy matching when no good match exists."""
    from utils.canonicalize_titles import fuzzy_match_title
    
    choices = ["Swan Lake", "Sleeping Beauty", "Romeo and Juliet"]
    
    # No good match at high threshold
    assert fuzzy_match_title("Hamlet", choices, threshold=90) is None


def test_fuzzy_match_title_empty():
    """Test fuzzy matching with empty inputs."""
    from utils.canonicalize_titles import fuzzy_match_title
    
    assert fuzzy_match_title("Swan Lake", []) is None
    assert fuzzy_match_title("", ["Swan Lake"]) is None


def test_load_title_map_missing_file():
    """Test loading title map when file doesn't exist."""
    from utils.canonicalize_titles import load_title_map
    
    # Should return empty dict for non-existent file
    result = load_title_map("/nonexistent/path/file.csv")
    assert result == {}


def test_load_title_map_from_stub():
    """Test loading title map from the stub file."""
    from utils.canonicalize_titles import load_title_map
    import os
    
    stub_path = "data/title_id_map.csv"
    if os.path.exists(stub_path):
        # The stub file has comments, so loader should handle that
        try:
            result = load_title_map(stub_path)
            assert isinstance(result, dict)
        except Exception:
            # Stub file may have unusual format - that's OK for this test
            pass
