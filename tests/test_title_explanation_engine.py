"""
Test suite for the Title Explanation Engine

This test validates that the SHAP-driven narrative generator produces
appropriate multi-paragraph explanations for different title types.

Author: Alberta Ballet Data Science Team
Date: December 2025
"""

import pytest
from ml.title_explanation_engine import (
    build_title_explanation,
    _describe_signal_level,
    _describe_category,
    _interpret_ticket_index,
    _format_list,
)


class TestTitleExplanationEngine:
    """Test the narrative generation engine."""
    
    def test_build_explanation_familiar_classic(self):
        """Test narrative generation for a familiar classic title."""
        title_metadata = {
            "Title": "Swan Lake",
            "Month": "December 2025",
            "Category": "adult_classic",
            "Familiarity": 135.0,
            "Motivation": 115.0,
            "SignalOnly": 125.0,
            "TicketIndex used": 120.0,
            "FutureSeasonalityFactor": 1.15,
            "PrimarySegment": "Adult Enthusiast",
            "SecondarySegment": "Tourist",
            "YYC_Singles": 4200,
            "YEG_Singles": 2800,
            "ReturnDecayPct": 0.0,
            "IsRemount": False,
        }
        
        narrative = build_title_explanation(title_metadata)
        
        # Validate output characteristics
        assert len(narrative) > 200, "Narrative should be substantial"
        assert "Swan Lake" in narrative
        assert "adult_classic" in narrative or "classical" in narrative
        assert "December" in narrative
        assert "135.0" in narrative or "Familiarity" in narrative
        assert "4200" in narrative or "4,200" in narrative  # Calgary tickets
        assert "2800" in narrative or "2,800" in narrative  # Edmonton tickets
        
        # Check that it contains key concepts
        assert any(term in narrative for term in ["Wikipedia", "Google", "YouTube", "Spotify"])
        assert "Ticket Index" in narrative
        
        # Should be HTML-safe (for PDF generation)
        assert "<b>" in narrative
        
        print(f"\nGenerated narrative for Swan Lake:\n{narrative}\n")
    
    def test_build_explanation_premiere(self):
        """Test narrative generation for a premiere."""
        title_metadata = {
            "Title": "New Contemporary Work",
            "Month": "March 2026",
            "Category": "contemporary",
            "Familiarity": 45.0,
            "Motivation": 65.0,
            "SignalOnly": 55.0,
            "TicketIndex used": 75.0,
            "FutureSeasonalityFactor": 0.92,
            "PrimarySegment": "Art Explorer",
            "SecondarySegment": "Adult Enthusiast",
            "YYC_Singles": 1800,
            "YEG_Singles": 1200,
            "ReturnDecayPct": 0.0,
            "IsRemount": False,
        }
        
        narrative = build_title_explanation(title_metadata)
        
        # Validate premiere-specific content
        assert "premiere" in narrative.lower() or "new" in narrative.lower()
        assert "contemporary" in narrative.lower()
        assert len(narrative) > 200
        
        print(f"\nGenerated narrative for Contemporary Premiere:\n{narrative}\n")
    
    def test_build_explanation_remount(self):
        """Test narrative generation for a remount."""
        title_metadata = {
            "Title": "The Nutcracker",
            "Month": "November 2025",
            "Category": "holiday",
            "Familiarity": 145.0,
            "Motivation": 130.0,
            "SignalOnly": 137.5,
            "TicketIndex used": 140.0,
            "FutureSeasonalityFactor": 1.25,
            "PrimarySegment": "Family",
            "SecondarySegment": "Holiday Seeker",
            "YYC_Singles": 6500,
            "YEG_Singles": 4300,
            "ReturnDecayPct": 0.0,
            "IsRemount": True,
            "YearsSinceLastRun": 1,
        }
        
        narrative = build_title_explanation(title_metadata)
        
        # Validate remount-specific content
        assert "remount" in narrative.lower()
        assert "1 year" in narrative or "annually" in narrative.lower()
        assert "Nutcracker" in narrative
        assert len(narrative) > 200
        
        print(f"\nGenerated narrative for Nutcracker Remount:\n{narrative}\n")
    
    def test_build_explanation_with_shap(self):
        """Test narrative generation with SHAP values."""
        title_metadata = {
            "Title": "Romeo and Juliet",
            "Month": "February 2026",
            "Category": "adult_classic",
            "Familiarity": 105.0,
            "Motivation": 95.0,
            "SignalOnly": 100.0,
            "TicketIndex used": 100.0,
            "FutureSeasonalityFactor": 0.98,
            "PrimarySegment": "Adult Enthusiast",
            "SecondarySegment": "",
            "YYC_Singles": 3000,
            "YEG_Singles": 2000,
            "ReturnDecayPct": 0.0,
            "IsRemount": False,
        }
        
        shap_values = {
            "familiarity_score": 8.5,
            "motivation_score": -2.3,
            "seasonality_factor": -1.5,
            "category_adult_classic": 5.2,
            "prior_median_tickets": 3.8,
        }
        
        narrative = build_title_explanation(
            title_metadata,
            shap_values=shap_values
        )
        
        # Validate SHAP-driven content
        assert len(narrative) > 200
        assert "Romeo and Juliet" in narrative
        
        # Should mention drivers or contributions
        assert any(term in narrative.lower() for term in 
                  ["driver", "contribute", "factor", "elevate", "moderate"])
        
        print(f"\nGenerated narrative for Romeo and Juliet with SHAP:\n{narrative}\n")
    
    def test_missing_optional_fields(self):
        """Test that narrative generation handles missing optional fields gracefully."""
        minimal_metadata = {
            "Title": "Test Ballet",
            "Month": "January 2026",
            "Category": "contemporary",
            "TicketIndex used": 85.0,
        }
        
        narrative = build_title_explanation(minimal_metadata)
        
        # Should still generate something reasonable
        assert len(narrative) > 50
        assert "Test Ballet" in narrative
        assert "January" in narrative
        
        print(f"\nGenerated narrative with minimal metadata:\n{narrative}\n")
    
    def test_signal_level_descriptions(self):
        """Test signal level description helper."""
        assert _describe_signal_level(150) == "exceptionally high"
        assert _describe_signal_level(110) == "strong"
        assert _describe_signal_level(85) == "above average"
        assert _describe_signal_level(65) == "moderate"
        assert _describe_signal_level(45) == "emerging"
        assert _describe_signal_level(25) == "limited"
    
    def test_category_descriptions(self):
        """Test category description helper."""
        assert "family" in _describe_category("family_classic").lower()
        assert "adult" in _describe_category("adult_classic").lower()
        assert "contemporary" in _describe_category("contemporary").lower()
        assert "holiday" in _describe_category("holiday").lower()
    
    def test_ticket_index_interpretation(self):
        """Test Ticket Index interpretation helper."""
        assert "exceptional" in _interpret_ticket_index(125).lower()
        assert "strong" in _interpret_ticket_index(110).lower()
        assert "benchmark" in _interpret_ticket_index(98).lower()
        assert "moderate" in _interpret_ticket_index(85).lower()
        assert "developing" in _interpret_ticket_index(65).lower()
    
    def test_list_formatting(self):
        """Test list formatting helper."""
        assert _format_list([]) == ""
        assert _format_list(["one"]) == "one"
        assert _format_list(["one", "two"]) == "one and two"
        assert _format_list(["one", "two", "three"]) == "one, two, and three"
    
    def test_narrative_word_count(self):
        """Test that narratives meet the target word count (~250-350 words)."""
        title_metadata = {
            "Title": "Giselle",
            "Month": "April 2026",
            "Category": "adult_classic",
            "Familiarity": 98.0,
            "Motivation": 88.0,
            "SignalOnly": 93.0,
            "TicketIndex used": 95.0,
            "FutureSeasonalityFactor": 1.02,
            "PrimarySegment": "Adult Enthusiast",
            "SecondarySegment": "Art Explorer",
            "YYC_Singles": 2900,
            "YEG_Singles": 1900,
            "ReturnDecayPct": 0.0,
            "IsRemount": False,
        }
        
        narrative = build_title_explanation(title_metadata)
        
        # Count words (rough approximation)
        # Strip HTML tags for word counting
        import re
        text_only = re.sub(r'<[^>]+>', '', narrative)
        words = text_only.split()
        word_count = len(words)
        
        print(f"\nWord count for Giselle narrative: {word_count}")
        
        # Should be in the target range (with some flexibility)
        assert 150 <= word_count <= 500, f"Word count {word_count} outside target range"
    
    def test_html_safety(self):
        """Test that generated narratives are HTML-safe for PDF generation."""
        title_metadata = {
            "Title": "Test & Special <Characters>",
            "Month": "May 2026",
            "Category": "contemporary",
            "TicketIndex used": 90.0,
            "YYC_Singles": 2000,
            "YEG_Singles": 1500,
        }
        
        narrative = build_title_explanation(title_metadata)
        
        # Should contain the title (though special chars might be escaped/handled)
        assert "Test" in narrative
        
        # Should have proper HTML formatting
        assert "<b>" in narrative


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
