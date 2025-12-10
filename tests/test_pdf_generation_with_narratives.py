"""
Integration test for PDF generation with new SHAP-driven narratives.

This test validates that the enhanced Season Report PDF can be generated
with the new multi-paragraph narratives.

Author: Alberta Ballet Data Science Team
Date: December 2025
"""

import pytest
import pandas as pd
import io


def test_pdf_generation_with_new_narratives():
    """
    Test that the PDF report can be generated with enhanced narratives.
    
    This is a smoke test to ensure the new narrative engine integrates
    properly with the PDF generation pipeline.
    """
    try:
        # Import PDF generation functions
        from streamlit_app import (
            build_full_pdf_report,
            _methodology_glossary_text,
            _plain_language_overview_text,
        )
        
        # Create sample plan data
        sample_data = {
            "Month": ["December 2025", "February 2026", "April 2026"],
            "Title": ["Swan Lake", "Romeo and Juliet", "Contemporary Work"],
            "Category": ["adult_classic", "adult_classic", "contemporary"],
            "Familiarity": [135.0, 105.0, 65.0],
            "Motivation": [115.0, 95.0, 75.0],
            "SignalOnly": [125.0, 100.0, 70.0],
            "TicketIndex used": [120.0, 100.0, 80.0],
            "FutureSeasonalityFactor": [1.15, 0.98, 1.02],
            "PrimarySegment": ["Adult Enthusiast", "Adult Enthusiast", "Art Explorer"],
            "SecondarySegment": ["Tourist", "Art Explorer", "Adult Enthusiast"],
            "YYC_Singles": [4200, 3000, 2000],
            "YEG_Singles": [2800, 2000, 1300],
            "ReturnDecayPct": [0.0, 0.0, 0.0],
            "IsRemount": [False, False, False],
        }
        
        plan_df = pd.DataFrame(sample_data)
        
        # Generate methodology paragraphs
        methodology_paragraphs = _methodology_glossary_text()
        
        # Generate the PDF
        pdf_bytes = build_full_pdf_report(
            methodology_paragraphs=methodology_paragraphs,
            plan_df=plan_df,
            season_year=2025,
            org_name="Alberta Ballet"
        )
        
        # Validate PDF was generated
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 10000, "PDF should be substantial in size"
        
        # Check PDF magic number (PDF files start with %PDF)
        assert pdf_bytes[:4] == b'%PDF', "Should be a valid PDF file"
        
        print(f"\n✓ PDF generated successfully: {len(pdf_bytes)} bytes")
        
        # Optionally save to a temp file for manual inspection
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes)
            output_path = f.name
        
        print(f"✓ PDF saved to: {output_path}")
        
        return True
        
    except ImportError as e:
        pytest.skip(f"Required dependencies not available: {e}")
    except Exception as e:
        pytest.fail(f"PDF generation failed: {e}")


def test_plain_language_overview_enhanced():
    """Test that the enhanced plain language overview generates properly."""
    try:
        from streamlit_app import _plain_language_overview_text
        
        overview_flowables = _plain_language_overview_text()
        
        # Should return a list of ReportLab flowables
        assert isinstance(overview_flowables, list)
        assert len(overview_flowables) > 10, "Should have multiple paragraphs and spacers"
        
        print(f"\n✓ Plain language overview generated: {len(overview_flowables)} flowables")
        
        return True
        
    except ImportError as e:
        pytest.skip(f"Required dependencies not available: {e}")


def test_narrative_for_row_integration():
    """Test that _narrative_for_row uses the new explanation engine."""
    try:
        from streamlit_app import _narrative_for_row
        
        sample_row = {
            "Title": "The Nutcracker",
            "Month": "December 2025",
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
        
        narrative = _narrative_for_row(sample_row)
        
        # Validate narrative
        assert isinstance(narrative, str)
        assert len(narrative) > 200, "Should be a substantial narrative"
        assert "Nutcracker" in narrative
        assert "<b>" in narrative, "Should contain HTML formatting"
        
        # Should be comprehensive (target 250-350 words)
        import re
        text_only = re.sub(r'<[^>]+>', '', narrative)
        words = text_only.split()
        word_count = len(words)
        
        print(f"\n✓ Narrative generated: {word_count} words")
        print(f"\nSample narrative:\n{narrative[:500]}...\n")
        
        assert 100 <= word_count <= 600, f"Word count {word_count} should be substantial"
        
        return True
        
    except ImportError as e:
        pytest.skip(f"Required dependencies not available: {e}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
