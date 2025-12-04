"""
Tests for Season Summary (Board View) functionality.

These tests verify the helper functions and DataFrame construction
for the board-friendly Season Summary feature.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestIndexStrengthRating:
    """Test the index_strength_rating helper function."""

    def test_very_weak_rating(self):
        """Test rating for very weak index (< 40)."""
        from streamlit_app import index_strength_rating
        
        assert index_strength_rating(0) == "★☆☆☆☆"
        assert index_strength_rating(20) == "★☆☆☆☆"
        assert index_strength_rating(39) == "★☆☆☆☆"

    def test_below_average_rating(self):
        """Test rating for below average index (40-70)."""
        from streamlit_app import index_strength_rating
        
        assert index_strength_rating(40) == "★★☆☆☆"
        assert index_strength_rating(55) == "★★☆☆☆"
        assert index_strength_rating(69) == "★★☆☆☆"

    def test_average_rating(self):
        """Test rating for average index (70-100)."""
        from streamlit_app import index_strength_rating
        
        assert index_strength_rating(70) == "★★★☆☆"
        assert index_strength_rating(85) == "★★★☆☆"
        assert index_strength_rating(99) == "★★★☆☆"

    def test_above_average_rating(self):
        """Test rating for above average index (100-130)."""
        from streamlit_app import index_strength_rating
        
        assert index_strength_rating(100) == "★★★★☆"
        assert index_strength_rating(115) == "★★★★☆"
        assert index_strength_rating(129) == "★★★★☆"

    def test_strong_rating(self):
        """Test rating for strong index (130+)."""
        from streamlit_app import index_strength_rating
        
        assert index_strength_rating(130) == "★★★★★"
        assert index_strength_rating(150) == "★★★★★"
        assert index_strength_rating(200) == "★★★★★"

    def test_invalid_input_returns_empty_stars(self):
        """Test that invalid input returns empty stars."""
        from streamlit_app import index_strength_rating
        
        assert index_strength_rating(None) == "☆☆☆☆☆"
        assert index_strength_rating("invalid") == "☆☆☆☆☆"
        assert index_strength_rating(np.nan) == "☆☆☆☆☆"


class TestSegmentTiltLabel:
    """Test the segment_tilt_label helper function."""

    def test_primary_segment_returned(self):
        """Test that PrimarySegment is used first."""
        from streamlit_app import segment_tilt_label
        
        row = {"PrimarySegment": "Family (Parents w/ kids)"}
        assert segment_tilt_label(row) == "Family (Parents w/ kids)"

    def test_dominant_segment_fallback(self):
        """Test fallback to dominant_audience_segment."""
        from streamlit_app import segment_tilt_label
        
        row = {"dominant_audience_segment": "Emerging Adults (18–34)"}
        assert segment_tilt_label(row) == "Emerging Adults (18–34)"

    def test_predicted_segment_fallback(self):
        """Test fallback to PredictedPrimarySegment."""
        from streamlit_app import segment_tilt_label
        
        row = {"PredictedPrimarySegment": "Core Classical (F35–64)"}
        assert segment_tilt_label(row) == "Core Classical (F35–64)"

    def test_general_population_default(self):
        """Test default value when no segment data available."""
        from streamlit_app import segment_tilt_label
        
        assert segment_tilt_label({}) == "General Population"
        assert segment_tilt_label({"PrimarySegment": ""}) == "General Population"
        assert segment_tilt_label({"PrimarySegment": None}) == "General Population"

    def test_priority_order(self):
        """Test that priority order is: PrimarySegment > dominant > Predicted."""
        from streamlit_app import segment_tilt_label
        
        row = {
            "PrimarySegment": "First",
            "dominant_audience_segment": "Second",
            "PredictedPrimarySegment": "Third",
        }
        assert segment_tilt_label(row) == "First"
        
        row_no_primary = {
            "dominant_audience_segment": "Second",
            "PredictedPrimarySegment": "Third",
        }
        assert segment_tilt_label(row_no_primary) == "Second"


class TestBuildSeasonSummary:
    """Test the build_season_summary function."""

    def test_returns_dataframe_with_correct_columns(self):
        """Test that output has all expected columns."""
        from streamlit_app import build_season_summary
        
        test_df = pd.DataFrame([{
            "Month": "September 2025",
            "Title": "Swan Lake",
            "Category": "classic_romance",
            "EstimatedTickets_Final": 5000,
            "YYC_Singles": 3000,
            "YEG_Singles": 2000,
            "Total_Mkt_Spend": 50000,
            "TicketIndex used": 95,
            "PrimarySegment": "Core Classical (F35–64)",
        }])
        
        result = build_season_summary(test_df)
        
        expected_cols = [
            "Month", "Show Title", "Category", "Estimated Tickets",
            "YYC Singles", "YEG Singles", "Total Marketing Spend",
            "Segment Tilt", "Index Strength"
        ]
        assert list(result.columns) == expected_cols

    def test_empty_dataframe_returns_empty_with_correct_columns(self):
        """Test that empty input returns empty DataFrame with correct columns."""
        from streamlit_app import build_season_summary
        
        result = build_season_summary(pd.DataFrame())
        
        assert result.empty
        assert "Month" in result.columns
        assert "Index Strength" in result.columns

    def test_month_extraction_from_full_format(self):
        """Test month extraction from 'September 2025' format."""
        from streamlit_app import build_season_summary
        
        test_df = pd.DataFrame([{
            "Month": "October 2025",
            "Title": "Test Show",
            "Category": "family_classic",
            "EstimatedTickets_Final": 1000,
            "YYC_Singles": 600,
            "YEG_Singles": 400,
        }])
        
        result = build_season_summary(test_df)
        assert result.iloc[0]["Month"] == "October"

    def test_calendar_order_sorting(self):
        """Test that rows are sorted by calendar month order."""
        from streamlit_app import build_season_summary
        
        test_df = pd.DataFrame([
            {"Month": "March 2026", "Title": "Show C", "Category": "c"},
            {"Month": "September 2025", "Title": "Show A", "Category": "a"},
            {"Month": "January 2026", "Title": "Show B", "Category": "b"},
        ])
        
        result = build_season_summary(test_df)
        months = result["Month"].tolist()
        
        assert months == ["September", "January", "March"]

    def test_estimated_tickets_calculation(self):
        """Test ticket calculation with fallback to YYC + YEG."""
        from streamlit_app import build_season_summary
        
        # With EstimatedTickets_Final
        test_df = pd.DataFrame([{
            "Month": "September 2025",
            "Title": "Test",
            "EstimatedTickets_Final": 5000,
            "YYC_Singles": 3000,
            "YEG_Singles": 2000,
        }])
        result = build_season_summary(test_df)
        assert result.iloc[0]["Estimated Tickets"] == 5000
        
        # Without EstimatedTickets_Final (fallback)
        test_df_fallback = pd.DataFrame([{
            "Month": "September 2025",
            "Title": "Test",
            "YYC_Singles": 3000,
            "YEG_Singles": 2000,
        }])
        result_fallback = build_season_summary(test_df_fallback)
        assert result_fallback.iloc[0]["Estimated Tickets"] == 5000

    def test_marketing_spend_formatting(self):
        """Test that marketing spend is formatted with dollar sign."""
        from streamlit_app import build_season_summary
        
        test_df = pd.DataFrame([{
            "Month": "September 2025",
            "Title": "Test",
            "Total_Mkt_Spend": 50000,
        }])
        
        result = build_season_summary(test_df)
        assert result.iloc[0]["Total Marketing Spend"] == "$50,000"

    def test_index_strength_conversion(self):
        """Test that ticket index is converted to star rating."""
        from streamlit_app import build_season_summary
        
        test_df = pd.DataFrame([{
            "Month": "September 2025",
            "Title": "Test",
            "TicketIndex used": 115,
        }])
        
        result = build_season_summary(test_df)
        assert result.iloc[0]["Index Strength"] == "★★★★☆"


class TestMakeSeasonSummaryTablePdf:
    """Test the PDF table generation for Season Summary."""

    def test_pdf_table_creation(self):
        """Test that PDF table is created from plan_df."""
        from streamlit_app import _make_season_summary_table_pdf
        
        test_df = pd.DataFrame([{
            "Month": "September 2025",
            "Title": "Test Show",
            "Category": "family_classic",
            "EstimatedTickets_Final": 1000,
            "YYC_Singles": 600,
            "YEG_Singles": 400,
        }])
        
        table = _make_season_summary_table_pdf(test_df)
        
        # ReportLab Table object should be returned
        from reportlab.platypus import Table
        assert isinstance(table, Table)

    def test_empty_df_returns_placeholder(self):
        """Test that empty DataFrame returns placeholder table."""
        from streamlit_app import _make_season_summary_table_pdf
        from reportlab.platypus import Table
        
        table = _make_season_summary_table_pdf(pd.DataFrame())
        
        assert isinstance(table, Table)


class TestBuildFullPdfReport:
    """Test the PDF report includes Season Summary."""

    def test_pdf_report_includes_season_summary(self):
        """Test that the PDF report contains Season Summary section."""
        from streamlit_app import build_full_pdf_report, _methodology_glossary_text
        
        test_df = pd.DataFrame([{
            "Month": "September 2025",
            "Title": "Test Show",
            "Category": "family_classic",
            "EstimatedTickets_Final": 1000,
            "YYC_Singles": 600,
            "YEG_Singles": 400,
            "TicketIndex used": 95,
            "FutureSeasonalityFactor": 1.05,
            "ReturnDecayPct": 0.0,
            "CityShare_Calgary": 0.6,
            "CityShare_Edmonton": 0.4,
            "Total_Mkt_Spend": 10000,
            "PrimarySegment": "Family",
        }])
        
        methodology = _methodology_glossary_text()
        
        pdf_bytes = build_full_pdf_report(
            methodology_paragraphs=methodology,
            plan_df=test_df,
            season_year=2025,
            org_name="Test Ballet"
        )
        
        # PDF should be generated successfully
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0
        # Check PDF header
        assert pdf_bytes[:4] == b"%PDF"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
