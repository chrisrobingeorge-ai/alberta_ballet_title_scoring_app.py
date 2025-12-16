"""
Test Suite for Intent Disambiguation Correction Module

Tests the penalty application logic, ticket recalculation, and edge cases.

Author: Alberta Ballet Data Science Team
Date: December 2025
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.intent_disambiguation import (
    apply_intent_disambiguation,
    get_penalty_for_category,
    get_all_penalty_categories,
    batch_apply_corrections,
    CATEGORY_PENALTIES,
    MOTIVATION_INDEX_WEIGHT
)


class TestPenaltyRetrieval:
    """Test penalty lookup functions."""
    
    def test_get_penalty_for_category_high_ambiguity(self):
        """Test retrieval of 20% penalties."""
        assert get_penalty_for_category("family_classic") == 0.20
        assert get_penalty_for_category("pop_ip") == 0.20
        assert get_penalty_for_category("classic_romance") == 0.20
        assert get_penalty_for_category("classic_comedy") == 0.20
    
    def test_get_penalty_for_category_moderate_ambiguity(self):
        """Test retrieval of 10% penalties."""
        assert get_penalty_for_category("romantic_tragedy") == 0.10
        assert get_penalty_for_category("adult_literary_drama") == 0.10
    
    def test_get_penalty_for_category_no_penalty(self):
        """Test categories with no penalty."""
        assert get_penalty_for_category("contemporary") == 0.0
        assert get_penalty_for_category("nutcracker") == 0.0
        assert get_penalty_for_category("unknown_category") == 0.0
    
    def test_get_penalty_case_insensitive(self):
        """Test that category lookup is case-insensitive."""
        assert get_penalty_for_category("FAMILY_CLASSIC") == 0.20
        assert get_penalty_for_category("Family_Classic") == 0.20
        assert get_penalty_for_category("Romantic_Tragedy") == 0.10
    
    def test_get_all_penalty_categories(self):
        """Test retrieving all penalty categories."""
        penalties = get_all_penalty_categories()
        assert len(penalties) == 6
        assert penalties["family_classic"] == 0.20
        assert penalties["romantic_tragedy"] == 0.10


class TestBasicCorrection:
    """Test basic correction functionality."""
    
    def test_cinderella_example(self):
        """Test the canonical Cinderella example from the prompt."""
        metadata = {
            "Title": "Cinderella",
            "Category": "family_classic",
            "Motivation": 100.0,
            "TicketIndex used": 100.0,
            "EstimatedTickets_Final": 11976
        }
        
        result = apply_intent_disambiguation(metadata)
        
        # Check Motivation correction: 100 * (1 - 0.20) = 80
        assert result["Motivation_corrected"] == 80.0
        assert result["Motivation_penalty_applied"] is True
        assert result["Motivation_penalty_pct"] == 0.20
        
        # Check TicketIndex correction
        # ΔMotivation = 80 - 100 = -20
        # ΔIndex = -20 * (1/6) = -3.333...
        # New Index = 100 + (-3.333) = 96.666...
        expected_delta_index = -20.0 * MOTIVATION_INDEX_WEIGHT
        expected_index = 100.0 + expected_delta_index
        assert abs(result["TicketIndex_corrected"] - expected_index) < 0.01
        
        # Check EstimatedTickets correction
        # k = 11976 / 100 = 119.76
        # New Tickets = 96.666... * 119.76 = 11580.8
        k = 11976 / 100.0
        expected_tickets = expected_index * k
        assert abs(result["EstimatedTickets_corrected"] - expected_tickets) < 1.0
        
        # Check flags
        assert result["IntentCorrectionApplied"] is True
    
    def test_romantic_tragedy_10_percent(self):
        """Test 10% penalty for romantic tragedy."""
        metadata = {
            "Title": "Romeo & Juliet",
            "Category": "romantic_tragedy",
            "Motivation": 90.0,
            "TicketIndex used": 95.0,
            "EstimatedTickets_Final": 10000
        }
        
        result = apply_intent_disambiguation(metadata)
        
        # Motivation: 90 * (1 - 0.10) = 81
        assert result["Motivation_corrected"] == 81.0
        assert result["Motivation_penalty_pct"] == 0.10
        
        # ΔMotivation = -9, ΔIndex = -9 * (1/6) = -1.5
        # New Index = 95 - 1.5 = 93.5
        expected_index = 95.0 + (-9.0 * MOTIVATION_INDEX_WEIGHT)
        assert abs(result["TicketIndex_corrected"] - expected_index) < 0.01
    
    def test_no_penalty_category(self):
        """Test that contemporary titles receive no penalty."""
        metadata = {
            "Title": "Modern Dance Showcase",
            "Category": "contemporary",
            "Motivation": 85.0,
            "TicketIndex used": 88.0,
            "EstimatedTickets_Final": 9500
        }
        
        result = apply_intent_disambiguation(metadata)
        
        # No changes expected
        assert result["Motivation_corrected"] == 85.0
        assert result["Motivation_penalty_applied"] is False
        assert result["Motivation_penalty_pct"] == 0.0
        assert result["TicketIndex_corrected"] == 88.0
        assert result["EstimatedTickets_corrected"] == 9500
        assert result["IntentCorrectionApplied"] is False
    
    def test_correction_disabled(self):
        """Test that apply_correction=False bypasses all penalties."""
        metadata = {
            "Title": "Cinderella",
            "Category": "family_classic",
            "Motivation": 100.0,
            "TicketIndex used": 100.0,
            "EstimatedTickets_Final": 11976
        }
        
        result = apply_intent_disambiguation(metadata, apply_correction=False)
        
        # Should return original values
        assert result["Motivation_corrected"] == 100.0
        assert result["Motivation_penalty_applied"] is False
        assert result["TicketIndex_corrected"] == 100.0
        assert result["EstimatedTickets_corrected"] == 11976
        assert result["IntentCorrectionApplied"] is False


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_motivation(self):
        """Test correction with zero Motivation."""
        metadata = {
            "Title": "Test",
            "Category": "family_classic",
            "Motivation": 0.0,
            "TicketIndex used": 50.0,
            "EstimatedTickets_Final": 5000
        }
        
        result = apply_intent_disambiguation(metadata)
        
        # 0 * 0.8 = 0
        assert result["Motivation_corrected"] == 0.0
        # ΔMotivation = 0, ΔIndex = 0
        assert result["TicketIndex_corrected"] == 50.0
    
    def test_zero_ticket_index(self):
        """Test correction when original TicketIndex is zero."""
        metadata = {
            "Title": "Test",
            "Category": "family_classic",
            "Motivation": 100.0,
            "TicketIndex used": 0.0,
            "EstimatedTickets_Final": 0.0
        }
        
        result = apply_intent_disambiguation(metadata)
        
        # Motivation gets corrected
        assert result["Motivation_corrected"] == 80.0
        
        # Index calculation: 0 + (-20 * 1/6) = -3.33... but clamped to 0
        expected_index = max(0.0, 0.0 + (-20.0 * MOTIVATION_INDEX_WEIGHT))
        assert result["TicketIndex_corrected"] == expected_index
        
        # Tickets remain 0 (k = 0)
        assert result["EstimatedTickets_corrected"] == 0.0
    
    def test_missing_required_keys(self):
        """Test error handling for missing required keys."""
        metadata = {
            "Title": "Test",
            "Category": "family_classic",
            # Missing Motivation, TicketIndex used, EstimatedTickets_Final
        }
        
        with pytest.raises(KeyError) as exc_info:
            apply_intent_disambiguation(metadata)
        
        assert "Missing required keys" in str(exc_info.value)
    
    def test_invalid_negative_motivation(self):
        """Test error handling for negative Motivation."""
        metadata = {
            "Title": "Test",
            "Category": "family_classic",
            "Motivation": -10.0,
            "TicketIndex used": 100.0,
            "EstimatedTickets_Final": 10000
        }
        
        with pytest.raises(ValueError) as exc_info:
            apply_intent_disambiguation(metadata)
        
        assert "Motivation must be >= 0" in str(exc_info.value)
    
    def test_invalid_negative_index(self):
        """Test error handling for negative TicketIndex."""
        metadata = {
            "Title": "Test",
            "Category": "family_classic",
            "Motivation": 100.0,
            "TicketIndex used": -50.0,
            "EstimatedTickets_Final": 10000
        }
        
        with pytest.raises(ValueError) as exc_info:
            apply_intent_disambiguation(metadata)
        
        assert "TicketIndex must be >= 0" in str(exc_info.value)
    
    def test_very_high_motivation(self):
        """Test correction with very high Motivation values."""
        metadata = {
            "Title": "Viral Hit",
            "Category": "pop_ip",
            "Motivation": 250.0,
            "TicketIndex used": 180.0,
            "EstimatedTickets_Final": 20000
        }
        
        result = apply_intent_disambiguation(metadata)
        
        # 250 * 0.8 = 200
        assert result["Motivation_corrected"] == 200.0
        
        # ΔMotivation = -50, ΔIndex = -50 * (1/6) = -8.333...
        expected_index = 180.0 + (-50.0 * MOTIVATION_INDEX_WEIGHT)
        assert abs(result["TicketIndex_corrected"] - expected_index) < 0.01


class TestBatchProcessing:
    """Test batch correction functionality."""
    
    def test_batch_apply_corrections(self):
        """Test batch processing of multiple titles."""
        titles = [
            {
                "Title": "Cinderella",
                "Category": "family_classic",
                "Motivation": 100.0,
                "TicketIndex used": 100.0,
                "EstimatedTickets_Final": 11976
            },
            {
                "Title": "Romeo & Juliet",
                "Category": "romantic_tragedy",
                "Motivation": 90.0,
                "TicketIndex used": 95.0,
                "EstimatedTickets_Final": 10000
            },
            {
                "Title": "Contemporary Work",
                "Category": "contemporary",
                "Motivation": 75.0,
                "TicketIndex used": 80.0,
                "EstimatedTickets_Final": 8000
            }
        ]
        
        results = batch_apply_corrections(titles)
        
        assert len(results) == 3
        
        # First title: 20% penalty
        assert results[0]["Motivation_corrected"] == 80.0
        assert results[0]["Motivation_penalty_applied"] is True
        
        # Second title: 10% penalty
        assert results[1]["Motivation_corrected"] == 81.0
        assert results[1]["Motivation_penalty_applied"] is True
        
        # Third title: no penalty
        assert results[2]["Motivation_corrected"] == 75.0
        assert results[2]["Motivation_penalty_applied"] is False
    
    def test_batch_with_errors(self):
        """Test batch processing with some invalid titles."""
        titles = [
            {
                "Title": "Valid Title",
                "Category": "family_classic",
                "Motivation": 100.0,
                "TicketIndex used": 100.0,
                "EstimatedTickets_Final": 11976
            },
            {
                "Title": "Invalid Title",
                "Category": "family_classic",
                # Missing required fields
            },
            {
                "Title": "Another Valid",
                "Category": "contemporary",
                "Motivation": 80.0,
                "TicketIndex used": 85.0,
                "EstimatedTickets_Final": 9000
            }
        ]
        
        # Should warn but not crash
        with pytest.warns(UserWarning):
            results = batch_apply_corrections(titles)
        
        # All three titles should be in results
        assert len(results) == 3
        
        # First title should be corrected
        assert results[0]["Motivation_corrected"] == 80.0
        
        # Second title should have correction flags set to False
        assert results[1]["IntentCorrectionApplied"] is False
        
        # Third title should be unchanged (no penalty for contemporary)
        assert results[2]["Motivation_corrected"] == 80.0


class TestMetadataPreservation:
    """Test that original metadata is preserved in results."""
    
    def test_original_fields_preserved(self):
        """Test that all original metadata fields are preserved."""
        metadata = {
            "Title": "Cinderella",
            "Category": "family_classic",
            "Motivation": 100.0,
            "TicketIndex used": 100.0,
            "EstimatedTickets_Final": 11976,
            "Month": "March 2026",
            "PrimarySegment": "Family",
            "YYC_Singles": 7000,
            "YEG_Singles": 4976,
            "CustomField": "preserved"
        }
        
        result = apply_intent_disambiguation(metadata)
        
        # Check original fields are preserved
        assert result["Title"] == "Cinderella"
        assert result["Month"] == "March 2026"
        assert result["PrimarySegment"] == "Family"
        assert result["YYC_Singles"] == 7000
        assert result["YEG_Singles"] == 4976
        assert result["CustomField"] == "preserved"
        
        # Check new fields are added
        assert "Motivation_corrected" in result
        assert "TicketIndex_corrected" in result
        assert "EstimatedTickets_corrected" in result


class TestMathematicalPrecision:
    """Test mathematical precision of calculations."""
    
    def test_index_weight_constant(self):
        """Test that MOTIVATION_INDEX_WEIGHT is correctly set."""
        assert MOTIVATION_INDEX_WEIGHT == 1.0 / 6.0
        assert abs(MOTIVATION_INDEX_WEIGHT - 0.16666666666666666) < 1e-10
    
    def test_precise_calculation_chain(self):
        """Test precise calculation through the full chain."""
        metadata = {
            "Title": "Test",
            "Category": "family_classic",
            "Motivation": 120.0,
            "TicketIndex used": 115.0,
            "EstimatedTickets_Final": 13800
        }
        
        result = apply_intent_disambiguation(metadata)
        
        # Step 1: Motivation correction
        # 120 * (1 - 0.20) = 120 * 0.8 = 96.0
        assert result["Motivation_corrected"] == 96.0
        
        # Step 2: Index correction
        # ΔMotivation = 96 - 120 = -24
        # ΔIndex = -24 * (1/6) = -4.0
        # New Index = 115 - 4 = 111.0
        delta_motivation = 96.0 - 120.0
        delta_index = delta_motivation * (1.0 / 6.0)
        expected_index = 115.0 + delta_index
        
        assert abs(result["TicketIndex_corrected"] - expected_index) < 0.001
        assert abs(result["TicketIndex_corrected"] - 111.0) < 0.001
        
        # Step 3: Tickets correction
        # k = 13800 / 115 = 120.0
        # New Tickets = 111.0 * 120.0 = 13320.0
        k = 13800.0 / 115.0
        expected_tickets = expected_index * k
        
        assert abs(result["EstimatedTickets_corrected"] - expected_tickets) < 0.001
        assert abs(result["EstimatedTickets_corrected"] - 13320.0) < 0.001


if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([__file__, "-v", "--tb=short"])
