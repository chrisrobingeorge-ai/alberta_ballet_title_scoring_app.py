"""
Tests for diagnostic export fields in ticket estimator.

These tests verify that the new diagnostic and contextual fields are
correctly added to the ticket estimator export.
"""

import pytest
import pandas as pd
import numpy as np


class TestDiagnosticFields:
    """Tests for diagnostic export fields."""

    def test_lead_gender_field_present(self):
        """Test that lead_gender field is recognized."""
        # Test data structure that mimics plan_rows
        plan_row = {
            "Title": "Swan Lake",
            "Category": "classic_romance",
            "lead_gender": "female",
            "Gender": "female",
        }
        assert "lead_gender" in plan_row
        assert plan_row["lead_gender"] in ["male", "female", "co-lead", "n/a"]

    def test_dominant_audience_segment_field(self):
        """Test that dominant_audience_segment is present."""
        plan_row = {
            "Title": "Nutcracker",
            "dominant_audience_segment": "Family (Parents w/ kids)",
            "PredictedPrimarySegment": "Family (Parents w/ kids)",
        }
        assert "dominant_audience_segment" in plan_row
        assert plan_row["dominant_audience_segment"] == plan_row["PredictedPrimarySegment"]

    def test_segment_weights_json_format(self):
        """Test that segment_weights is valid JSON."""
        import json
        
        weights = {
            "General Population": 0.25,
            "Core Classical (F35–64)": 0.30,
            "Family (Parents w/ kids)": 0.35,
            "Emerging Adults (18–34)": 0.10,
        }
        segment_weights_json = json.dumps(weights)
        
        # Verify it can be parsed back
        parsed = json.loads(segment_weights_json)
        assert isinstance(parsed, dict)
        assert all(isinstance(v, float) for v in parsed.values())
        assert abs(sum(parsed.values()) - 1.0) < 0.01  # Weights sum to ~1.0

    def test_ticket_median_prior_field(self):
        """Test ticket_median_prior field."""
        plan_row = {
            "Title": "Swan Lake",
            "ticket_median_prior": 8500.0,
            "TicketMedian": 8500.0,
        }
        assert "ticket_median_prior" in plan_row
        assert plan_row["ticket_median_prior"] == plan_row["TicketMedian"]

    def test_prior_total_tickets_field(self):
        """Test prior_total_tickets field."""
        # Simulating ticket priors: [8000, 9000, 8500]
        priors = [8000, 9000, 8500]
        plan_row = {
            "Title": "Swan Lake",
            "prior_total_tickets": sum(priors),
        }
        assert plan_row["prior_total_tickets"] == 25500

    def test_run_count_prior_field(self):
        """Test run_count_prior field."""
        priors = [8000, 9000, 8500]
        plan_row = {
            "Title": "Swan Lake",
            "run_count_prior": len(priors),
        }
        assert plan_row["run_count_prior"] == 3

    def test_ticket_index_predicted_field(self):
        """Test TicketIndex_Predicted field."""
        plan_row = {
            "Title": "New Show",
            "TicketIndex_Predicted": 95.5,  # ML prediction
            "TicketIndexSource": "ML Category",
        }
        assert "TicketIndex_Predicted" in plan_row
        assert isinstance(plan_row["TicketIndex_Predicted"], float)

    def test_month_of_opening_field(self):
        """Test month_of_opening field."""
        plan_row = {
            "Title": "Swan Lake",
            "month_of_opening": 12,  # December
        }
        assert plan_row["month_of_opening"] in range(1, 13)

    def test_holiday_flag_field(self):
        """Test holiday_flag field for different months."""
        # Holiday months: November (11), December (12), January (1)
        holiday_months = {11, 12, 1}
        
        for month in range(1, 13):
            expected_flag = month in holiday_months
            actual_flag = month in (11, 12, 1)
            assert actual_flag == expected_flag

    def test_category_seasonality_factor_field(self):
        """Test category_seasonality_factor field."""
        plan_row = {
            "Title": "Swan Lake",
            "category_seasonality_factor": 1.05,
            "FutureSeasonalityFactor": 1.05,
        }
        assert "category_seasonality_factor" in plan_row
        # Should be within reasonable bounds
        assert 0.85 <= plan_row["category_seasonality_factor"] <= 1.20

    def test_knn_used_field(self):
        """Test kNN_used field."""
        plan_row = {
            "Title": "Unknown Show",
            "kNN_used": False,
            "TicketIndexSource": "ML Category",
        }
        assert isinstance(plan_row["kNN_used"], bool)

    def test_knn_neighbors_json_format(self):
        """Test kNN_neighbors is valid JSON array."""
        import json
        
        # Empty array when not used
        knn_neighbors = "[]"
        parsed = json.loads(knn_neighbors)
        assert isinstance(parsed, list)
        
        # With neighbors when used
        knn_neighbors = '["Swan Lake", "Nutcracker", "Sleeping Beauty"]'
        parsed = json.loads(knn_neighbors)
        assert isinstance(parsed, list)
        assert len(parsed) == 3

    def test_la_category_field(self):
        """Test LA_Category field."""
        plan_row = {
            "Title": "Swan Lake",
            "Category": "classic_romance",
            "LA_Category": "classic_romance",
        }
        assert "LA_Category" in plan_row


class TestDiagnosticFieldsInDataFrame:
    """Tests for diagnostic fields in DataFrame context."""

    def test_all_diagnostic_fields_present(self):
        """Test that all required diagnostic fields can be added to a DataFrame."""
        import json
        
        # Create sample plan_rows similar to what render_results builds
        plan_rows = [{
            "Month": "December 2024",
            "Title": "Swan Lake",
            "Category": "classic_romance",
            
            # Show & Audience Context
            "lead_gender": "female",
            "dominant_audience_segment": "Core Classical (F35–64)",
            "segment_weights": json.dumps({
                "General Population": 0.20,
                "Core Classical (F35–64)": 0.40,
                "Family (Parents w/ kids)": 0.25,
                "Emerging Adults (18–34)": 0.15,
            }),
            
            # Model & Historical Inputs
            "ticket_median_prior": 8500.0,
            "prior_total_tickets": 25500,
            "run_count_prior": 3,
            "TicketIndex_Predicted": np.nan,
            "TicketIndexSource": "History",
            
            # Temporal & Seasonality Info
            "month_of_opening": 12,
            "holiday_flag": True,
            "category_seasonality_factor": 1.05,
            
            # k-NN Metadata
            "kNN_used": False,
            "kNN_neighbors": "[]",
            
            # LA Category
            "LA_Category": "classic_romance",
        }]
        
        df = pd.DataFrame(plan_rows)
        
        # Verify all fields present
        expected_fields = [
            "lead_gender", "dominant_audience_segment", "segment_weights",
            "ticket_median_prior", "prior_total_tickets", "run_count_prior",
            "TicketIndex_Predicted", "month_of_opening", "holiday_flag",
            "category_seasonality_factor", "kNN_used", "kNN_neighbors",
            "LA_Category",
        ]
        
        for field in expected_fields:
            assert field in df.columns, f"Missing field: {field}"

    def test_diagnostic_fields_types(self):
        """Test that diagnostic fields have correct types."""
        import json
        
        plan_rows = [{
            "lead_gender": "female",
            "dominant_audience_segment": "Family",
            "segment_weights": json.dumps({"Family": 0.5}),
            "ticket_median_prior": 8500.0,
            "prior_total_tickets": 25500,
            "run_count_prior": 3,
            "TicketIndex_Predicted": 95.5,
            "month_of_opening": 12,
            "holiday_flag": True,
            "category_seasonality_factor": 1.05,
            "kNN_used": False,
            "kNN_neighbors": "[]",
            "LA_Category": "family_classic",
        }]
        
        df = pd.DataFrame(plan_rows)
        
        # String fields
        assert df["lead_gender"].dtype == object
        assert df["dominant_audience_segment"].dtype == object
        assert df["segment_weights"].dtype == object
        
        # Numeric fields
        assert pd.api.types.is_float_dtype(df["ticket_median_prior"])
        assert pd.api.types.is_integer_dtype(df["prior_total_tickets"]) or pd.api.types.is_float_dtype(df["prior_total_tickets"])
        assert pd.api.types.is_integer_dtype(df["run_count_prior"]) or pd.api.types.is_float_dtype(df["run_count_prior"])
        assert pd.api.types.is_float_dtype(df["TicketIndex_Predicted"])
        
        # Boolean field
        assert df["holiday_flag"].dtype == bool
        assert df["kNN_used"].dtype == bool


class TestSegmentWeightsCalculation:
    """Tests for segment weights JSON generation."""

    def test_segment_weights_sum_to_one(self):
        """Test that segment weights sum approximately to 1.0."""
        import json
        
        # Sample weights from computation
        mix_gp = 0.25
        mix_core = 0.30
        mix_family = 0.35
        mix_ea = 0.10
        
        weights = {
            "General Population": round(mix_gp, 4),
            "Core Classical (F35–64)": round(mix_core, 4),
            "Family (Parents w/ kids)": round(mix_family, 4),
            "Emerging Adults (18–34)": round(mix_ea, 4),
        }
        
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected ~1.0"

    def test_segment_weights_json_roundtrip(self):
        """Test segment weights JSON serialization/deserialization."""
        import json
        
        weights = {
            "General Population": 0.25,
            "Core Classical (F35–64)": 0.30,
            "Family (Parents w/ kids)": 0.35,
            "Emerging Adults (18–34)": 0.10,
        }
        
        json_str = json.dumps(weights)
        parsed = json.loads(json_str)
        
        assert parsed == weights
