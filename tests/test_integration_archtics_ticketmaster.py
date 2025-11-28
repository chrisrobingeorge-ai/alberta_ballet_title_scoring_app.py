"""
Tests for Archtics + Ticketmaster Integration

These tests cover:
- Data normalization and field mapping
- City detection and splits
- Metric calculations (load factor, averages)
- CSV export with correct column order
- Stub/fixture tests for API responses
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.normalizer import (
    NormalizedShowData,
    ShowDataNormalizer,
    get_normalized_columns,
    NORMALIZED_COLUMNS,
)
from integrations.csv_exporter import (
    export_show_csv,
    validate_csv_schema,
    get_export_stats,
    _sanitize_filename,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_normalized_data():
    """Sample normalized show data for testing."""
    return NormalizedShowData(
        show_title="The Nutcracker",
        show_title_id="nutcracker-2024",
        production_season="2024-25",
        city="Calgary",
        venue_name="Jubilee Auditorium Calgary",
        venue_capacity=2500,
        performance_count_city=8,
        performance_count_total=16,
        single_tickets_calgary=12000,
        single_tickets_edmonton=8000,
        subscription_tickets_calgary=5000,
        subscription_tickets_edmonton=3500,
        total_single_tickets=20000,
        total_subscription_tickets=8500,
        total_tickets_all=28500,
        avg_tickets_per_performance=1781.25,
        load_factor=0.7125,
        channel_mix_distribution="web:15000,phone:8000,walkup:5500",
        comp_ticket_share=0.02,
        refund_cancellation_rate=0.015,
        pricing_tier_structure="standard:45-95,premium:80-150",
        average_base_ticket_price=87.50,
        opening_date="2024-12-01",
        closing_date="2024-12-24",
        weekday_vs_weekend_mix="weekday:10,weekend:6",
    )


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# =============================================================================
# Normalizer Tests
# =============================================================================

class TestShowDataNormalizer:
    """Tests for the ShowDataNormalizer class."""
    
    def test_normalizer_initialization(self):
        """Test normalizer initializes correctly."""
        normalizer = ShowDataNormalizer()
        assert normalizer is not None
    
    def test_normalize_empty_data(self):
        """Test normalization with no data sources."""
        normalizer = ShowDataNormalizer()
        result = normalizer.normalize(
            show_title="Test Show",
            show_id="test-show",
        )
        
        assert result.show_title == "Test Show"
        assert result.show_title_id == "test-show"
        assert result.total_tickets_all == 0
        assert not result.source_tm
        assert not result.source_archtics
        assert len(result.warnings) > 0  # Should have missing data warnings
    
    def test_city_detection_calgary(self):
        """Test city detection for Calgary patterns."""
        normalizer = ShowDataNormalizer()
        
        assert normalizer._detect_city("Calgary") == "Calgary"
        assert normalizer._detect_city("CALGARY") == "Calgary"
        assert normalizer._detect_city("Jubilee Auditorium Calgary") == "Calgary"
        assert normalizer._detect_city("YYC Centre") == "Calgary"
    
    def test_city_detection_edmonton(self):
        """Test city detection for Edmonton patterns."""
        normalizer = ShowDataNormalizer()
        
        assert normalizer._detect_city("Edmonton") == "Edmonton"
        assert normalizer._detect_city("EDMONTON") == "Edmonton"
        assert normalizer._detect_city("Jubilee Auditorium Edmonton") == "Edmonton"
        assert normalizer._detect_city("YEG Arena") == "Edmonton"
    
    def test_city_detection_none(self):
        """Test city detection returns None for unknown locations."""
        normalizer = ShowDataNormalizer()
        
        assert normalizer._detect_city("Vancouver") is None
        assert normalizer._detect_city("Toronto") is None
        assert normalizer._detect_city("") is None
        assert normalizer._detect_city(None) is None
    
    def test_weekday_weekend_mix_calculation(self):
        """Test weekday/weekend mix calculation."""
        normalizer = ShowDataNormalizer()
        
        # Monday is 0, so 2024-12-02 is Monday, 2024-12-07 is Saturday
        dates = ["2024-12-02", "2024-12-03", "2024-12-04", "2024-12-07", "2024-12-08"]
        result = normalizer._calculate_weekday_weekend_mix(dates)
        
        assert "weekday:3" in result
        assert "weekend:2" in result
    
    def test_serialize_dict(self):
        """Test dictionary serialization."""
        normalizer = ShowDataNormalizer()
        
        data = {"web": 100, "phone": 50, "walkup": 25}
        result = normalizer._serialize_dict(data)
        
        assert "phone:50" in result
        assert "walkup:25" in result
        assert "web:100" in result
    
    def test_calculate_avg_price(self):
        """Test average price calculation."""
        normalizer = ShowDataNormalizer()
        
        price_ranges = [
            {"min": 50, "max": 100},
            {"min": 80, "max": 120},
        ]
        result = normalizer._calculate_avg_price(price_ranges)
        
        # (75 + 100) / 2 = 87.5
        assert result == 87.5
    
    def test_calculate_avg_price_empty(self):
        """Test average price with no data."""
        normalizer = ShowDataNormalizer()
        
        assert normalizer._calculate_avg_price([]) is None
        assert normalizer._calculate_avg_price([{}]) is None


class TestNormalizedShowData:
    """Tests for the NormalizedShowData dataclass."""
    
    def test_to_dict(self, sample_normalized_data):
        """Test conversion to dictionary."""
        result = sample_normalized_data.to_dict()
        
        assert isinstance(result, dict)
        assert result["show_title"] == "The Nutcracker"
        assert result["total_tickets_all"] == 28500
        assert result["load_factor"] == 0.7125
    
    def test_to_dict_has_all_columns(self, sample_normalized_data):
        """Test that to_dict returns all expected columns."""
        result = sample_normalized_data.to_dict()
        
        for col in NORMALIZED_COLUMNS:
            assert col in result, f"Missing column: {col}"


# =============================================================================
# CSV Exporter Tests
# =============================================================================

class TestCSVExporter:
    """Tests for CSV export functionality."""
    
    def test_export_single_record(self, sample_normalized_data, temp_output_dir):
        """Test exporting a single normalized record."""
        output_path = os.path.join(temp_output_dir, "test_export.csv")
        
        result_path = export_show_csv(
            sample_normalized_data,
            output_path=output_path,
        )
        
        assert result_path == output_path
        assert os.path.exists(output_path)
        
        # Validate the file
        is_valid, errors = validate_csv_schema(output_path)
        assert is_valid, f"Schema validation failed: {errors}"
    
    def test_export_with_auto_filename(self, sample_normalized_data, temp_output_dir):
        """Test auto-generated filename from show_id."""
        result_path = export_show_csv(
            sample_normalized_data,
            output_dir=temp_output_dir,
        )
        
        # Check the filename follows the expected pattern
        from pathlib import Path
        filename = Path(result_path).name
        assert filename.endswith("_archtics_ticketmaster.csv")
        assert "nutcracker" in filename.lower()
        assert os.path.exists(result_path)
    
    def test_export_multiple_records(self, sample_normalized_data, temp_output_dir):
        """Test exporting multiple records."""
        # Create a second record
        second = NormalizedShowData(
            show_title="Swan Lake",
            show_title_id="swan-lake-2024",
            production_season="2024-25",
            total_tickets_all=15000,
        )
        
        output_path = os.path.join(temp_output_dir, "multi_export.csv")
        result_path = export_show_csv(
            [sample_normalized_data, second],
            output_path=output_path,
        )
        
        assert os.path.exists(output_path)
        
        # Check row count
        stats = get_export_stats(output_path)
        assert stats["row_count"] == 2
    
    def test_column_order(self, sample_normalized_data, temp_output_dir):
        """Test that CSV columns are in the correct order."""
        output_path = os.path.join(temp_output_dir, "column_order_test.csv")
        export_show_csv(sample_normalized_data, output_path=output_path)
        
        import csv
        with open(output_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
        
        assert header == NORMALIZED_COLUMNS
    
    def test_validate_schema_missing_columns(self, temp_output_dir):
        """Test schema validation catches missing columns."""
        # Create a CSV with missing columns
        output_path = os.path.join(temp_output_dir, "incomplete.csv")
        with open(output_path, "w") as f:
            f.write("show_title,show_title_id\n")
            f.write("Test,test-123\n")
        
        is_valid, errors = validate_csv_schema(output_path)
        
        assert not is_valid
        assert any("Missing columns" in e for e in errors)
    
    def test_get_export_stats(self, sample_normalized_data, temp_output_dir):
        """Test export statistics collection."""
        output_path = os.path.join(temp_output_dir, "stats_test.csv")
        export_show_csv(sample_normalized_data, output_path=output_path)
        
        stats = get_export_stats(output_path)
        
        assert stats["exists"]
        assert stats["row_count"] == 1
        assert stats["column_count"] == len(NORMALIZED_COLUMNS)
        assert stats["file_size_bytes"] > 0


class TestSanitizeFilename:
    """Tests for filename sanitization."""
    
    def test_spaces_replaced(self):
        """Test spaces are replaced with underscores."""
        assert _sanitize_filename("The Nutcracker") == "the_nutcracker"
    
    def test_special_chars_removed(self):
        """Test special characters are removed."""
        assert _sanitize_filename("Show: 2024") == "show_2024"
        assert _sanitize_filename("Test<>Name") == "testname"
    
    def test_lowercase(self):
        """Test result is lowercase."""
        assert _sanitize_filename("UPPERCASE") == "uppercase"
    
    def test_multiple_underscores_collapsed(self):
        """Test multiple underscores are collapsed."""
        assert _sanitize_filename("a - b - c") == "a_b_c"


# =============================================================================
# Column Schema Tests
# =============================================================================

class TestNormalizedColumns:
    """Tests for the normalized column schema."""
    
    def test_get_normalized_columns_returns_list(self):
        """Test get_normalized_columns returns a list."""
        columns = get_normalized_columns()
        assert isinstance(columns, list)
        assert len(columns) > 0
    
    def test_expected_columns_present(self):
        """Test all expected columns are in the schema."""
        columns = get_normalized_columns()
        
        expected = [
            "show_title",
            "show_title_id",
            "production_season",
            "city",
            "venue_name",
            "venue_capacity",
            "total_tickets_all",
            "load_factor",
            "opening_date",
            "closing_date",
        ]
        
        for col in expected:
            assert col in columns, f"Missing expected column: {col}"
    
    def test_column_order_is_deterministic(self):
        """Test column order is the same each time."""
        columns1 = get_normalized_columns()
        columns2 = get_normalized_columns()
        
        assert columns1 == columns2


# =============================================================================
# Metric Calculation Tests
# =============================================================================

class TestMetricCalculations:
    """Tests for metric calculations during normalization."""
    
    def test_avg_tickets_per_performance(self):
        """Test average tickets per performance calculation."""
        data = NormalizedShowData(
            show_title="Test",
            show_title_id="test",
            total_tickets_all=1000,
            performance_count_total=10,
        )
        
        normalizer = ShowDataNormalizer()
        normalizer._calculate_metrics(data)
        
        assert data.avg_tickets_per_performance == 100.0
    
    def test_load_factor_calculation(self):
        """Test load factor calculation."""
        data = NormalizedShowData(
            show_title="Test",
            show_title_id="test",
            total_tickets_all=2000,
            venue_capacity=500,
            performance_count_total=5,
        )
        
        normalizer = ShowDataNormalizer()
        normalizer._calculate_metrics(data)
        
        # 2000 / (500 * 5) = 0.8
        assert data.load_factor == 0.8
    
    def test_load_factor_zero_capacity(self):
        """Test load factor with zero capacity."""
        data = NormalizedShowData(
            show_title="Test",
            show_title_id="test",
            total_tickets_all=1000,
            venue_capacity=0,
            performance_count_total=5,
        )
        
        normalizer = ShowDataNormalizer()
        normalizer._calculate_metrics(data)
        
        assert data.load_factor is None
    
    def test_comp_ticket_share(self, sample_normalized_data):
        """Test comp ticket share is a valid ratio."""
        assert 0 <= sample_normalized_data.comp_ticket_share <= 1
    
    def test_refund_cancellation_rate(self, sample_normalized_data):
        """Test refund/cancellation rate is a valid ratio."""
        assert 0 <= sample_normalized_data.refund_cancellation_rate <= 1


# =============================================================================
# City Split Tests
# =============================================================================

class TestCitySplits:
    """Tests for Calgary/Edmonton ticket splits."""
    
    def test_city_totals_match(self, sample_normalized_data):
        """Test that city splits add up to totals."""
        assert (
            sample_normalized_data.single_tickets_calgary +
            sample_normalized_data.single_tickets_edmonton
        ) == sample_normalized_data.total_single_tickets
        
        assert (
            sample_normalized_data.subscription_tickets_calgary +
            sample_normalized_data.subscription_tickets_edmonton
        ) == sample_normalized_data.total_subscription_tickets
    
    def test_total_tickets_all(self, sample_normalized_data):
        """Test total tickets calculation."""
        expected = (
            sample_normalized_data.total_single_tickets +
            sample_normalized_data.total_subscription_tickets
        )
        assert sample_normalized_data.total_tickets_all == expected


# =============================================================================
# Stub/Fixture Tests
# =============================================================================

class TestStubPayloads:
    """Tests using stub API payloads."""
    
    @pytest.fixture
    def calgary_stub_data(self):
        """Stub payload for Calgary scenario."""
        return {
            "event_name": "The Nutcracker",
            "venue": "Jubilee Auditorium Calgary",
            "city": "Calgary",
            "performances": 8,
            "single_tickets": 12000,
            "subscription_tickets": 5000,
            "comp_tickets": 250,
            "refunds": 150,
            "venue_capacity": 2500,
        }
    
    @pytest.fixture
    def edmonton_stub_data(self):
        """Stub payload for Edmonton scenario."""
        return {
            "event_name": "Swan Lake",
            "venue": "Jubilee Auditorium Edmonton",
            "city": "Edmonton",
            "performances": 6,
            "single_tickets": 8500,
            "subscription_tickets": 3200,
            "comp_tickets": 180,
            "refunds": 95,
            "venue_capacity": 2500,
        }
    
    def test_calgary_scenario_normalization(self, calgary_stub_data):
        """Test normalization with Calgary stub data."""
        # Create a mock sales summary-like object
        class MockSales:
            single_tickets_sold = calgary_stub_data["single_tickets"]
            subscription_tickets_sold = calgary_stub_data["subscription_tickets"]
            comp_tickets = calgary_stub_data["comp_tickets"]
            refunds = calgary_stub_data["refunds"]
            cancellations = 0
            channel_mix = {"web": 8000, "phone": 4000}
            price_tier_breakdown = {}
        
        normalizer = ShowDataNormalizer()
        result = normalizer.normalize(
            show_title=calgary_stub_data["event_name"],
            show_id="nutcracker-calgary-2024",
            archtics_sales=MockSales(),
        )
        
        # Verify Calgary gets the single tickets since no TM events to detect city
        # With default split (60/40), Calgary should get 60%
        assert result.single_tickets_calgary > 0
        assert result.total_single_tickets == calgary_stub_data["single_tickets"]
    
    def test_edmonton_scenario_normalization(self, edmonton_stub_data):
        """Test normalization with Edmonton stub data."""
        class MockSales:
            single_tickets_sold = edmonton_stub_data["single_tickets"]
            subscription_tickets_sold = edmonton_stub_data["subscription_tickets"]
            comp_tickets = edmonton_stub_data["comp_tickets"]
            refunds = edmonton_stub_data["refunds"]
            cancellations = 0
            channel_mix = {}
            price_tier_breakdown = {}
        
        normalizer = ShowDataNormalizer()
        result = normalizer.normalize(
            show_title=edmonton_stub_data["event_name"],
            show_id="swan-lake-edmonton-2024",
            archtics_sales=MockSales(),
        )
        
        assert result.total_single_tickets == edmonton_stub_data["single_tickets"]
        assert result.total_subscription_tickets == edmonton_stub_data["subscription_tickets"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
