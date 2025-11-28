"""
Tests for batch processing functionality in pull_show_data.py

These tests cover:
- Reading show titles from CSV files
- Deduplication of show titles
- Show ID generation
- Batch processing result tracking
"""

import csv
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.pull_show_data import (
    BatchResult,
    BatchSummary,
    read_show_titles_from_csv,
    deduplicate_titles,
    generate_show_id,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file with show titles."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['show_title', 'Single Tickets - Calgary', 'Single Tickets - Edmonton'])
        writer.writerow(['The Nutcracker', '5000', '3000'])
        writer.writerow(['Swan Lake', '4000', '2500'])
        writer.writerow(['Cinderella', '6000', '4000'])
        writer.writerow(['The Nutcracker', '5500', '3200'])  # Duplicate
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_csv_file_alt_column():
    """Create a temporary CSV file with alternative column name 'Show Title'."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Show Title', 'Sales'])
        writer.writerow(['Romeo and Juliet', '8000'])
        writer.writerow(['Giselle', '5000'])
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_csv_file_title_column():
    """Create a temporary CSV file with column name 'title'."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['title', 'venue'])
        writer.writerow(['Don Quixote', 'Jubilee Auditorium'])
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_csv_invalid_columns():
    """Create a temporary CSV file with no recognizable show title column."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['production_name', 'venue'])
        writer.writerow(['Test Show', 'Some Venue'])
        f.flush()
        yield f.name
    os.unlink(f.name)


# =============================================================================
# Tests for read_show_titles_from_csv
# =============================================================================

class TestReadShowTitlesFromCSV:
    """Tests for reading show titles from CSV files."""
    
    def test_read_standard_column(self, temp_csv_file):
        """Test reading CSV with 'show_title' column."""
        titles = read_show_titles_from_csv(temp_csv_file)
        
        assert len(titles) == 4  # Includes duplicate
        assert 'The Nutcracker' in titles
        assert 'Swan Lake' in titles
        assert 'Cinderella' in titles
    
    def test_read_alt_column(self, temp_csv_file_alt_column):
        """Test reading CSV with 'Show Title' column."""
        titles = read_show_titles_from_csv(temp_csv_file_alt_column)
        
        assert len(titles) == 2
        assert 'Romeo and Juliet' in titles
        assert 'Giselle' in titles
    
    def test_read_title_column(self, temp_csv_file_title_column):
        """Test reading CSV with 'title' column."""
        titles = read_show_titles_from_csv(temp_csv_file_title_column)
        
        assert len(titles) == 1
        assert 'Don Quixote' in titles
    
    def test_file_not_found(self):
        """Test handling of missing CSV file."""
        with pytest.raises(FileNotFoundError):
            read_show_titles_from_csv("/nonexistent/path.csv")
    
    def test_invalid_columns(self, temp_csv_invalid_columns):
        """Test handling of CSV with no recognizable show title column."""
        with pytest.raises(ValueError) as exc_info:
            read_show_titles_from_csv(temp_csv_invalid_columns)
        
        assert "Could not find show title column" in str(exc_info.value)


# =============================================================================
# Tests for deduplicate_titles
# =============================================================================

class TestDeduplicateTitles:
    """Tests for title deduplication."""
    
    def test_no_duplicates(self):
        """Test with no duplicates."""
        titles = ['Swan Lake', 'Giselle', 'The Nutcracker']
        unique, count = deduplicate_titles(titles)
        
        assert len(unique) == 3
        assert count == 0
        assert unique == titles
    
    def test_with_duplicates(self):
        """Test with duplicate titles."""
        titles = ['Swan Lake', 'Giselle', 'Swan Lake', 'The Nutcracker', 'Giselle']
        unique, count = deduplicate_titles(titles)
        
        assert len(unique) == 3
        assert count == 2
        assert 'Swan Lake' in unique
        assert 'Giselle' in unique
        assert 'The Nutcracker' in unique
    
    def test_case_insensitive(self):
        """Test that deduplication is case-insensitive."""
        titles = ['Swan Lake', 'SWAN LAKE', 'swan lake']
        unique, count = deduplicate_titles(titles)
        
        assert len(unique) == 1
        assert count == 2
        # Preserves first occurrence
        assert unique[0] == 'Swan Lake'
    
    def test_preserves_order(self):
        """Test that original order is preserved."""
        titles = ['C', 'A', 'B', 'A']
        unique, count = deduplicate_titles(titles)
        
        assert unique == ['C', 'A', 'B']
        assert count == 1
    
    def test_empty_list(self):
        """Test with empty list."""
        unique, count = deduplicate_titles([])
        
        assert unique == []
        assert count == 0
    
    def test_whitespace_handling(self):
        """Test handling of whitespace in titles."""
        titles = ['Swan Lake', '  Swan Lake  ', 'Swan Lake']
        unique, count = deduplicate_titles(titles)
        
        assert len(unique) == 1
        assert count == 2


# =============================================================================
# Tests for generate_show_id
# =============================================================================

class TestGenerateShowId:
    """Tests for show ID generation."""
    
    def test_basic_id(self):
        """Test basic ID generation."""
        show_id = generate_show_id('The Nutcracker')
        
        assert show_id == 'the_nutcracker'
    
    def test_with_season(self):
        """Test ID generation with season."""
        show_id = generate_show_id('Swan Lake', '2024-25')
        
        assert show_id == 'swan_lake_2024_25'
    
    def test_special_characters(self):
        """Test that special characters are removed."""
        show_id = generate_show_id("Handmaid's Tale: The Ballet!")
        
        assert show_id == 'handmaids_tale_the_ballet'
    
    def test_multiple_spaces(self):
        """Test handling of multiple spaces."""
        show_id = generate_show_id('Don   Quixote')
        
        assert show_id == 'don_quixote'
    
    def test_ampersand(self):
        """Test handling of ampersand."""
        show_id = generate_show_id('Der Wolf & Rite of Spring')
        
        assert show_id == 'der_wolf_rite_of_spring'
    
    def test_hyphen_in_title(self):
        """Test handling of hyphens in title."""
        show_id = generate_show_id('BJM - L Cohen')
        
        assert show_id == 'bjm_l_cohen'


# =============================================================================
# Tests for BatchResult
# =============================================================================

class TestBatchResult:
    """Tests for BatchResult dataclass."""
    
    def test_default_values(self):
        """Test default values for BatchResult."""
        result = BatchResult(show_title='Test', success=False)
        
        assert result.show_title == 'Test'
        assert result.success is False
        assert result.output_path is None
        assert result.error_message is None
        assert result.total_tickets == 0
        assert result.performances == 0
    
    def test_successful_result(self):
        """Test creating a successful result."""
        result = BatchResult(
            show_title='Swan Lake',
            success=True,
            output_path='data/swan_lake_archtics_ticketmaster.csv',
            total_tickets=10000,
            performances=8,
        )
        
        assert result.success is True
        assert result.total_tickets == 10000
    
    def test_failed_result(self):
        """Test creating a failed result."""
        result = BatchResult(
            show_title='Unknown Show',
            success=False,
            error_message='No data found',
        )
        
        assert result.success is False
        assert result.error_message == 'No data found'


# =============================================================================
# Tests for BatchSummary
# =============================================================================

class TestBatchSummary:
    """Tests for BatchSummary dataclass."""
    
    def test_default_values(self):
        """Test default values for BatchSummary."""
        summary = BatchSummary()
        
        assert summary.total_shows == 0
        assert summary.successful == 0
        assert summary.failed == 0
        assert summary.skipped_duplicates == 0
        assert summary.results == []
    
    def test_add_successful_result(self):
        """Test adding a successful result."""
        summary = BatchSummary(total_shows=1)
        result = BatchResult(show_title='Swan Lake', success=True)
        
        summary.add_result(result)
        
        assert summary.successful == 1
        assert summary.failed == 0
        assert len(summary.results) == 1
    
    def test_add_failed_result(self):
        """Test adding a failed result."""
        summary = BatchSummary(total_shows=1)
        result = BatchResult(show_title='Unknown', success=False)
        
        summary.add_result(result)
        
        assert summary.successful == 0
        assert summary.failed == 1
        assert len(summary.results) == 1
    
    def test_multiple_results(self):
        """Test adding multiple results."""
        summary = BatchSummary(total_shows=3)
        
        summary.add_result(BatchResult(show_title='A', success=True))
        summary.add_result(BatchResult(show_title='B', success=True))
        summary.add_result(BatchResult(show_title='C', success=False))
        
        assert summary.successful == 2
        assert summary.failed == 1
        assert len(summary.results) == 3


# =============================================================================
# Integration test with real CSV file
# =============================================================================

class TestIntegrationWithRealCSV:
    """Integration tests using the actual history_city_sales.csv file."""
    
    def test_read_history_city_sales(self):
        """Test reading the actual history_city_sales.csv file."""
        csv_path = Path(__file__).parent.parent / 'data' / 'productions' / 'history_city_sales.csv'
        
        if not csv_path.exists():
            pytest.skip("history_city_sales.csv not found")
        
        titles = read_show_titles_from_csv(str(csv_path))
        
        # Should have at least some titles
        assert len(titles) > 0
        
        # Known titles should be present
        assert 'Swan Lake' in titles
        assert 'Cinderella' in titles
    
    def test_deduplicate_history_city_sales(self):
        """Test deduplication of titles from history_city_sales.csv."""
        csv_path = Path(__file__).parent.parent / 'data' / 'productions' / 'history_city_sales.csv'
        
        if not csv_path.exists():
            pytest.skip("history_city_sales.csv not found")
        
        titles = read_show_titles_from_csv(str(csv_path))
        unique, duplicates = deduplicate_titles(titles)
        
        # Should have removed some duplicates
        assert len(unique) <= len(titles)
        
        # Duplicates are expected (e.g., Cinderella appears twice)
        # Based on the CSV file we saw earlier, we expect some duplicates
        if duplicates > 0:
            assert len(unique) < len(titles)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
