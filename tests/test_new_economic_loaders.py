"""Tests for new economic data loader functions."""
import pytest
import pandas as pd
from pathlib import Path

from data.loader import (
    load_nanos_consumer_confidence,
    load_nanos_better_off,
    load_commodity_price_index,
    load_boc_cpi_monthly,
    load_census_data,
    validate_data_source,
    DataRegistryReport,
    load_audience_analytics,
    load_live_analytics_raw,
    _clean_dataframe,
)


class TestDataFrameCleaning:
    """Tests for DataFrame cleaning helper function."""
    
    def test_clean_dataframe_drops_unnamed_columns(self):
        """_clean_dataframe should drop columns matching 'unnamed:' pattern."""
        df = pd.DataFrame({
            'good_col': [1, 2, 3],
            'Unnamed: 0': [4, 5, 6],
            'unnamed:_1': [7, 8, 9],
        })
        result = _clean_dataframe(df, drop_unnamed=True)
        
        assert 'good_col' in result.columns
        assert 'Unnamed: 0' not in result.columns
        assert 'unnamed:_1' not in result.columns
    
    def test_clean_dataframe_converts_numeric_columns(self):
        """_clean_dataframe should convert string columns to numeric."""
        df = pd.DataFrame({
            'value': ['1.5', '2.5', 'invalid'],
            'other': ['a', 'b', 'c'],
        })
        result = _clean_dataframe(df, drop_unnamed=False, numeric_columns=['value'])
        
        assert pd.api.types.is_numeric_dtype(result['value'])
        assert result['value'].iloc[0] == 1.5
        assert result['value'].iloc[1] == 2.5
        assert pd.isna(result['value'].iloc[2])  # 'invalid' should become NaN
    
    def test_clean_dataframe_drop_empty_unnamed_only(self):
        """_clean_dataframe should drop only empty unnamed columns when flag is set."""
        df = pd.DataFrame({
            'Unnamed: 0': [1, 2, 3],  # Has data
            'Unnamed: 1': [None, None, None],  # Empty
            'good_col': [4, 5, 6],
        })
        result = _clean_dataframe(df, drop_unnamed=True, drop_empty_unnamed_only=True)
        
        assert 'good_col' in result.columns
        assert 'Unnamed: 0' in result.columns  # Should remain (has data)
        assert 'Unnamed: 1' not in result.columns  # Should be dropped (empty)


class TestNanosConsumerConfidenceLoader:
    """Tests for Nanos consumer confidence data loading."""
    
    def test_load_nanos_consumer_confidence_returns_dataframe(self):
        """load_nanos_consumer_confidence should return a DataFrame."""
        result = load_nanos_consumer_confidence()
        assert isinstance(result, pd.DataFrame)
    
    def test_load_nanos_has_expected_columns_when_available(self):
        """When file exists, should have expected schema."""
        result = load_nanos_consumer_confidence()
        if not result.empty:
            expected_cols = ['category', 'subcategory', 'metric', 'value']
            for col in expected_cols:
                assert col in result.columns, f"Missing expected column: {col}"
    
    def test_no_unnamed_columns(self):
        """Loader should not return unnamed columns."""
        result = load_nanos_consumer_confidence()
        if not result.empty:
            unnamed = [c for c in result.columns if 'unnamed' in str(c).lower()]
            assert len(unnamed) == 0, f"Found unnamed columns: {unnamed}"
    
    def test_value_column_is_numeric(self):
        """Value column should be numeric dtype."""
        result = load_nanos_consumer_confidence()
        if not result.empty and 'value' in result.columns:
            assert pd.api.types.is_numeric_dtype(result['value']), \
                f"Expected numeric dtype, got {result['value'].dtype}"
    
    def test_fallback_on_missing_file(self):
        """Should return empty DataFrame when file is missing and fallback is True."""
        result = load_nanos_consumer_confidence(
            csv_name="nonexistent_file.csv",
            fallback_empty=True
        )
        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestNanosBetterOffLoader:
    """Tests for Nanos Better Off survey data loading."""
    
    def test_load_nanos_better_off_returns_dataframe(self):
        """load_nanos_better_off should return a DataFrame."""
        result = load_nanos_better_off()
        assert isinstance(result, pd.DataFrame)
    
    def test_no_unnamed_columns(self):
        """Loader should not return unnamed columns."""
        result = load_nanos_better_off()
        if not result.empty:
            unnamed = [c for c in result.columns if 'unnamed' in str(c).lower()]
            assert len(unnamed) == 0, f"Found unnamed columns: {unnamed}"
    
    def test_value_column_is_numeric(self):
        """Value column should be numeric dtype."""
        result = load_nanos_better_off()
        if not result.empty and 'value' in result.columns:
            assert pd.api.types.is_numeric_dtype(result['value']), \
                f"Expected numeric dtype, got {result['value'].dtype}"
    
    def test_fallback_on_missing_file(self):
        """Should return empty DataFrame when file is missing and fallback is True."""
        result = load_nanos_better_off(
            csv_name="nonexistent_file.csv",
            fallback_empty=True
        )
        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestAudienceAnalyticsLoader:
    """Tests for audience analytics data loading."""
    
    def test_load_audience_analytics_returns_dataframe(self):
        """load_audience_analytics should return a DataFrame."""
        result = load_audience_analytics()
        assert isinstance(result, pd.DataFrame)
    
    def test_no_empty_unnamed_columns(self):
        """Loader should not return empty unnamed columns."""
        result = load_audience_analytics()
        if not result.empty:
            for col in result.columns:
                if 'unnamed' in str(col).lower():
                    # If an unnamed column exists, it should have data
                    assert result[col].notna().any(), \
                        f"Found empty unnamed column: {col}"


class TestLiveAnalyticsLoader:
    """Tests for live analytics data loading."""
    
    def test_load_live_analytics_raw_returns_dataframe(self):
        """load_live_analytics_raw should return a DataFrame."""
        result = load_live_analytics_raw()
        assert isinstance(result, pd.DataFrame)
    
    def test_no_empty_unnamed_columns(self):
        """Loader should not return empty unnamed columns."""
        result = load_live_analytics_raw()
        if not result.empty:
            for col in result.columns:
                if 'unnamed' in str(col).lower():
                    # If an unnamed column exists, it should have data
                    assert result[col].notna().any(), \
                        f"Found empty unnamed column: {col}"


class TestCommodityPriceIndexLoader:
    """Tests for commodity price index data loading."""
    
    def test_load_commodity_returns_dataframe(self):
        """load_commodity_price_index should return a DataFrame."""
        result = load_commodity_price_index()
        assert isinstance(result, pd.DataFrame)
    
    def test_has_energy_column_when_available(self):
        """When file exists, should have A.ENER column."""
        result = load_commodity_price_index()
        if not result.empty:
            # Column names may be lowercased
            energy_cols = [c for c in result.columns if 'ener' in c.lower()]
            assert len(energy_cols) > 0, "Missing Energy index column"
    
    def test_date_column_is_datetime(self):
        """Date column should be parsed as datetime."""
        result = load_commodity_price_index()
        if not result.empty and 'date' in result.columns:
            assert pd.api.types.is_datetime64_any_dtype(result['date'])
    
    def test_fallback_on_missing_file(self):
        """Should return empty DataFrame when file is missing and fallback is True."""
        result = load_commodity_price_index(
            csv_name="nonexistent_file.csv",
            fallback_empty=True
        )
        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestBOCCPILoader:
    """Tests for Bank of Canada CPI data loading."""
    
    def test_load_cpi_returns_dataframe(self):
        """load_boc_cpi_monthly should return a DataFrame."""
        result = load_boc_cpi_monthly()
        assert isinstance(result, pd.DataFrame)
    
    def test_has_cpi_column_when_available(self):
        """When file exists, should have CPI column."""
        result = load_boc_cpi_monthly()
        if not result.empty:
            cpi_cols = [c for c in result.columns if 'V41690973' in c or 'cpi' in c.lower()]
            assert len(cpi_cols) > 0, "Missing CPI column"
    
    def test_date_column_is_datetime(self):
        """Date column should be parsed as datetime."""
        result = load_boc_cpi_monthly()
        if not result.empty and 'date' in result.columns:
            assert pd.api.types.is_datetime64_any_dtype(result['date'])
    
    def test_fallback_on_missing_file(self):
        """Should return empty DataFrame when file is missing and fallback is True."""
        result = load_boc_cpi_monthly(
            csv_name="nonexistent_file.csv",
            fallback_empty=True
        )
        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestCensusDataLoader:
    """Tests for census data loading."""
    
    def test_load_calgary_census_returns_dataframe(self):
        """load_census_data for Calgary should return a DataFrame."""
        result = load_census_data('Calgary')
        assert isinstance(result, pd.DataFrame)
    
    def test_load_edmonton_census_returns_dataframe(self):
        """load_census_data for Edmonton should return a DataFrame."""
        result = load_census_data('Edmonton')
        assert isinstance(result, pd.DataFrame)
    
    def test_fallback_on_missing_city(self):
        """Should return empty DataFrame for non-existent city."""
        result = load_census_data('FakeCity', fallback_empty=True)
        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestDataSourceValidation:
    """Tests for data source validation functions."""
    
    def test_validate_nonexistent_file(self):
        """Validation of non-existent file should report error."""
        report = validate_data_source("nonexistent_file.csv")
        
        assert isinstance(report, DataRegistryReport)
        assert report.exists is False
        assert len(report.validation_errors) > 0
    
    def test_validate_existing_file(self):
        """Validation of existing file should succeed."""
        # Use a known existing file
        report = validate_data_source("economics/commodity_price_index.csv")
        
        if report.exists:
            assert report.row_count > 0
            assert report.column_count > 0
    
    def test_report_includes_schema_info(self):
        """Report should include schema information when file exists."""
        report = validate_data_source("economics/commodity_price_index.csv")
        
        if report.exists:
            assert report.row_count >= 0
            assert report.column_count >= 0
            assert isinstance(report.null_counts, dict)
            assert isinstance(report.outlier_flags, dict)
    
    def test_date_coverage_extraction(self):
        """Report should extract date coverage when date column specified."""
        report = validate_data_source(
            "economics/commodity_price_index.csv",
            date_column='date'
        )
        
        if report.exists:
            # Should have date coverage info
            assert report.date_coverage_start is not None or len(report.validation_warnings) > 0


class TestDataRegistryReportDataclass:
    """Tests for DataRegistryReport dataclass."""
    
    def test_report_has_required_fields(self):
        """DataRegistryReport should have all required fields."""
        report = DataRegistryReport(
            source_name="test.csv",
            path="/path/to/test.csv",
            exists=True,
            row_count=100,
            column_count=10,
            date_coverage_start="2020-01-01",
            date_coverage_end="2024-12-31",
            null_counts={'col1': 5},
            outlier_flags={'col2': 3},
            validation_errors=[],
            validation_warnings=["Minor warning"]
        )
        
        assert report.source_name == "test.csv"
        assert report.exists is True
        assert report.row_count == 100
        assert report.column_count == 10
        assert 'col1' in report.null_counts
        assert len(report.validation_warnings) == 1
