"""
Configuration Validation Tests

Tests that verify all configuration files exist and contain required columns.
"""

import sys
from pathlib import Path

import pytest
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


CONFIG_DIR = Path(__file__).parent.parent / "config"
DATA_DIR = Path(__file__).parent.parent / "data"


class TestConfigFilesExist:
    """Tests that all required configuration files exist."""
    
    def test_feature_inventory_exists(self):
        """Test ml_feature_inventory_alberta_ballet.csv exists."""
        path = CONFIG_DIR / "ml_feature_inventory_alberta_ballet.csv"
        assert path.exists(), f"Missing file: {path}"
    
    def test_leakage_audit_exists(self):
        """Test ml_leakage_audit_alberta_ballet.csv exists."""
        path = CONFIG_DIR / "ml_leakage_audit_alberta_ballet.csv"
        assert path.exists(), f"Missing file: {path}"
    
    def test_join_keys_exists(self):
        """Test ml_join_keys_alberta_ballet.csv exists."""
        path = CONFIG_DIR / "ml_join_keys_alberta_ballet.csv"
        assert path.exists(), f"Missing file: {path}"
    
    def test_pipelines_exists(self):
        """Test ml_pipelines_alberta_ballet.csv exists."""
        path = CONFIG_DIR / "ml_pipelines_alberta_ballet.csv"
        assert path.exists(), f"Missing file: {path}"
    
    def test_modelling_tasks_exists(self):
        """Test ml_modelling_tasks_alberta_ballet.csv exists."""
        path = CONFIG_DIR / "ml_modelling_tasks_alberta_ballet.csv"
        assert path.exists(), f"Missing file: {path}"
    
    def test_data_sources_exists(self):
        """Test ml_data_sources_alberta_ballet.csv exists."""
        path = CONFIG_DIR / "ml_data_sources_alberta_ballet.csv"
        assert path.exists(), f"Missing file: {path}"
    
    def test_registry_module_exists(self):
        """Test registry.py module exists."""
        path = CONFIG_DIR / "registry.py"
        assert path.exists(), f"Missing file: {path}"
    
    def test_validation_module_exists(self):
        """Test validation.py module exists."""
        path = CONFIG_DIR / "validation.py"
        assert path.exists(), f"Missing file: {path}"


class TestConfigFilesReadable:
    """Tests that configuration files can be read as CSV."""
    
    def test_feature_inventory_readable(self):
        """Test ml_feature_inventory_alberta_ballet.csv is valid CSV."""
        path = CONFIG_DIR / "ml_feature_inventory_alberta_ballet.csv"
        df = pd.read_csv(path)
        assert len(df) > 0, "File is empty"
    
    def test_leakage_audit_readable(self):
        """Test ml_leakage_audit_alberta_ballet.csv is valid CSV."""
        path = CONFIG_DIR / "ml_leakage_audit_alberta_ballet.csv"
        df = pd.read_csv(path)
        assert len(df) > 0, "File is empty"
    
    def test_join_keys_readable(self):
        """Test ml_join_keys_alberta_ballet.csv is valid CSV."""
        path = CONFIG_DIR / "ml_join_keys_alberta_ballet.csv"
        df = pd.read_csv(path)
        assert len(df) > 0, "File is empty"
    
    def test_pipelines_readable(self):
        """Test ml_pipelines_alberta_ballet.csv is valid CSV."""
        path = CONFIG_DIR / "ml_pipelines_alberta_ballet.csv"
        df = pd.read_csv(path)
        assert len(df) > 0, "File is empty"
    
    def test_modelling_tasks_readable(self):
        """Test ml_modelling_tasks_alberta_ballet.csv is valid CSV."""
        path = CONFIG_DIR / "ml_modelling_tasks_alberta_ballet.csv"
        df = pd.read_csv(path)
        assert len(df) > 0, "File is empty"
    
    def test_data_sources_readable(self):
        """Test ml_data_sources_alberta_ballet.csv is valid CSV."""
        path = CONFIG_DIR / "ml_data_sources_alberta_ballet.csv"
        df = pd.read_csv(path)
        assert len(df) > 0, "File is empty"


class TestConfigFileColumns:
    """Tests that configuration files have required columns."""
    
    def test_feature_inventory_columns(self):
        """Test ml_feature_inventory_alberta_ballet.csv has required columns."""
        path = CONFIG_DIR / "ml_feature_inventory_alberta_ballet.csv"
        df = pd.read_csv(path)
        required = ["Theme", "Feature Name", "Description", "Data Type", "Status"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_leakage_audit_columns(self):
        """Test ml_leakage_audit_alberta_ballet.csv has required columns."""
        path = CONFIG_DIR / "ml_leakage_audit_alberta_ballet.csv"
        df = pd.read_csv(path)
        required = ["Feature Name", "Leakage Risk (Y/N)", "Allowed at Forecast Time (Y/N)"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_join_keys_columns(self):
        """Test ml_join_keys_alberta_ballet.csv has required columns."""
        path = CONFIG_DIR / "ml_join_keys_alberta_ballet.csv"
        df = pd.read_csv(path)
        required = ["Join Key", "Role / Purpose", "Connected Data Sources"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_pipelines_columns(self):
        """Test ml_pipelines_alberta_ballet.csv has required columns."""
        path = CONFIG_DIR / "ml_pipelines_alberta_ballet.csv"
        df = pd.read_csv(path)
        required = ["Pipeline Type", "Pipeline Name", "Description"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_modelling_tasks_columns(self):
        """Test ml_modelling_tasks_alberta_ballet.csv has required columns."""
        path = CONFIG_DIR / "ml_modelling_tasks_alberta_ballet.csv"
        df = pd.read_csv(path)
        required = ["Task ID", "Task Description", "Status"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_data_sources_columns(self):
        """Test ml_data_sources_alberta_ballet.csv has required columns."""
        path = CONFIG_DIR / "ml_data_sources_alberta_ballet.csv"
        df = pd.read_csv(path)
        required = ["Data Source Name", "Description", "Owner"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"


class TestDataFilesExist:
    """Tests that required data files exist."""
    
    def test_history_city_sales_exists(self):
        """Test history_city_sales.csv exists."""
        path = DATA_DIR / "productions" / "history_city_sales.csv"
        assert path.exists(), f"Missing file: {path}"
    
    def test_baselines_exists(self):
        """Test baselines.csv exists."""
        path = DATA_DIR / "productions" / "baselines.csv"
        assert path.exists(), f"Missing file: {path}"


class TestValidationModule:
    """Tests for the config validation module."""
    
    def test_validate_config_function(self):
        """Test validate_config returns expected structure."""
        from config.validation import validate_config
        
        success, errors = validate_config()
        assert isinstance(success, bool)
        assert isinstance(errors, list)
    
    def test_get_config_status_function(self):
        """Test get_config_status returns status for all files."""
        from config.validation import get_config_status
        
        status = get_config_status()
        assert isinstance(status, dict)
        assert len(status) > 0
        
        # Check structure of status entries
        for file_name, file_status in status.items():
            assert "exists" in file_status
            assert "valid" in file_status
            assert "errors" in file_status
    
    def test_validate_data_files_function(self):
        """Test validate_data_files checks data files."""
        from config.validation import validate_data_files
        
        success, errors = validate_data_files()
        assert isinstance(success, bool)
        assert isinstance(errors, list)


class TestRegistryLoaders:
    """Tests for registry loader functions."""
    
    def test_load_feature_inventory(self):
        """Test load_feature_inventory returns non-empty DataFrame."""
        from config.registry import load_feature_inventory
        df = load_feature_inventory()
        assert not df.empty, "Feature inventory is empty"
    
    def test_load_leakage_audit(self):
        """Test load_leakage_audit returns non-empty DataFrame."""
        from config.registry import load_leakage_audit
        df = load_leakage_audit()
        assert not df.empty, "Leakage audit is empty"
    
    def test_load_join_keys(self):
        """Test load_join_keys returns non-empty DataFrame."""
        from config.registry import load_join_keys
        df = load_join_keys()
        assert not df.empty, "Join keys is empty"
    
    def test_load_pipelines(self):
        """Test load_pipelines returns non-empty DataFrame."""
        from config.registry import load_pipelines
        df = load_pipelines()
        assert not df.empty, "Pipelines is empty"
    
    def test_load_modelling_tasks(self):
        """Test load_modelling_tasks returns non-empty DataFrame."""
        from config.registry import load_modelling_tasks
        df = load_modelling_tasks()
        assert not df.empty, "Modelling tasks is empty"
    
    def test_load_data_sources(self):
        """Test load_data_sources returns non-empty DataFrame."""
        from config.registry import load_data_sources
        df = load_data_sources()
        assert not df.empty, "Data sources is empty"


class TestValidateAllCoreData:
    """Tests for validate_all_core_data function."""
    
    def test_validate_all_core_data_returns_expected_structure(self):
        """Test validate_all_core_data returns tuple with bool and list."""
        from config.validation import validate_all_core_data
        
        success, errors = validate_all_core_data()
        assert isinstance(success, bool)
        assert isinstance(errors, list)
    
    def test_validate_all_core_data_passes_for_checked_in_data(self):
        """Test validate_all_core_data passes for all checked-in data files."""
        from config.validation import validate_all_core_data
        
        success, errors = validate_all_core_data()
        assert success, f"Validation failed with errors: {errors}"
    
    def test_get_core_data_status_returns_all_files(self):
        """Test get_core_data_status returns status for all core files."""
        from config.validation import get_core_data_status, CORE_DATA_SCHEMAS
        
        status = get_core_data_status()
        assert isinstance(status, dict)
        assert len(status) == len(CORE_DATA_SCHEMAS)
        
        for file_name, file_status in status.items():
            assert "exists" in file_status
            assert "valid" in file_status
            assert "errors" in file_status
            assert "row_count" in file_status
            assert "description" in file_status
    
    def test_all_core_data_files_exist(self):
        """Test all core data files exist on disk."""
        from config.validation import get_core_data_status
        
        status = get_core_data_status()
        for file_name, file_status in status.items():
            assert file_status["exists"], f"Missing core data file: {file_name}"
    
    def test_all_core_data_files_have_rows(self):
        """Test all core data files have at least one row."""
        from config.validation import get_core_data_status
        
        status = get_core_data_status()
        for file_name, file_status in status.items():
            if file_status["valid"]:
                assert file_status["row_count"] > 0, f"Empty file: {file_name}"
    
    def test_validate_all_core_data_strict_raises_on_error(self):
        """Test validate_all_core_data_strict raises ConfigValidationError on failure."""
        from config.validation import (
            validate_all_core_data_strict,
            ConfigValidationError,
        )
        
        # Should not raise for valid data
        validate_all_core_data_strict()


class TestCoreDataSchemaDetection:
    """Tests that verify schema changes are detected."""
    
    def test_missing_column_detected(self, tmp_path):
        """Test validation fails when a required column is missing."""
        from config.validation import validate_core_data_file
        
        # Create a temp CSV missing a required column
        csv_file = tmp_path / "test_missing_column.csv"
        csv_file.write_text("region,category\nProvince,classic_romance\n")
        
        errors = validate_core_data_file(
            file_path=csv_file,
            required_columns=["region", "category", "segment", "weight"],
            column_types={},
            file_name="test_file.csv"
        )
        
        assert len(errors) == 2  # Missing segment and weight
        assert any("segment" in e for e in errors)
        assert any("weight" in e for e in errors)
    
    def test_renamed_column_detected(self, tmp_path):
        """Test validation fails when a column is renamed."""
        from config.validation import validate_core_data_file
        
        # Create a temp CSV with a renamed column
        csv_file = tmp_path / "test_renamed_column.csv"
        csv_file.write_text("title,start_date,ending_date\nTest,2024-01-01,2024-01-31\n")
        
        errors = validate_core_data_file(
            file_path=csv_file,
            required_columns=["title", "start_date", "end_date"],
            column_types={},
            file_name="test_file.csv"
        )
        
        assert len(errors) == 1
        assert "end_date" in errors[0]
    
    def test_empty_file_detected(self, tmp_path):
        """Test validation fails when file is empty (header only)."""
        from config.validation import validate_core_data_file
        
        # Create a temp CSV with only headers
        csv_file = tmp_path / "test_empty.csv"
        csv_file.write_text("title,start_date,end_date\n")
        
        errors = validate_core_data_file(
            file_path=csv_file,
            required_columns=["title", "start_date", "end_date"],
            column_types={},
            file_name="test_file.csv"
        )
        
        assert len(errors) == 1
        assert "empty" in errors[0].lower()
    
    def test_missing_file_detected(self, tmp_path):
        """Test validation fails when file is missing."""
        from config.validation import validate_core_data_file
        
        missing_file = tmp_path / "nonexistent.csv"
        
        errors = validate_core_data_file(
            file_path=missing_file,
            required_columns=["col1"],
            column_types={},
            file_name="nonexistent.csv"
        )
        
        assert len(errors) == 1
        assert "Missing file" in errors[0]
    
    def test_wrong_column_type_detected(self, tmp_path):
        """Test validation fails when column has wrong type."""
        from config.validation import validate_core_data_file
        
        # Create a temp CSV with wrong column type
        csv_file = tmp_path / "test_wrong_type.csv"
        csv_file.write_text("value\nabc\nxyz\n")
        
        errors = validate_core_data_file(
            file_path=csv_file,
            required_columns=["value"],
            column_types={"value": "numeric"},
            file_name="test_file.csv"
        )
        
        assert len(errors) == 1
        assert "numeric" in errors[0].lower()
    
    def test_correct_file_passes(self, tmp_path):
        """Test validation passes when file matches schema."""
        from config.validation import validate_core_data_file
        
        # Create a valid temp CSV
        csv_file = tmp_path / "test_valid.csv"
        csv_file.write_text("region,category,segment,weight\nProvince,classic_romance,General Population,1.00\n")
        
        errors = validate_core_data_file(
            file_path=csv_file,
            required_columns=["region", "category", "segment", "weight"],
            column_types={"region": "str", "weight": "numeric"},
            file_name="test_file.csv"
        )
        
        assert len(errors) == 0


class TestCoreDataSchemas:
    """Tests for individual core data file schemas."""
    
    def test_productions_history_city_sales_schema(self):
        """Test history_city_sales.csv has expected schema."""
        from config.validation import CORE_DATA_SCHEMAS
        
        schema = CORE_DATA_SCHEMAS.get("productions/history_city_sales.csv")
        assert schema is not None
        assert "show_title" in schema["required_columns"]
    
    def test_productions_baselines_schema(self):
        """Test baselines.csv has expected schema."""
        from config.validation import CORE_DATA_SCHEMAS
        
        schema = CORE_DATA_SCHEMAS.get("productions/baselines.csv")
        assert schema is not None
        required = schema["required_columns"]
        assert "title" in required
        assert "wiki" in required
        assert "trends" in required
        assert "youtube" in required
        assert "chartmetric" in required
    
    def test_productions_segment_priors_schema(self):
        """Test segment_priors.csv has expected schema."""
        from config.validation import CORE_DATA_SCHEMAS
        
        schema = CORE_DATA_SCHEMAS.get("productions/segment_priors.csv")
        assert schema is not None
        required = schema["required_columns"]
        assert "region" in required
        assert "category" in required
        assert "segment" in required
        assert "weight" in required
    
    def test_productions_past_runs_schema(self):
        """Test past_runs.csv has expected schema."""
        from config.validation import CORE_DATA_SCHEMAS
        
        schema = CORE_DATA_SCHEMAS.get("productions/past_runs.csv")
        assert schema is not None
        required = schema["required_columns"]
        assert "title" in required
        assert "start_date" in required
        assert "end_date" in required
    
    def test_economics_oil_price_schema(self):
        """Test oil_price.csv has expected schema."""
        from config.validation import CORE_DATA_SCHEMAS
        
        schema = CORE_DATA_SCHEMAS.get("economics/oil_price.csv")
        assert schema is not None
        required = schema["required_columns"]
        assert "date" in required
        assert "wcs_oil_price" in required
    
    def test_economics_unemployment_schema(self):
        """Test unemployment_by_city.csv has expected schema."""
        from config.validation import CORE_DATA_SCHEMAS
        
        schema = CORE_DATA_SCHEMAS.get("economics/unemployment_by_city.csv")
        assert schema is not None
        required = schema["required_columns"]
        assert "date" in required
        assert "unemployment_rate" in required
        assert "region" in required
    
    def test_economics_cpi_schema(self):
        """Test boc_cpi_monthly.csv has expected schema."""
        from config.validation import CORE_DATA_SCHEMAS
        
        schema = CORE_DATA_SCHEMAS.get("economics/boc_cpi_monthly.csv")
        assert schema is not None
        assert "date" in schema["required_columns"]
    
    def test_economics_nanos_consumer_confidence_schema(self):
        """Test nanos_consumer_confidence.csv has expected schema."""
        from config.validation import CORE_DATA_SCHEMAS
        
        schema = CORE_DATA_SCHEMAS.get("economics/nanos_consumer_confidence.csv")
        assert schema is not None
        required = schema["required_columns"]
        assert "category" in required
        assert "subcategory" in required
        assert "metric" in required
        assert "value" in required
    
    def test_environment_weather_calgary_schema(self):
        """Test weatherstats_calgary_daily.csv has expected schema."""
        from config.validation import CORE_DATA_SCHEMAS
        
        schema = CORE_DATA_SCHEMAS.get("environment/weatherstats_calgary_daily.csv")
        assert schema is not None
        required = schema["required_columns"]
        assert "date" in required
        assert "max_temperature" in required
        assert "min_temperature" in required
        assert "precipitation" in required
    
    def test_environment_weather_edmonton_schema(self):
        """Test weatherstats_edmonton_daily.csv has expected schema."""
        from config.validation import CORE_DATA_SCHEMAS
        
        schema = CORE_DATA_SCHEMAS.get("environment/weatherstats_edmonton_daily.csv")
        assert schema is not None
        required = schema["required_columns"]
        assert "date" in required
        assert "max_temperature" in required
        assert "min_temperature" in required
        assert "precipitation" in required
    
    def test_audiences_nanos_arts_donors_schema(self):
        """Test nanos_arts_donors.csv has expected schema."""
        from config.validation import CORE_DATA_SCHEMAS
        
        schema = CORE_DATA_SCHEMAS.get("audiences/nanos_arts_donors.csv")
        assert schema is not None
        required = schema["required_columns"]
        assert "section" in required
        assert "subcategory" in required
        assert "metric" in required
        assert "value" in required


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
