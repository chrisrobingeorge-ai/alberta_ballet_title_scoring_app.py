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
        path = DATA_DIR / "history_city_sales.csv"
        assert path.exists(), f"Missing file: {path}"
    
    def test_baselines_exists(self):
        """Test baselines.csv exists."""
        path = DATA_DIR / "baselines.csv"
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
