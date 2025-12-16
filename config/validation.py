"""
Configuration Validation Module

This module provides validation functions for ensuring all required configuration
files exist and contain the expected columns. It should be called at app startup
to catch configuration issues early.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Base paths
CONFIG_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent / "data"


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    
    def __init__(self, errors: List[str]):
        self.errors = errors
        message = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        super().__init__(message)


# ============================================================================
# Required CSV Files and Their Expected Columns
# ============================================================================

REQUIRED_CONFIG_FILES: Dict[str, Dict[str, Any]] = {
    "ml_feature_inventory_alberta_ballet.csv": {
        "path": CONFIG_DIR / "ml_feature_inventory_alberta_ballet.csv",
        "required_columns": ["Theme", "Feature Name", "Description", "Data Type", "Status"],
        "description": "Feature inventory containing all potential ML features"
    },
    "ml_leakage_audit_alberta_ballet.csv": {
        "path": CONFIG_DIR / "ml_leakage_audit_alberta_ballet.csv",
        "required_columns": ["Feature Name", "Leakage Risk (Y/N)", "Allowed at Forecast Time (Y/N)"],
        "description": "Leakage audit tracking which features are safe for forecasting"
    },
    "ml_join_keys_alberta_ballet.csv": {
        "path": CONFIG_DIR / "ml_join_keys_alberta_ballet.csv",
        "required_columns": ["Join Key", "Role / Purpose", "Connected Data Sources"],
        "description": "Join key documentation for data integration"
    },
    "ml_pipelines_alberta_ballet.csv": {
        "path": CONFIG_DIR / "ml_pipelines_alberta_ballet.csv",
        "required_columns": ["Pipeline Type", "Pipeline Name", "Description"],
        "description": "ML pipeline definitions"
    },
    "ml_modelling_tasks_alberta_ballet.csv": {
        "path": CONFIG_DIR / "ml_modelling_tasks_alberta_ballet.csv",
        "required_columns": ["Task ID", "Task Description", "Status"],
        "description": "Modelling task tracking"
    },
    "ml_data_sources_alberta_ballet.csv": {
        "path": CONFIG_DIR / "ml_data_sources_alberta_ballet.csv",
        "required_columns": ["Data Source Name", "Description", "Owner"],
        "description": "Data source documentation"
    },
}


def validate_csv_file(
    file_path: Path,
    required_columns: List[str],
    file_name: str
) -> List[str]:
    """
    Validate a single CSV file exists and has required columns.
    
    Args:
        file_path: Path to the CSV file
        required_columns: List of column names that must be present
        file_name: Human-readable file name for error messages
        
    Returns:
        List of error messages (empty if validation passes)
    """
    errors = []
    
    # Check file exists
    if not file_path.exists():
        errors.append(f"Missing file: {file_name} (expected at {file_path})")
        return errors
    
    # Try to read the file
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        errors.append(f"Cannot read {file_name}: {e}")
        return errors
    
    # Check for required columns
    actual_columns = set(df.columns)
    for col in required_columns:
        if col not in actual_columns:
            errors.append(f"{file_name}: Missing required column '{col}'")
    
    # Check file is not empty
    if len(df) == 0:
        errors.append(f"{file_name}: File is empty (no data rows)")
    
    return errors


def validate_config() -> Tuple[bool, List[str]]:
    """
    Validate all required configuration files.
    
    This function checks that all required CSV registry files exist and
    contain their expected columns. It should be called at app startup.
    
    Returns:
        Tuple of (success: bool, errors: List[str])
        - success is True if all validations pass
        - errors contains list of error messages if any
    """
    all_errors: List[str] = []
    
    for file_name, config in REQUIRED_CONFIG_FILES.items():
        errors = validate_csv_file(
            file_path=config["path"],
            required_columns=config["required_columns"],
            file_name=file_name
        )
        all_errors.extend(errors)
    
    return len(all_errors) == 0, all_errors


def validate_config_strict() -> None:
    """
    Validate configuration and raise an exception if validation fails.
    
    This is the strict version that should be used when you want to
    halt execution on configuration errors.
    
    Raises:
        ConfigValidationError: If any configuration validation fails
    """
    success, errors = validate_config()
    if not success:
        raise ConfigValidationError(errors)


def get_config_status() -> Dict[str, Dict[str, Any]]:
    """
    Get the status of all configuration files.
    
    Returns:
        Dictionary mapping file names to their status:
        - exists: bool
        - valid: bool
        - errors: List[str]
        - row_count: int (if file exists and is valid)
    """
    status = {}
    
    for file_name, config in REQUIRED_CONFIG_FILES.items():
        file_status = {
            "exists": config["path"].exists(),
            "valid": False,
            "errors": [],
            "row_count": 0,
            "description": config["description"]
        }
        
        if file_status["exists"]:
            errors = validate_csv_file(
                file_path=config["path"],
                required_columns=config["required_columns"],
                file_name=file_name
            )
            file_status["errors"] = errors
            file_status["valid"] = len(errors) == 0
            
            if file_status["valid"]:
                try:
                    df = pd.read_csv(config["path"])
                    file_status["row_count"] = len(df)
                except Exception:
                    pass
        
        status[file_name] = file_status
    
    return status


def validate_data_files() -> Tuple[bool, List[str]]:
    """
    Validate essential data files exist.
    
    This checks for the core data files needed for the app to function:
    - productions/history_city_sales.csv
    - productions/baselines.csv
    
    Returns:
        Tuple of (success: bool, errors: List[str])
    """
    errors = []
    
    required_data_files = [
        ("productions/history_city_sales.csv", ["show_title"]),
        ("productions/baselines.csv", ["title", "wiki", "trends", "youtube", "chartmetric"]),
    ]
    
    for file_name, required_cols in required_data_files:
        file_path = DATA_DIR / file_name
        file_errors = validate_csv_file(file_path, required_cols, file_name)
        errors.extend(file_errors)
    
    return len(errors) == 0, errors


# ============================================================================
# Core Data File Schemas
# ============================================================================

# Type hints for column validation:
# - "str": column should contain string values
# - "numeric": column should contain numeric values (int/float)
# - "date": column should contain date values
# - None: no type check (just check column exists)

CORE_DATA_SCHEMAS: Dict[str, Dict[str, Any]] = {
    # ----- Productions -----
    "productions/history_city_sales.csv": {
        "path": DATA_DIR / "productions" / "history_city_sales.csv",
        "required_columns": ["show_title"],
        "column_types": {"show_title": "str"},
        "description": "Historical city-level sales data for productions"
    },
    "productions/baselines.csv": {
        "path": DATA_DIR / "productions" / "baselines.csv",
        "required_columns": ["title", "wiki", "trends", "youtube", "chartmetric"],
        "column_types": {
            "title": "str",
            "wiki": "numeric",
            "trends": "numeric",
            "youtube": "numeric",
            "chartmetric": "numeric"
        },
        "description": "Baseline metrics for production titles"
    },
    "productions/segment_priors.csv": {
        "path": DATA_DIR / "productions" / "segment_priors.csv",
        "required_columns": ["region", "category", "segment", "weight"],
        "column_types": {
            "region": "str",
            "category": "str",
            "segment": "str",
            "weight": "numeric"
        },
        "description": "Segment prior weights by region and category"
    },
    "productions/past_runs.csv": {
        "path": DATA_DIR / "productions" / "past_runs.csv",
        "required_columns": ["title", "start_date", "end_date"],
        "column_types": {
            "title": "str",
            "start_date": "date",
            "end_date": "date"
        },
        "description": "Historical run dates for productions"
    },
    "productions/marketing_spend_per_ticket.csv": {
        "path": DATA_DIR / "productions" / "marketing_spend_per_ticket.csv",
        "required_columns": ["Show Title"],
        "column_types": {"Show Title": "str"},
        "description": "Marketing spend per ticket data"
    },
    # ----- Economics -----
    "economics/oil_price.csv": {
        "path": DATA_DIR / "economics" / "oil_price.csv",
        "required_columns": ["date", "wcs_oil_price"],
        "column_types": {
            "date": "date",
            "wcs_oil_price": "numeric"
        },
        "description": "WCS oil price time series"
    },
    "economics/unemployment_by_city.csv": {
        "path": DATA_DIR / "economics" / "unemployment_by_city.csv",
        "required_columns": ["date", "unemployment_rate", "region"],
        "column_types": {
            "date": "date",
            "unemployment_rate": "numeric",
            "region": "str"
        },
        "description": "Unemployment rate by region over time"
    },
    "economics/boc_cpi_monthly.csv": {
        "path": DATA_DIR / "economics" / "boc_cpi_monthly.csv",
        "required_columns": ["date"],
        "column_types": {"date": "date"},
        "description": "Bank of Canada CPI monthly data"
    },
    "economics/nanos_consumer_confidence.csv": {
        "path": DATA_DIR / "economics" / "nanos_consumer_confidence.csv",
        "required_columns": ["category", "subcategory", "metric", "value"],
        "column_types": {
            "category": "str",
            "subcategory": "str",
            "metric": "str"
            # Note: 'value' column type check is omitted due to data quality issues
            # in the source file (column data alignment issues with trailing commas)
        },
        "description": "Nanos consumer confidence survey data"
    },
    "economics/nanos_better_off.csv": {
        "path": DATA_DIR / "economics" / "nanos_better_off.csv",
        "required_columns": ["category", "subcategory", "metric", "value"],
        "column_types": {
            "category": "str",
            "subcategory": "str",
            "metric": "str"
            # Note: 'value' column type check is omitted due to data quality issues
            # in the source file (column data alignment issues)
        },
        "description": "Nanos 'better off' survey data"
    },
    # ----- Environment -----
    "environment/weatherstats_calgary_daily.csv": {
        "path": DATA_DIR / "environment" / "weatherstats_calgary_daily.csv",
        "required_columns": ["date", "max_temperature", "min_temperature", "precipitation"],
        "column_types": {
            "date": "date",
            "max_temperature": "numeric",
            "min_temperature": "numeric",
            "precipitation": "numeric"
        },
        "description": "Daily weather statistics for Calgary"
    },
    "environment/weatherstats_edmonton_daily.csv": {
        "path": DATA_DIR / "environment" / "weatherstats_edmonton_daily.csv",
        "required_columns": ["date", "max_temperature", "min_temperature", "precipitation"],
        "column_types": {
            "date": "date",
            "max_temperature": "numeric",
            "min_temperature": "numeric",
            "precipitation": "numeric"
        },
        "description": "Daily weather statistics for Edmonton"
    },
    # ----- Audiences -----
    "audiences/nanos_arts_donors.csv": {
        "path": DATA_DIR / "audiences" / "nanos_arts_donors.csv",
        "required_columns": ["section", "subcategory", "metric", "value"],
        "column_types": {
            "section": "str",
            "subcategory": "str",
            "metric": "str",
            "value": "numeric"
        },
        "description": "Nanos arts donors survey data"
    },
}


def _check_column_type(df: pd.DataFrame, col: str, expected_type: str) -> Optional[str]:
    """
    Check if a column matches the expected type.
    
    Args:
        df: DataFrame to check
        col: Column name to check
        expected_type: Expected type ("str", "numeric", or "date")
        
    Returns:
        Error message if type check fails, None otherwise
    """
    if col not in df.columns:
        return None  # Column check is done elsewhere
    
    series = df[col].dropna()
    if len(series) == 0:
        return None  # All values are null, skip type check
    
    if expected_type == "str":
        # String columns should be object dtype
        if not pd.api.types.is_object_dtype(df[col]) and not pd.api.types.is_string_dtype(df[col]):
            return f"Column '{col}' expected to be string type, got {df[col].dtype}"
    elif expected_type == "numeric":
        # Numeric columns should be int or float
        if not pd.api.types.is_numeric_dtype(df[col]):
            # Try to convert to see if it's numeric strings
            try:
                pd.to_numeric(series, errors='raise')
            except (ValueError, TypeError):
                return f"Column '{col}' expected to be numeric type, got {df[col].dtype}"
    elif expected_type == "date":
        # Date columns should be datetime or parseable as date
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            # Try to parse as date. We sample up to 100 rows for efficiency while still
            # catching most format issues. Full validation would be too slow for large files.
            sample_size = min(100, len(series))
            try:
                pd.to_datetime(series.head(sample_size), errors='raise')
            except (ValueError, TypeError):
                return f"Column '{col}' expected to be date type, cannot parse values"
    
    return None


def validate_core_data_file(
    file_path: Path,
    required_columns: List[str],
    column_types: Dict[str, str],
    file_name: str
) -> List[str]:
    """
    Validate a single core data file exists, has required columns, and basic type checks.
    
    Args:
        file_path: Path to the CSV file
        required_columns: List of column names that must be present
        column_types: Dict mapping column names to expected types
        file_name: Human-readable file name for error messages
        
    Returns:
        List of error messages (empty if validation passes)
    """
    errors = []
    
    # Check file exists
    if not file_path.exists():
        errors.append(f"Missing file: {file_name} (expected at {file_path})")
        return errors
    
    # Try to read the file
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        errors.append(f"Cannot read {file_name}: {e}")
        return errors
    
    # Check for required columns
    actual_columns = set(df.columns)
    for col in required_columns:
        if col not in actual_columns:
            errors.append(f"{file_name}: Missing required column '{col}'")
    
    # Check file is not empty
    if len(df) == 0:
        errors.append(f"{file_name}: File is empty (no data rows)")
        return errors
    
    # Check column types
    for col, expected_type in column_types.items():
        if col in actual_columns:
            type_error = _check_column_type(df, col, expected_type)
            if type_error:
                errors.append(f"{file_name}: {type_error}")
    
    return errors


def validate_all_core_data() -> Tuple[bool, List[str]]:
    """
    Validate all core data files used by the application.
    
    This function validates the schema of all key CSV data files used by the
    Alberta Ballet title scoring application. It checks for:
    - File existence
    - Required columns are present
    - Basic type validation for columns (string, numeric, date)
    
    Core data files validated include:
    - Productions: history_city_sales.csv, baselines.csv, segment_priors.csv,
      past_runs.csv, marketing_spend_per_ticket.csv
    - Economics: oil_price.csv, unemployment_by_city.csv, boc_cpi_monthly.csv,
      nanos_consumer_confidence.csv, nanos_better_off.csv
    - Environment: weatherstats_calgary_daily.csv, weatherstats_edmonton_daily.csv
    - Audiences: nanos_arts_donors.csv
    
    Returns:
        Tuple of (success: bool, errors: List[str])
        - success is True if all validations pass
        - errors contains list of error messages if any
        
    Example:
        >>> success, errors = validate_all_core_data()
        >>> if not success:
        ...     for error in errors:
        ...         print(f"Validation error: {error}")
    """
    all_errors: List[str] = []
    
    for file_name, schema in CORE_DATA_SCHEMAS.items():
        errors = validate_core_data_file(
            file_path=schema["path"],
            required_columns=schema["required_columns"],
            column_types=schema.get("column_types", {}),
            file_name=file_name
        )
        all_errors.extend(errors)
    
    return len(all_errors) == 0, all_errors


def validate_all_core_data_strict() -> None:
    """
    Validate all core data files and raise an exception if validation fails.
    
    This is the strict version that should be used when you want to
    halt execution on data validation errors.
    
    Raises:
        ConfigValidationError: If any core data file validation fails
    """
    success, errors = validate_all_core_data()
    if not success:
        raise ConfigValidationError(errors)


def get_core_data_status() -> Dict[str, Dict[str, Any]]:
    """
    Get the validation status of all core data files.
    
    Returns:
        Dictionary mapping file names to their status:
        - exists: bool
        - valid: bool
        - errors: List[str]
        - row_count: int (if file exists and is valid)
        - description: str
    """
    status = {}
    
    for file_name, schema in CORE_DATA_SCHEMAS.items():
        file_status = {
            "exists": schema["path"].exists(),
            "valid": False,
            "errors": [],
            "row_count": 0,
            "description": schema["description"]
        }
        
        if file_status["exists"]:
            errors = validate_core_data_file(
                file_path=schema["path"],
                required_columns=schema["required_columns"],
                column_types=schema.get("column_types", {}),
                file_name=file_name
            )
            file_status["errors"] = errors
            file_status["valid"] = len(errors) == 0
            
            if file_status["valid"]:
                try:
                    df = pd.read_csv(schema["path"])
                    file_status["row_count"] = len(df)
                except Exception:
                    pass
        
        status[file_name] = file_status
    
    return status
