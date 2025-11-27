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
    - history_city_sales.csv
    - baselines.csv
    
    Returns:
        Tuple of (success: bool, errors: List[str])
    """
    errors = []
    
    required_data_files = [
        ("history_city_sales.csv", ["show_title"]),
        ("baselines.csv", ["title", "wiki", "trends", "youtube", "spotify"]),
    ]
    
    for file_name, required_cols in required_data_files:
        file_path = DATA_DIR / file_name
        file_errors = validate_csv_file(file_path, required_cols, file_name)
        errors.extend(file_errors)
    
    return len(errors) == 0, errors
