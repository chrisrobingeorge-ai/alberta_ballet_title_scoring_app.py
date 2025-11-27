from pathlib import Path
from functools import lru_cache
from typing import Optional
import os

import pandas as pd

CONFIG_DIR = Path(__file__).parent


def _get_file_mtime(path: Path) -> float:
    """Get file modification time for cache invalidation."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


@lru_cache(maxsize=8)
def _load_csv_cached(file_path: str, mtime: float) -> pd.DataFrame:
    """
    Load a CSV file with caching based on modification time.
    
    Args:
        file_path: Path to CSV file
        mtime: File modification time (used for cache key)
        
    Returns:
        DataFrame from CSV
    """
    return pd.read_csv(file_path)


def load_feature_inventory() -> pd.DataFrame:
    """Load the feature inventory CSV as a DataFrame.
    
    This function is cached based on file modification time.
    """
    path = CONFIG_DIR / "ml_feature_inventory_alberta_ballet.csv"
    return _load_csv_cached(str(path), _get_file_mtime(path))


def load_join_keys() -> pd.DataFrame:
    """Load the join keys CSV as a DataFrame.
    
    This function is cached based on file modification time.
    """
    path = CONFIG_DIR / "ml_join_keys_alberta_ballet.csv"
    return _load_csv_cached(str(path), _get_file_mtime(path))


def load_data_sources() -> pd.DataFrame:
    """Load the data sources CSV as a DataFrame.
    
    This function is cached based on file modification time.
    """
    path = CONFIG_DIR / "ml_data_sources_alberta_ballet.csv"
    return _load_csv_cached(str(path), _get_file_mtime(path))


def load_pipelines() -> pd.DataFrame:
    """Load the pipelines CSV as a DataFrame.
    
    This function is cached based on file modification time.
    """
    path = CONFIG_DIR / "ml_pipelines_alberta_ballet.csv"
    return _load_csv_cached(str(path), _get_file_mtime(path))


def load_leakage_audit() -> pd.DataFrame:
    """Load the leakage audit CSV as a DataFrame.
    
    This function is cached based on file modification time.
    """
    path = CONFIG_DIR / "ml_leakage_audit_alberta_ballet.csv"
    return _load_csv_cached(str(path), _get_file_mtime(path))


def load_modelling_tasks() -> pd.DataFrame:
    """Load the modelling tasks CSV as a DataFrame.
    
    This function is cached based on file modification time.
    """
    path = CONFIG_DIR / "ml_modelling_tasks_alberta_ballet.csv"
    return _load_csv_cached(str(path), _get_file_mtime(path))
