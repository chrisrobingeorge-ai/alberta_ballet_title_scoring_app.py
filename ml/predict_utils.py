"""
Prediction Utilities for Streamlit Integration

This module provides utilities for loading trained models and making predictions
in the Streamlit app. It supports:
- Loading model pipelines
- Making predictions with optional calibration
- KNN fallback for cold-start titles

All functions are designed to fail gracefully when optional dependencies
or model files are not available.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Defensive imports
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    joblib = None

try:
    from ml.knn_fallback import KNNFallback, BASELINE_FEATURES
    KNN_AVAILABLE = True
except ImportError:
    KNN_AVAILABLE = False
    KNNFallback = None
    BASELINE_FEATURES = ["wiki", "trends", "youtube", "spotify"]

# Default paths
DEFAULT_MODEL_PATH = "models/model_xgb_remount_postcovid.joblib"
DEFAULT_CALIBRATION_PATH = "models/calibration.json"


class ModelNotFoundError(Exception):
    """Raised when model file is not found."""
    pass


class ModelLoadError(Exception):
    """Raised when model file exists but cannot be loaded (corrupted or incompatible)."""
    pass


class PredictionError(Exception):
    """Raised when prediction fails."""
    pass


def load_model_pipeline(path: str = DEFAULT_MODEL_PATH, raise_on_error: bool = True):
    """
    Load a trained sklearn/xgboost pipeline from disk.
    
    Args:
        path: Path to the .joblib model file
        raise_on_error: If True, raise exceptions. If False, return None on error.
        
    Returns:
        The loaded pipeline object, or None if raise_on_error is False and loading fails
        
    Raises:
        ModelNotFoundError: If the model file doesn't exist
        ModelLoadError: If the model file is corrupted or incompatible
        ImportError: If joblib is not available
    """
    if not JOBLIB_AVAILABLE:
        if raise_on_error:
            raise ImportError(
                "joblib is required to load models. Install with: pip install joblib"
            )
        return None
    
    if not os.path.exists(path):
        if raise_on_error:
            raise ModelNotFoundError(
                f"Model file not found: {path}\n"
                f"Run scripts/train_safe_model.py to train a model first."
            )
        return None
    
    try:
        return joblib.load(path)
    except Exception as e:
        if raise_on_error:
            raise ModelLoadError(
                f"Failed to load model from {path}. "
                f"The file may be corrupted or incompatible with current dependencies.\n"
                f"Error: {e}\n"
                f"Try retraining with: python scripts/train_safe_model.py"
            )
        return None


def load_model_metadata(
    model_path: str = DEFAULT_MODEL_PATH
) -> Optional[Dict[str, Any]]:
    """
    Load metadata for a trained model.
    
    The metadata file should be at the same path as the model but with .json extension.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Metadata dictionary or None if not found
    """
    metadata_path = model_path.replace(".joblib", ".json").replace(".pkl", ".json")
    
    if not os.path.exists(metadata_path):
        return None
    
    with open(metadata_path, "r") as f:
        return json.load(f)


def load_calibration(
    path: str = DEFAULT_CALIBRATION_PATH
) -> Optional[Dict[str, Any]]:
    """
    Load calibration parameters.
    
    Args:
        path: Path to calibration JSON file
        
    Returns:
        Calibration parameters dictionary or None if not found
    """
    if not os.path.exists(path):
        return None
    
    with open(path, "r") as f:
        return json.load(f)


def apply_calibration(
    predictions: np.ndarray,
    calibration: Dict[str, Any],
    categories: Optional[np.ndarray] = None,
    remount_years: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply calibration to predictions.
    
    Args:
        predictions: Array of predictions to calibrate
        calibration: Calibration parameters from load_calibration()
        categories: Category values (for per_category mode)
        remount_years: Years since last run (for by_remount_bin mode)
        
    Returns:
        Calibrated predictions
    """
    mode = calibration.get("mode", "global")
    params = calibration.get("parameters", {})
    
    predictions = np.asarray(predictions, dtype=float)
    calibrated = np.zeros_like(predictions)
    
    if mode == "global":
        p = params.get("global", {"alpha": 1.0, "beta": 0.0})
        calibrated = p["alpha"] * predictions + p["beta"]
    
    elif mode == "per_category" and categories is not None:
        categories = np.asarray(categories)
        for cat in np.unique(categories):
            if pd.isna(cat):
                continue
            mask = categories == cat
            p = params.get(str(cat), {"alpha": 1.0, "beta": 0.0})
            calibrated[mask] = p["alpha"] * predictions[mask] + p["beta"]
    
    elif mode == "by_remount_bin" and remount_years is not None:
        remount_years = np.asarray(remount_years)
        for i, years in enumerate(remount_years):
            if pd.isna(years):
                bin_name = "old_4y+"
            elif years <= 2:
                bin_name = "recent_0-2y"
            elif years <= 4:
                bin_name = "medium_2-4y"
            else:
                bin_name = "old_4y+"
            
            p = params.get(bin_name, {"alpha": 1.0, "beta": 0.0})
            calibrated[i] = p["alpha"] * predictions[i] + p["beta"]
    
    else:
        # Fallback to no calibration
        calibrated = predictions
    
    return calibrated


def predict_with_pipeline(
    df_features: pd.DataFrame,
    model_path: str = DEFAULT_MODEL_PATH,
    apply_calibration_flag: bool = False,
    calibration_mode: str = "global",
    calibration_path: str = DEFAULT_CALIBRATION_PATH
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """
    Make predictions using the trained pipeline.
    
    Args:
        df_features: DataFrame with feature columns
        model_path: Path to model file
        apply_calibration_flag: Whether to apply calibration
        calibration_mode: Calibration mode (for reference)
        calibration_path: Path to calibration file
        
    Returns:
        Tuple of (predictions array, metadata dict or None)
        
    Raises:
        ModelNotFoundError: If model not found
        PredictionError: If prediction fails
    """
    # Load model
    pipeline = load_model_pipeline(model_path)
    metadata = load_model_metadata(model_path)
    
    # Make predictions
    try:
        # Predictions are in log space if model was trained with log target
        predictions_log = pipeline.predict(df_features)
        
        # Inverse log transform
        predictions = np.expm1(predictions_log)
        
    except Exception as e:
        raise PredictionError(f"Prediction failed: {e}")
    
    # Apply calibration if requested
    if apply_calibration_flag:
        calibration = load_calibration(calibration_path)
        if calibration:
            categories = (
                df_features["category"].values 
                if "category" in df_features.columns 
                else None
            )
            remount_years = (
                df_features["years_since_last_run"].values 
                if "years_since_last_run" in df_features.columns 
                else None
            )
            predictions = apply_calibration(
                predictions, 
                calibration, 
                categories, 
                remount_years
            )
    
    return predictions, metadata


# Cache for KNN index
_knn_index_cache: Dict[str, Any] = {}


def _hash_dataframe(df: pd.DataFrame) -> str:
    """Create a hash of a DataFrame for cache invalidation."""
    # Use a simple hash based on shape and sample of data
    content = f"{df.shape}_{df.columns.tolist()}_{df.head(5).to_json()}"
    return hashlib.md5(content.encode()).hexdigest()


def build_knn_index_from_baselines(
    baselines_df: pd.DataFrame,
    ticket_priors: Optional[pd.DataFrame] = None,
    outcome_col: str = "ticket_median"
) -> Optional["KNNFallback"]:
    """
    Build a KNN index for cold-start fallback predictions.
    
    This function is memoized based on the input DataFrame hash.
    The cache is invalidated when the input data changes.
    
    Args:
        baselines_df: DataFrame with baseline signal columns
        ticket_priors: DataFrame with ticket priors (to merge outcomes)
        outcome_col: Name of outcome column
        
    Returns:
        Fitted KNNFallback instance or None if not possible
    """
    global _knn_index_cache
    
    if not KNN_AVAILABLE:
        return None
    
    # Create cache key
    baselines_hash = _hash_dataframe(baselines_df)
    priors_hash = _hash_dataframe(ticket_priors) if ticket_priors is not None else "none"
    cache_key = f"{baselines_hash}_{priors_hash}_{outcome_col}"
    
    # Check cache
    if cache_key in _knn_index_cache:
        return _knn_index_cache[cache_key]
    
    df = baselines_df.copy()
    
    # Merge ticket priors if provided
    if ticket_priors is not None and outcome_col in ticket_priors.columns:
        # Assume both have a 'title' or 'canonical_title' column
        merge_col = "canonical_title" if "canonical_title" in df.columns else "title"
        if merge_col in ticket_priors.columns:
            df = df.merge(
                ticket_priors[[merge_col, outcome_col]],
                on=merge_col,
                how="left"
            )
    
    # Check we have outcome column
    if outcome_col not in df.columns:
        _knn_index_cache[cache_key] = None
        return None
    
    # Filter to rows with valid outcomes
    df = df[df[outcome_col].notna() & (df[outcome_col] > 0)]
    
    if len(df) < 3:
        _knn_index_cache[cache_key] = None
        return None
    
    # Build index
    try:
        knn = KNNFallback(k=5, metric="cosine", normalize=True)
        knn.build_index(df, outcome_col=outcome_col)
        _knn_index_cache[cache_key] = knn
        return knn
    except Exception:
        _knn_index_cache[cache_key] = None
        return None


def knn_fallback_predict(
    title_baseline: Dict[str, float],
    knn_index: Optional["KNNFallback"],
    k: int = 5
) -> Optional[float]:
    """
    Make a KNN fallback prediction for a cold-start title.
    
    Args:
        title_baseline: Dict with baseline signals (wiki, trends, youtube, spotify)
        knn_index: A fitted KNNFallback instance
        k: Number of neighbors
        
    Returns:
        Predicted value or None if prediction not possible
    """
    if knn_index is None:
        return None
    
    if not KNN_AVAILABLE:
        return None
    
    try:
        prediction = knn_index.predict(title_baseline, k=k)
        if np.isnan(prediction):
            return None
        return float(prediction)
    except Exception:
        return None


def get_model_info() -> Dict[str, Any]:
    """
    Get information about available models and their status.
    
    Returns:
        Dictionary with model availability and metadata
    """
    info = {
        "pipeline_available": False,
        "calibration_available": False,
        "knn_available": KNN_AVAILABLE,
        "pipeline_metadata": None,
        "calibration_mode": None,
    }
    
    # Check pipeline
    if os.path.exists(DEFAULT_MODEL_PATH):
        info["pipeline_available"] = True
        info["pipeline_metadata"] = load_model_metadata(DEFAULT_MODEL_PATH)
    
    # Check calibration
    if os.path.exists(DEFAULT_CALIBRATION_PATH):
        info["calibration_available"] = True
        cal = load_calibration(DEFAULT_CALIBRATION_PATH)
        if cal:
            info["calibration_mode"] = cal.get("mode")
    
    return info


# Convenience function for Streamlit
def is_ml_model_available() -> bool:
    """Check if trained ML model is available."""
    return os.path.exists(DEFAULT_MODEL_PATH)


def is_calibration_available() -> bool:
    """Check if calibration parameters are available."""
    return os.path.exists(DEFAULT_CALIBRATION_PATH)
