"""
Tests to ensure the target column is never included as a feature in the ML dataset.

These tests verify that build_dataset correctly excludes TARGET_COL from
the feature set X, preventing target leakage during model training.
"""

import pytest
import pandas as pd

from ml.dataset import (
    TARGET_COL,
    build_dataset,
    ensure_no_target_in_features,
)
from data.features import get_feature_list, derive_basic_features, apply_registry_renames
from data.loader import load_history_sales


def test_target_col_is_defined():
    """Verify TARGET_COL is defined and is total_single_tickets."""
    assert TARGET_COL == "total_single_tickets"


def test_target_col_in_raw_features():
    """Confirm TARGET_COL is present in the raw feature-engineered data."""
    raw = load_history_sales()
    fe = derive_basic_features(apply_registry_renames(raw))
    assert TARGET_COL in fe.columns, (
        f"TARGET_COL '{TARGET_COL}' should be present in raw feature data "
        f"before exclusion. Found columns: {list(fe.columns)}"
    )


def test_target_col_not_in_build_dataset_X():
    """Confirm TARGET_COL is NOT in the returned X from build_dataset."""
    X, y = build_dataset()
    assert TARGET_COL not in X.columns, (
        f"TARGET_COL '{TARGET_COL}' must NOT be in X.columns to avoid target leakage. "
        f"Found columns: {list(X.columns)}"
    )


def test_target_col_is_in_y():
    """Confirm the target column values are properly returned in y."""
    X, y = build_dataset()
    assert len(y) > 0, "y should contain target values"
    assert isinstance(y, pd.Series), "y should be a pandas Series"
    # y should be a Series with numeric values
    assert pd.api.types.is_numeric_dtype(y.dtype), (
        f"y should contain numeric target values, got dtype {y.dtype}"
    )


def test_ensure_no_target_in_features_removes_target():
    """Test the helper function removes target column when present."""
    df = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
        "total_single_tickets": [100, 200, 300],
    })
    result = ensure_no_target_in_features(df, "total_single_tickets")
    assert "total_single_tickets" not in result.columns
    assert "feature1" in result.columns
    assert "feature2" in result.columns
    assert len(result.columns) == 2


def test_ensure_no_target_in_features_no_change_when_absent():
    """Test the helper function returns unchanged df when target not present."""
    df = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
    })
    result = ensure_no_target_in_features(df, "total_single_tickets")
    assert list(result.columns) == ["feature1", "feature2"]
    assert len(result.columns) == 2


def test_ensure_no_target_in_features_preserves_data():
    """Test the helper function preserves the data values correctly."""
    df = pd.DataFrame({
        "feature1": [1, 2, 3],
        "total_single_tickets": [100, 200, 300],
    })
    result = ensure_no_target_in_features(df, "total_single_tickets")
    assert list(result["feature1"]) == [1, 2, 3]


def test_build_dataset_returns_aligned_X_and_y():
    """Verify X and y have the same number of rows after build_dataset."""
    X, y = build_dataset()
    assert len(X) == len(y), (
        f"X and y should have same length. X has {len(X)} rows, y has {len(y)} rows."
    )


def test_no_model_training_uses_target_as_predictor():
    """
    Meta-test to ensure no training path uses total_single_tickets as a predictor.
    
    This test verifies the X returned by build_dataset never contains the target,
    regardless of filter settings.
    """
    # Test with various filter combinations
    test_cases = [
        {"theme_filters": None, "status": None, "forecast_time_only": True},
        {"theme_filters": None, "status": None, "forecast_time_only": False},
    ]
    
    for kwargs in test_cases:
        X, y = build_dataset(**kwargs)
        assert TARGET_COL not in X.columns, (
            f"TARGET_COL found in X with params {kwargs}. "
            f"This would cause target leakage in model training."
        )
