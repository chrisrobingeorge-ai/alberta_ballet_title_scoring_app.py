"""
Tests to ensure modelling dataset doesn't contain leaky columns.

These tests verify that the data pipeline correctly excludes current-run
ticket columns from being used as predictors, which would cause data leakage.
"""

import os
import pytest
import pandas as pd


# Patterns that indicate current-run ticket data (forbidden as predictors)
FORBIDDEN_PATTERNS = [
    "single_tickets",
    "single tickets",
    "subscription_tickets", 
    "subscription tickets",
    "total_tickets",
    "total tickets",
    "total_single_tickets",
    "total single tickets",
    "yourmodel_",
    "_tickets_-_",
    "_tickets_calgary",
    "_tickets_edmonton",
]

# Columns that are allowed even though they contain "ticket"
ALLOWED_COLUMNS = {
    "prior_total_tickets",
    "mean_last_3_seasons",
    "ticket_median_prior",
    "ticket_mean_prior",
    "ticket_std_prior",
    "ticket_index_deseason",
    "target_ticket_median",  # Target column is allowed
    "years_since_last_ticket",
}


def is_forbidden_column(col_name: str) -> bool:
    """Check if a column name matches forbidden patterns."""
    col_lower = col_name.lower().strip()
    
    # Allow explicitly permitted columns
    if col_lower in ALLOWED_COLUMNS:
        return False
    
    # Check forbidden patterns
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in col_lower:
            return True
    
    return False


def test_is_forbidden_column_function():
    """Test the leakage detection function itself."""
    # These should be forbidden
    assert is_forbidden_column("Single Tickets - Calgary")
    assert is_forbidden_column("single_tickets_calgary")
    assert is_forbidden_column("Total Single Tickets")
    assert is_forbidden_column("Subscription_Tickets_-_Edmonton")
    assert is_forbidden_column("YourModel_Single_Tickets_Calgary")
    
    # These should be allowed
    assert not is_forbidden_column("prior_total_tickets")
    assert not is_forbidden_column("ticket_median_prior")
    assert not is_forbidden_column("wiki")
    assert not is_forbidden_column("category")
    assert not is_forbidden_column("target_ticket_median")


def test_no_leakage_in_modelling_dataset():
    """
    Test that modelling_dataset.csv (if present) doesn't contain 
    forbidden current-run ticket columns as features.
    """
    dataset_path = "data/modelling_dataset.csv"
    
    if not os.path.exists(dataset_path):
        pytest.skip(f"Modelling dataset not found at {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    
    # Identify columns that would be used as features
    # (exclude obvious non-features)
    non_feature_cols = {"title", "canonical_title", "show_title", "show_title_id"}
    
    # Also exclude target columns
    target_cols = {"target_ticket_median", "target", "ticket_median", "Total_Tickets"}
    
    exclude_cols = non_feature_cols | target_cols
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Check for forbidden columns
    forbidden_found = []
    for col in feature_cols:
        if is_forbidden_column(col):
            forbidden_found.append(col)
    
    assert len(forbidden_found) == 0, (
        f"DATA LEAKAGE: Found forbidden current-run ticket columns in features:\n"
        f"  {forbidden_found}\n"
        f"These columns must not be used as predictors."
    )


def test_history_file_has_expected_columns():
    """
    Verify that history_city_sales.csv has the expected structure.
    This helps catch schema changes early.
    """
    history_path = "data/history_city_sales.csv"
    
    if not os.path.exists(history_path):
        pytest.skip(f"History file not found at {history_path}")
    
    df = pd.read_csv(history_path, nrows=5)
    
    # Should have a title column
    title_variants = ["Show Title", "show_title", "Title", "title"]
    has_title = any(v in df.columns for v in title_variants)
    assert has_title, f"No title column found. Columns: {list(df.columns)}"
    
    # Should have ticket columns
    has_ticket_col = any("ticket" in c.lower() for c in df.columns)
    assert has_ticket_col, f"No ticket columns found. Columns: {list(df.columns)}"


def test_build_modelling_dataset_assertions():
    """
    Test that the build_modelling_dataset script has proper leakage assertions.
    """
    import sys
    sys.path.insert(0, ".")
    
    try:
        from scripts.build_modelling_dataset import is_forbidden_predictor, assert_no_leakage
    except ImportError:
        pytest.skip("build_modelling_dataset module not importable")
    
    # Test is_forbidden_predictor
    assert is_forbidden_predictor("Single Tickets - Calgary")
    assert is_forbidden_predictor("total_tickets")
    assert not is_forbidden_predictor("wiki")
    assert not is_forbidden_predictor("prior_total_tickets")
    
    # Test assert_no_leakage raises on forbidden columns
    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_no_leakage(
            pd.DataFrame(),
            ["wiki", "Single Tickets - Calgary"],
            "test"
        )
    
    # Should not raise for clean columns
    assert_no_leakage(
        pd.DataFrame(),
        ["wiki", "trends", "prior_total_tickets"],
        "test"
    )


def test_train_safe_model_assertions():
    """
    Test that train_safe_model.py has proper leakage assertions.
    """
    import sys
    sys.path.insert(0, ".")
    
    try:
        from scripts.train_safe_model import is_forbidden_feature, assert_safe_features
    except ImportError:
        pytest.skip("train_safe_model module not importable")
    
    # Test is_forbidden_feature
    assert is_forbidden_feature("Single Tickets - Calgary")
    assert is_forbidden_feature("YourModel_Total_Single_Tickets")
    assert not is_forbidden_feature("wiki")
    assert not is_forbidden_feature("prior_total_tickets")
    
    # Test assert_safe_features raises on forbidden columns
    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_safe_features(["wiki", "total_tickets"])
    
    # Should not raise for clean columns
    assert_safe_features(["wiki", "trends", "category"])
