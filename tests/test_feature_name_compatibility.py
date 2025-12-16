"""
Test feature name compatibility between title_scoring_helper and model.

This test ensures that the column names provided by title_scoring_helper
match what the model expects, preventing the "X has 5 features, but 
ColumnTransformer is expecting 35 features" error.
"""

import pandas as pd
import pytest
import joblib
from pathlib import Path

from ml.scoring import score_runs_for_planning, _prepare_features_for_model


def test_categorical_feature_names_match_model():
    """
    Test that categorical features use correct names that match the model.
    
    The model expects 'category' and 'opening_season', not 'genre' and 'season'.
    This test ensures title_scoring_helper provides the correct names.
    """
    # Load the actual production model
    model_path = Path(__file__).parent.parent / "models" / "model_xgb_remount_postcovid.joblib"
    if not model_path.exists():
        pytest.skip("Production model not found, skipping test")
    
    model = joblib.load(model_path)
    
    # Extract expected categorical features from model
    preprocessor = model.named_steps['preprocessor']
    categorical_features = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'cat':
            categorical_features = columns
            break
    
    assert 'category' in categorical_features, "Model expects 'category' feature"
    assert 'opening_season' in categorical_features, "Model expects 'opening_season' feature"
    
    # These wrong names should NOT be in the model's expected features
    assert 'genre' not in categorical_features, "'genre' should not be a model feature"
    assert 'season' not in categorical_features, "'season' should not be a model feature (it's a label column)"


def test_correct_column_names_preserve_values():
    """
    Test that using correct column names preserves user-provided values.
    
    When title_scoring_helper provides 'category' and 'opening_season',
    these values should be preserved through to the model prediction.
    """
    # Input with correct column names (as fixed in title_scoring_helper.py)
    df_input = pd.DataFrame([{
        'wiki': 80.0,
        'trends': 60.0,
        'youtube': 70.0,
        'chartmetric': 75.0,
        'category': 'classical',
        'opening_season': '2025-26',
    }])
    
    # Load model
    model_path = Path(__file__).parent.parent / "models" / "model_xgb_remount_postcovid.joblib"
    if not model_path.exists():
        pytest.skip("Production model not found, skipping test")
    
    model = joblib.load(model_path)
    
    # Prepare features
    df_prepared = _prepare_features_for_model(df_input, model=model)
    
    # Verify values are preserved
    assert df_prepared['category'].values[0] == 'classical', \
        "Category value should be preserved"
    assert df_prepared['opening_season'].values[0] == '2025-26', \
        "Opening_season value should be preserved"
    
    # Values should NOT be 'missing'
    assert df_prepared['category'].values[0] != 'missing', \
        "Category should not default to 'missing' when provided"
    assert df_prepared['opening_season'].values[0] != 'missing', \
        "Opening_season should not default to 'missing' when provided"


def test_wrong_column_names_lose_values():
    """
    Test that using wrong column names (old bug) loses user-provided values.
    
    This test documents the OLD behavior where providing 'genre' and 'season'
    resulted in values being lost and replaced with 'missing'.
    """
    # Input with wrong column names (OLD buggy behavior)
    df_input = pd.DataFrame([{
        'wiki': 80.0,
        'trends': 60.0,
        'youtube': 70.0,
        'chartmetric': 75.0,
        'genre': 'classical',  # WRONG: should be 'category'
        'season': '2025-26',   # WRONG: should be 'opening_season'
    }])
    
    # Load model
    model_path = Path(__file__).parent.parent / "models" / "model_xgb_remount_postcovid.joblib"
    if not model_path.exists():
        pytest.skip("Production model not found, skipping test")
    
    model = joblib.load(model_path)
    
    # Prepare features
    df_prepared = _prepare_features_for_model(df_input, model=model)
    
    # With wrong column names, values get lost
    assert df_prepared['category'].values[0] == 'missing', \
        "Category defaults to 'missing' when 'genre' is provided instead"
    assert df_prepared['opening_season'].values[0] == 'missing', \
        "Opening_season defaults to 'missing' when 'season' is provided instead"


def test_scoring_succeeds_with_correct_names():
    """
    Test that scoring succeeds with correct column names.
    
    This is the main integration test that ensures the fix works end-to-end.
    """
    # Input with correct column names
    df_input = pd.DataFrame([{
        'wiki': 80.0,
        'trends': 60.0,
        'youtube': 70.0,
        'chartmetric': 75.0,
        'category': 'classical',
        'opening_season': '2025-26',
    }])
    
    # Scoring should succeed
    result = score_runs_for_planning(df_input, n_bootstrap=10)
    
    # Check that we got predictions
    assert 'forecast_single_tickets' in result.columns
    assert result['forecast_single_tickets'].notna().all()
    assert result['forecast_single_tickets'].values[0] > 0


def test_all_35_features_present_after_preparation():
    """
    Test that _prepare_features_for_model adds all 35 expected features.
    
    This ensures we never get "X has 5 features, but ColumnTransformer 
    is expecting 35 features" error.
    """
    # Minimal input
    df_input = pd.DataFrame([{
        'wiki': 80.0,
        'trends': 60.0,
        'youtube': 70.0,
        'chartmetric': 75.0,
        'category': 'classical',
        'opening_season': '2025-26',
    }])
    
    # Load model
    model_path = Path(__file__).parent.parent / "models" / "model_xgb_remount_postcovid.joblib"
    if not model_path.exists():
        pytest.skip("Production model not found, skipping test")
    
    model = joblib.load(model_path)
    
    # Prepare features
    df_prepared = _prepare_features_for_model(df_input, model=model)
    
    # Get expected feature count from model
    preprocessor = model.named_steps['preprocessor']
    expected_features = []
    for name, transformer, columns in preprocessor.transformers_:
        if name != 'remainder':
            expected_features.extend(columns)
    
    # Verify feature count matches
    assert len(df_prepared.columns) == len(expected_features), \
        f"Expected {len(expected_features)} features, got {len(df_prepared.columns)}"
    
    # Verify all expected features are present
    missing = set(expected_features) - set(df_prepared.columns)
    assert len(missing) == 0, f"Missing features: {missing}"
    
    # Verify model.predict works (doesn't raise the 35 features error)
    try:
        prediction = model.predict(df_prepared)
        assert prediction is not None
    except ValueError as e:
        if "X has" in str(e) and "features" in str(e):
            pytest.fail(f"Got feature count mismatch error: {e}")
        raise
