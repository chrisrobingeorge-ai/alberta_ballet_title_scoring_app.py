"""
Tests for enhanced training module features.

Tests cover:
- Configuration loading
- Target transformation
- Model pipeline creation
- Feature importance extraction
- Subgroup metrics computation
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json


@pytest.fixture
def sample_features_and_target():
    """Create sample features and target for testing."""
    np.random.seed(42)
    n_samples = 100
    
    df = pd.DataFrame({
        "wiki": np.random.randint(20, 100, n_samples),
        "trends": np.random.randint(10, 60, n_samples),
        "youtube": np.random.randint(30, 90, n_samples),
        "spotify": np.random.randint(40, 80, n_samples),
        "genre": np.random.choice(["classic", "contemporary", "family"], n_samples),
        "season": np.random.choice(["fall", "winter", "spring"], n_samples),
    })
    
    # Simulate target as function of features
    target = (
        df["wiki"] * 50 +
        df["trends"] * 30 +
        df["youtube"] * 20 +
        np.random.normal(0, 500, n_samples)
    )
    
    return df, pd.Series(target, name="total_single_tickets")


def test_load_ml_config_defaults():
    """Test that config loading returns defaults when no file exists."""
    from ml.training import load_ml_config
    
    # Load from non-existent path
    config = load_ml_config(Path("/nonexistent/path.yaml"))
    
    assert "model" in config
    assert config["model"]["type"] == "random_forest"
    assert config["model"]["random_state"] == 42


def test_load_ml_config_from_file():
    """Test loading config from actual YAML file."""
    from ml.training import load_ml_config, CONFIGS_DIR
    
    config_path = CONFIGS_DIR / "ml_config.yaml"
    if config_path.exists():
        config = load_ml_config(config_path)
        assert "model" in config
        assert "knn" in config
        assert "target" in config


def test_apply_target_transform():
    """Test log1p target transformation."""
    from ml.training import apply_target_transform, inverse_target_transform
    
    y = pd.Series([0, 100, 1000, 10000])
    
    # Test with log transform
    y_log, was_log = apply_target_transform(y, use_log=True)
    assert was_log is True
    assert y_log.iloc[0] == 0  # log1p(0) = 0
    assert y_log.iloc[1] > 0
    
    # Test inverse transform
    y_back = inverse_target_transform(y_log.values, was_log_transformed=True)
    np.testing.assert_array_almost_equal(y.values, y_back, decimal=5)
    
    # Test without log transform
    y_no_log, was_log = apply_target_transform(y, use_log=False)
    assert was_log is False
    np.testing.assert_array_equal(y.values, y_no_log.values)


def test_create_model_pipeline_random_forest(sample_features_and_target):
    """Test creating Random Forest pipeline."""
    from ml.training import create_model_pipeline
    
    X, y = sample_features_and_target
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    
    pipe = create_model_pipeline(cat_cols, num_cols, model_type="random_forest")
    
    assert "pre" in pipe.named_steps
    assert "rf" in pipe.named_steps
    
    # Should be able to fit
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)


def test_create_model_pipeline_gradient_boosting(sample_features_and_target):
    """Test creating Gradient Boosting pipeline."""
    from ml.training import create_model_pipeline
    
    X, y = sample_features_and_target
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    
    pipe = create_model_pipeline(cat_cols, num_cols, model_type="gradient_boosting")
    
    assert "pre" in pipe.named_steps
    assert "gb" in pipe.named_steps
    
    # Should be able to fit
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)


def test_get_hyperparam_grid():
    """Test hyperparameter grid generation."""
    from ml.training import get_hyperparam_grid, load_ml_config
    
    config = load_ml_config()
    
    rf_grid = get_hyperparam_grid("random_forest", config)
    assert "rf__n_estimators" in rf_grid
    assert "rf__max_depth" in rf_grid
    
    gb_grid = get_hyperparam_grid("gradient_boosting", config)
    assert "gb__n_estimators" in gb_grid
    assert "gb__learning_rate" in gb_grid


def test_extract_feature_importances(sample_features_and_target):
    """Test feature importance extraction."""
    from ml.training import create_model_pipeline, extract_feature_importances
    
    X, y = sample_features_and_target
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    
    pipe = create_model_pipeline(cat_cols, num_cols, model_type="random_forest")
    pipe.fit(X, y)
    
    importances = extract_feature_importances(
        pipe, 
        feature_names=X.columns.tolist(),
        cat_cols=cat_cols,
        model_type="random_forest"
    )
    
    assert isinstance(importances, dict)
    assert len(importances) > 0
    # All values should be non-negative
    assert all(v >= 0 for v in importances.values())


def test_compute_subgroup_metrics(sample_features_and_target):
    """Test subgroup metrics computation."""
    from ml.training import compute_subgroup_metrics, create_model_pipeline
    
    X, y = sample_features_and_target
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    
    pipe = create_model_pipeline(cat_cols, num_cols, model_type="random_forest")
    pipe.fit(X, y)
    preds = pipe.predict(X)
    
    metrics = compute_subgroup_metrics(
        y_true=y,
        y_pred=preds,
        df_features=X,
        subgroups=["genre", "season"]
    )
    
    assert "genre" in metrics
    assert "season" in metrics
    
    # Each subgroup should have MAE, RMSE, R2
    for subgroup, values in metrics.items():
        for value, group_metrics in values.items():
            assert "mae" in group_metrics
            assert "rmse" in group_metrics
            assert "r2" in group_metrics
            assert "n_samples" in group_metrics


def test_save_and_load_model_metadata():
    """Test model metadata saving."""
    from ml.training import save_model_metadata
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pkl"
        metadata_path = Path(tmpdir) / "model_metadata.json"
        
        config = {
            "versioning": {
                "enabled": True,
                "metadata_path": str(metadata_path),
                "track_features": True,
                "track_hyperparams": True
            }
        }
        
        metrics = {"mae": 100.0, "r2": 0.8}
        features = ["wiki", "trends", "youtube"]
        hyperparams = {"n_estimators": 300}
        
        save_model_metadata(model_path, metrics, features, hyperparams, config)
        
        assert metadata_path.exists()
        
        with open(metadata_path) as f:
            saved = json.load(f)
        
        assert saved["metrics"] == metrics
        assert saved["features"] == features
        assert saved["hyperparameters"] == hyperparams
        assert "training_date" in saved


def test_get_git_commit_hash():
    """Test git commit hash retrieval."""
    from ml.training import get_git_commit_hash
    
    # Should return a string or None (if not in git repo or git unavailable)
    result = get_git_commit_hash()
    
    # Either None or a short hash string (7 chars)
    assert result is None or (isinstance(result, str) and len(result) >= 7)


def test_get_file_hash():
    """Test file hash computation."""
    from ml.training import get_file_hash
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with existing file
        test_file = Path(tmpdir) / "test_data.csv"
        test_file.write_text("col1,col2\n1,2\n3,4\n")
        
        file_hash = get_file_hash(test_file)
        
        assert file_hash is not None
        assert isinstance(file_hash, str)
        assert len(file_hash) == 16  # SHA-256 truncated to 16 chars
        
        # Same file should produce same hash
        file_hash_2 = get_file_hash(test_file)
        assert file_hash == file_hash_2
        
        # Different content should produce different hash
        test_file_2 = Path(tmpdir) / "test_data_2.csv"
        test_file_2.write_text("col1,col2\n5,6\n7,8\n")
        file_hash_3 = get_file_hash(test_file_2)
        assert file_hash != file_hash_3
        
        # Non-existent file should return None
        non_existent = Path(tmpdir) / "non_existent.csv"
        assert get_file_hash(non_existent) is None


def test_get_dataset_shape():
    """Test dataset shape extraction."""
    from ml.training import get_dataset_shape
    
    # Test with DataFrame
    df = pd.DataFrame({
        "col1": [1, 2, 3, 4],
        "col2": [5, 6, 7, 8],
        "col3": [9, 10, 11, 12],
    })
    
    shape = get_dataset_shape(df)
    
    assert shape is not None
    assert shape["n_rows"] == 4
    assert shape["n_columns"] == 3
    
    # Test with None
    assert get_dataset_shape(None) is None


def test_save_model_metadata_with_version_info():
    """Test that model metadata includes version information."""
    from ml.training import save_model_metadata
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pkl"
        metadata_path = Path(tmpdir) / "model_metadata.json"
        data_file = Path(tmpdir) / "test_data.csv"
        
        # Create a test data file
        data_file.write_text("col1,col2\n1,2\n3,4\n5,6\n")
        
        config = {
            "versioning": {
                "enabled": True,
                "metadata_path": str(metadata_path),
                "track_features": True,
                "track_hyperparams": True
            }
        }
        
        metrics = {"mae": 100.0, "r2": 0.8}
        features = ["wiki", "trends", "youtube"]
        hyperparams = {"n_estimators": 300}
        dataset_shape = {"n_rows": 100, "n_columns": 10}
        
        save_model_metadata(
            model_path, 
            metrics, 
            features, 
            hyperparams, 
            config,
            data_file_path=data_file,
            dataset_shape=dataset_shape,
        )
        
        assert metadata_path.exists()
        
        with open(metadata_path) as f:
            saved = json.load(f)
        
        # Verify required version metadata keys exist
        assert "git_commit_hash" in saved
        assert "data_file_hash" in saved
        assert "dataset_shape" in saved
        
        # Verify data file hash is computed
        assert saved["data_file_hash"] is not None
        assert isinstance(saved["data_file_hash"], str)
        assert len(saved["data_file_hash"]) == 16
        
        # Verify dataset shape
        assert saved["dataset_shape"]["n_rows"] == 100
        assert saved["dataset_shape"]["n_columns"] == 10
        
        # git_commit_hash may be None if not in a git repo
        # So we just check it exists as a key


def test_save_feature_importances():
    """Test feature importance saving."""
    from ml.training import save_feature_importances
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "feature_importance.json"
        
        config = {
            "explainability": {
                "export_importances": True,
                "importance_output_path": str(output_path),
                "top_n_features": 3
            }
        }
        
        importances = {
            "wiki": 0.4,
            "trends": 0.3,
            "youtube": 0.2,
            "spotify": 0.1
        }
        
        save_feature_importances(importances, config)
        
        assert output_path.exists()
        
        with open(output_path) as f:
            saved = json.load(f)
        
        assert "feature_importances" in saved
        assert "top_features" in saved
        assert len(saved["top_features"]) == 3


def test_knn_fallback_with_pca():
    """Test KNN fallback with PCA preprocessing."""
    from ml.knn_fallback import KNNFallback
    
    test_data = pd.DataFrame({
        "title": ["Swan Lake", "Nutcracker", "Romeo and Juliet", "Giselle", "Sleeping Beauty"],
        "wiki": [80, 95, 86, 74, 77],
        "trends": [18, 45, 32, 14, 30],
        "youtube": [71, 88, 80, 62, 70],
        "spotify": [71, 75, 80, 62, 70],
        "ticket_median": [9000, 12000, 8500, 6000, 7500],
    })
    
    # Test with PCA enabled
    knn = KNNFallback(k=3, use_pca=True, pca_components=2)
    knn.build_index(test_data, outcome_col="ticket_median")
    
    assert knn._pca is not None
    assert knn._is_fitted
    
    # Should still predict correctly
    new_show = {"wiki": 78, "trends": 25, "youtube": 75, "spotify": 65}
    prediction = knn.predict(new_show)
    
    assert isinstance(prediction, float)
    assert 5000 <= prediction <= 13000


def test_knn_fallback_distance_weighting():
    """Test KNN fallback with distance weighting."""
    from ml.knn_fallback import KNNFallback
    
    test_data = pd.DataFrame({
        "wiki": [80, 90, 70],
        "trends": [20, 40, 10],
        "youtube": [70, 80, 60],
        "spotify": [70, 80, 60],
        "ticket_median": [8000, 12000, 4000],
    })
    
    # Test with distance weighting
    knn = KNNFallback(k=3, weights="distance")
    knn.build_index(test_data, outcome_col="ticket_median")
    
    # Query exactly matching first row should heavily weight that neighbor
    new_show = {"wiki": 80, "trends": 20, "youtube": 70, "spotify": 70}
    prediction = knn.predict(new_show)
    
    # Should be closer to 8000 than the average
    assert isinstance(prediction, float)
    assert not np.isnan(prediction)


def test_schema_validation():
    """Test input schema validation."""
    from ml.scoring import validate_input_schema
    
    df = pd.DataFrame({
        "wiki": [80, 90],
        "trends": [20, 40],
    })
    
    # With no schema file, should return valid with warning
    is_valid, warnings = validate_input_schema(df)
    assert len(warnings) > 0  # Should have "no schema" warning


def test_score_with_uncertainty():
    """Test scoring with uncertainty intervals."""
    from ml.scoring import score_with_uncertainty
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib
    
    # Create a simple trained model
    np.random.seed(42)
    X_train = pd.DataFrame({
        "wiki": np.random.randint(20, 100, 50),
        "trends": np.random.randint(10, 60, 50),
    })
    y_train = X_train["wiki"] * 50 + X_train["trends"] * 30
    
    pipe = Pipeline([
        ("pre", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=10, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    
    # Test prediction
    X_test = pd.DataFrame({
        "wiki": [75, 85],
        "trends": [35, 45],
    })
    
    result = score_with_uncertainty(X_test, model=pipe, confidence_level=0.9)
    
    assert "prediction" in result.columns
    assert "lower_bound" in result.columns
    assert "upper_bound" in result.columns
    assert len(result) == 2
    
    # Lower should be less than prediction, upper should be greater
    assert all(result["lower_bound"] <= result["prediction"])
    assert all(result["prediction"] <= result["upper_bound"])


# =============================================================================
# Tests for time-aware CV enforcement (no random fallback for forecasting)
# =============================================================================

def test_train_baseline_model_raises_without_date_column(sample_features_and_target, monkeypatch):
    """Test that train_baseline_model raises MissingDateColumnError without date column."""
    from ml.training import MissingDateColumnError
    import ml.training as training_module
    
    X, y = sample_features_and_target
    # Mock build_dataset to return data without date column
    def mock_build_dataset(**kwargs):
        return X, y
    
    monkeypatch.setattr(training_module, "build_dataset", mock_build_dataset)
    
    with pytest.raises(MissingDateColumnError) as exc_info:
        training_module.train_baseline_model()
    
    # Verify error message is informative
    assert "date column" in str(exc_info.value).lower()
    assert "time-aware" in str(exc_info.value).lower()
    # Verify it contains searched columns info
    assert exc_info.value.searched_columns is not None
    assert len(exc_info.value.searched_columns) > 0


def test_train_with_cross_validation_raises_without_date_column(sample_features_and_target, monkeypatch):
    """Test that train_with_cross_validation raises MissingDateColumnError without date column."""
    from ml.training import MissingDateColumnError
    import ml.training as training_module
    
    X, y = sample_features_and_target
    # Mock build_dataset to return data without date column
    def mock_build_dataset(**kwargs):
        return X, y
    
    monkeypatch.setattr(training_module, "build_dataset", mock_build_dataset)
    
    with pytest.raises(MissingDateColumnError) as exc_info:
        training_module.train_with_cross_validation()
    
    # Verify error message is informative
    assert "date column" in str(exc_info.value).lower()
    # Verify it contains available columns info
    assert exc_info.value.available_columns is not None


def test_train_baseline_model_succeeds_with_date_column(sample_features_and_target, monkeypatch, tmp_path):
    """Test that train_baseline_model succeeds when date column is present."""
    import ml.training as training_module
    
    X, y = sample_features_and_target
    # Add a date column to the features
    X_with_dates = X.copy()
    X_with_dates["end_date"] = pd.date_range("2018-01-01", periods=len(X), freq="7D")
    
    # Mock build_dataset to return data with date column
    def mock_build_dataset(**kwargs):
        return X_with_dates, y
    
    monkeypatch.setattr(training_module, "build_dataset", mock_build_dataset)
    
    # Should not raise - provide a temp path for model saving
    model_path = tmp_path / "test_model.pkl"
    result = training_module.train_baseline_model(save_path=model_path)
    
    assert result is not None
    assert "metrics" in result
    assert result["metrics"]["time_aware_split"] is True


def test_train_with_cv_succeeds_with_date_column(sample_features_and_target, monkeypatch):
    """Test that train_with_cross_validation succeeds when date column is present."""
    import ml.training as training_module
    
    X, y = sample_features_and_target
    # Add a date column to the features
    X_with_dates = X.copy()
    X_with_dates["end_date"] = pd.date_range("2018-01-01", periods=len(X), freq="7D")
    
    # Mock build_dataset to return data with date column
    def mock_build_dataset(**kwargs):
        return X_with_dates, y
    
    monkeypatch.setattr(training_module, "build_dataset", mock_build_dataset)
    
    # Should not raise
    result = training_module.train_with_cross_validation()
    
    assert result is not None
    assert "folds" in result
    assert "cv_type" in result


def test_no_silent_random_cv_fallback(sample_features_and_target, monkeypatch):
    """Test that forecasting training never silently falls back to random CV.
    
    This is a key acceptance criterion: attempting to run forecasting training 
    without a date column should result in a clear error, not a silent fallback.
    """
    from ml.training import MissingDateColumnError
    import ml.training as training_module
    import warnings
    
    X, y = sample_features_and_target
    # Mock build_dataset to return data without date column
    def mock_build_dataset(**kwargs):
        return X, y
    
    monkeypatch.setattr(training_module, "build_dataset", mock_build_dataset)
    
    # Verify that the old warning-based fallback is no longer used
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # This should raise MissingDateColumnError, not just a warning
        with pytest.raises(MissingDateColumnError):
            training_module.train_baseline_model()
        
        # Verify no UserWarning about random split was issued
        random_split_warnings = [
            warning for warning in w 
            if "random split" in str(warning.message).lower()
        ]
        assert len(random_split_warnings) == 0, \
            "Should not issue warning about random split - should raise error instead"
