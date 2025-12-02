"""
End-to-End ML Pipeline Tests

These tests verify the complete ML training pipeline works correctly,
from dataset building through model training and prediction.
"""

import os
import sys
import json
import tempfile
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def synthetic_modelling_dataset():
    """Create a small synthetic modelling dataset for testing."""
    np.random.seed(42)
    n_samples = 20
    
    # Generate sequential dates for time-aware cross-validation
    start_date = pd.Timestamp("2020-01-01")
    end_dates = pd.date_range(start=start_date, periods=n_samples, freq="3ME")
    
    data = {
        "title": [f"Test Title {i}" for i in range(n_samples)],
        "canonical_title": [f"test_title_{i}" for i in range(n_samples)],
        "end_date": end_dates,
        "wiki": np.random.randint(30, 80, n_samples),
        "trends": np.random.randint(20, 70, n_samples),
        "youtube": np.random.randint(25, 75, n_samples),
        "spotify": np.random.randint(15, 65, n_samples),
        "category": np.random.choice(["family_classic", "classic_romance", "contemporary"], n_samples),
        "gender": np.random.choice(["female", "male", "co", "na"], n_samples),
        "years_since_last_run": np.random.choice([0, 1, 2, 3, 5, np.nan], n_samples),
        "is_remount_recent": np.random.choice([0, 1], n_samples),
        "prior_total_tickets": np.random.randint(5000, 15000, n_samples).astype(float),
        "target_ticket_median": np.random.randint(8000, 18000, n_samples).astype(float),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestDatasetBuilding:
    """Tests for the modelling dataset builder."""
    
    def test_dataset_has_required_columns(self, synthetic_modelling_dataset):
        """Verify the dataset contains required columns."""
        required_cols = ["wiki", "trends", "youtube", "spotify", "category", "target_ticket_median"]
        for col in required_cols:
            assert col in synthetic_modelling_dataset.columns, f"Missing column: {col}"
    
    def test_dataset_no_leakage_columns(self, synthetic_modelling_dataset):
        """Verify no leakage columns are present."""
        forbidden_patterns = [
            "single_tickets", "total_tickets",
            "yourmodel_", "_tickets_calgary", "_tickets_edmonton"
        ]
        
        for col in synthetic_modelling_dataset.columns:
            col_lower = col.lower()
            for pattern in forbidden_patterns:
                if "prior" in col_lower or "median" in col_lower:
                    continue  # These are allowed
                assert pattern not in col_lower, f"Potential leakage column: {col}"
    
    def test_target_column_valid(self, synthetic_modelling_dataset):
        """Verify target column has valid values."""
        target = synthetic_modelling_dataset["target_ticket_median"]
        assert target.notna().sum() > 0, "Target column has no valid values"
        assert (target > 0).any(), "Target column has no positive values"


class TestModelTraining:
    """Tests for the model training pipeline."""
    
    def test_training_with_synthetic_data(self, synthetic_modelling_dataset, temp_output_dir):
        """Test that training runs successfully with synthetic data."""
        # Save synthetic dataset
        dataset_path = os.path.join(temp_output_dir, "test_dataset.csv")
        synthetic_modelling_dataset.to_csv(dataset_path, index=False)
        
        model_path = os.path.join(temp_output_dir, "test_model.joblib")
        metadata_path = os.path.join(temp_output_dir, "test_model.json")
        importance_path = os.path.join(temp_output_dir, "feature_importances.csv")
        
        # Import training function
        try:
            from scripts.train_safe_model import train_model
        except ImportError:
            pytest.skip("train_safe_model.py not available")
        
        # Run training
        results = train_model(
            dataset_path=dataset_path,
            model_type="xgboost",
            target_col="target_ticket_median",
            model_output_path=model_path,
            metadata_output_path=metadata_path,
            importance_output_path=importance_path,
            tune=False,
            save_shap=False,
            seed=42,
            verbose=False
        )
        
        assert results["success"], "Training did not complete successfully"
        assert os.path.exists(model_path), "Model file was not created"
        assert os.path.exists(metadata_path), "Metadata file was not created"
    
    def test_model_artifact_exists_after_training(self, synthetic_modelling_dataset, temp_output_dir):
        """Verify model artifacts are created correctly."""
        dataset_path = os.path.join(temp_output_dir, "test_dataset.csv")
        synthetic_modelling_dataset.to_csv(dataset_path, index=False)
        
        model_path = os.path.join(temp_output_dir, "test_model.joblib")
        metadata_path = os.path.join(temp_output_dir, "test_model.json")
        
        try:
            from scripts.train_safe_model import train_model
            
            train_model(
                dataset_path=dataset_path,
                model_output_path=model_path,
                metadata_output_path=metadata_path,
                tune=False,
                verbose=False
            )
        except ImportError:
            pytest.skip("train_safe_model.py not available")
        
        # Check artifacts
        assert os.path.exists(model_path), "Model artifact not created"
        assert os.path.getsize(model_path) > 0, "Model artifact is empty"
        
        assert os.path.exists(metadata_path), "Metadata not created"
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        assert "training_date" in metadata
        assert "features" in metadata
        assert "cv_metrics" in metadata
    
    def test_metadata_json_valid(self, synthetic_modelling_dataset, temp_output_dir):
        """Verify metadata JSON is valid and contains expected fields."""
        dataset_path = os.path.join(temp_output_dir, "test_dataset.csv")
        synthetic_modelling_dataset.to_csv(dataset_path, index=False)
        
        metadata_path = os.path.join(temp_output_dir, "test_model.json")
        model_path = os.path.join(temp_output_dir, "test_model.joblib")
        
        try:
            from scripts.train_safe_model import train_model
            
            train_model(
                dataset_path=dataset_path,
                model_output_path=model_path,
                metadata_output_path=metadata_path,
                tune=False,
                verbose=False
            )
        except ImportError:
            pytest.skip("train_safe_model.py not available")
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Check required fields
        required_fields = ["training_date", "model_type", "n_samples", "features", "cv_metrics"]
        for field in required_fields:
            assert field in metadata, f"Missing metadata field: {field}"
        
        # Check cv_metrics structure
        cv_metrics = metadata["cv_metrics"]
        assert "mae_mean" in cv_metrics
        assert "r2_mean" in cv_metrics
        assert cv_metrics["mae_mean"] > 0


class TestModelPrediction:
    """Tests for model predictions."""
    
    def test_predictions_return_numeric_array(self, synthetic_modelling_dataset, temp_output_dir):
        """Verify predictions return a numeric array without errors."""
        dataset_path = os.path.join(temp_output_dir, "test_dataset.csv")
        synthetic_modelling_dataset.to_csv(dataset_path, index=False)
        
        model_path = os.path.join(temp_output_dir, "test_model.joblib")
        
        try:
            from scripts.train_safe_model import train_model
            from ml.predict_utils import load_model_pipeline
            import joblib
        except ImportError:
            pytest.skip("Required modules not available")
        
        # Train model
        train_model(
            dataset_path=dataset_path,
            model_output_path=model_path,
            tune=False,
            verbose=False
        )
        
        # Load and predict
        pipeline = joblib.load(model_path)
        
        # Prepare test data
        test_data = synthetic_modelling_dataset.drop(
            columns=["title", "canonical_title", "target_ticket_median"],
            errors="ignore"
        ).head(5)
        
        predictions = pipeline.predict(test_data)
        
        # Verify predictions
        assert isinstance(predictions, np.ndarray), "Predictions should be numpy array"
        assert len(predictions) == 5, "Should have 5 predictions"
        assert np.issubdtype(predictions.dtype, np.number), "Predictions should be numeric"
        assert not np.any(np.isnan(predictions)), "Predictions should not contain NaN"
    
    def test_prediction_values_reasonable(self, synthetic_modelling_dataset, temp_output_dir):
        """Verify predictions are in a reasonable range."""
        dataset_path = os.path.join(temp_output_dir, "test_dataset.csv")
        synthetic_modelling_dataset.to_csv(dataset_path, index=False)
        
        model_path = os.path.join(temp_output_dir, "test_model.joblib")
        
        try:
            from scripts.train_safe_model import train_model
            import joblib
        except ImportError:
            pytest.skip("Required modules not available")
        
        train_model(
            dataset_path=dataset_path,
            model_output_path=model_path,
            tune=False,
            verbose=False
        )
        
        pipeline = joblib.load(model_path)
        
        test_data = synthetic_modelling_dataset.drop(
            columns=["title", "canonical_title", "target_ticket_median"],
            errors="ignore"
        ).head(5)
        
        # Predictions are in log space, need to transform back
        predictions_log = pipeline.predict(test_data)
        predictions = np.expm1(predictions_log)
        
        # Verify reasonable range (tickets should be positive and not astronomical)
        assert all(predictions > 0), "Predictions should be positive"
        assert all(predictions < 100000), "Predictions should be reasonable (< 100k)"


class TestConfigValidation:
    """Tests for configuration validation."""
    
    def test_validate_config_exists(self):
        """Test that validate_config function exists and is callable."""
        try:
            from config.validation import validate_config
            success, errors = validate_config()
            assert isinstance(success, bool)
            assert isinstance(errors, list)
        except ImportError:
            pytest.skip("config.validation module not available")
    
    def test_all_registry_files_exist(self):
        """Test that all required registry CSV files exist."""
        try:
            from config.validation import get_config_status
            status = get_config_status()
            
            for file_name, file_status in status.items():
                assert file_status["exists"], f"Missing config file: {file_name}"
        except ImportError:
            pytest.skip("config.validation module not available")
    
    def test_registry_files_have_required_columns(self):
        """Test that registry files have their required columns."""
        try:
            from config.validation import validate_config
            success, errors = validate_config()
            
            # Filter to just column-related errors
            column_errors = [e for e in errors if "Missing required column" in e]
            assert len(column_errors) == 0, f"Missing columns: {column_errors}"
        except ImportError:
            pytest.skip("config.validation module not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
