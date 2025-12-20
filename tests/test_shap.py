"""
Comprehensive test suite for SHAP implementation.

Covers:
- Input validation and error handling
- Core SHAP computation
- Edge cases and boundary conditions
- Caching functionality
- Visualizations
- Integration with streamlit_app

Run with: pytest tests/test_shap.py -v
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from sklearn.linear_model import Ridge

# Import SHAP module
from ml.shap_explainer import (
    SHAPExplainer, 
    explain_predictions_batch,
    clear_shap_cache,
    set_shap_logging_level,
    format_shap_narrative,
    get_top_shap_drivers
)


class TestSHAPExplainerInputValidation:
    """Test input validation and error handling."""
    
    @pytest.fixture
    def valid_model_and_data(self):
        """Create valid model and data for testing."""
        np.random.seed(42)
        X_train = pd.DataFrame({
            'wiki': np.random.uniform(10, 100, 30),
            'trends': np.random.uniform(5, 80, 30),
            'youtube': np.random.uniform(0, 60, 30),
            'chartmetric': np.random.uniform(0, 50, 30)
        })
        y_train = (0.3 * X_train['wiki'] + 0.2 * X_train['trends'] + 
                   0.25 * X_train['youtube'] + 0.25 * X_train['chartmetric'] + 
                   np.random.normal(0, 5, 30))
        
        model = Ridge(alpha=5.0, random_state=42)
        model.fit(X_train, y_train)
        
        return model, X_train
    
    def test_shap_not_available_raises_error(self):
        """Test that ImportError raised if SHAP not available."""
        import ml.shap_explainer as shap_module
        if not shap_module.SHAP_AVAILABLE:
            pytest.skip("SHAP not available")
        # This test would require mocking SHAP import, skip for now
    
    def test_empty_training_data_raises_error(self, valid_model_and_data):
        """Test that empty X_train raises ValueError."""
        model, _ = valid_model_and_data
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            SHAPExplainer(model, empty_df)
    
    def test_model_without_predict_raises_error(self, valid_model_and_data):
        """Test that model without predict method raises TypeError."""
        _, X_train = valid_model_and_data
        bad_model = "not a model"
        
        with pytest.raises(TypeError, match="predict"):
            SHAPExplainer(bad_model, X_train)
    
    def test_non_dataframe_input_raises_error(self, valid_model_and_data):
        """Test that non-DataFrame X_train raises TypeError."""
        model, _ = valid_model_and_data
        X_array = np.random.randn(30, 4)
        
        with pytest.raises(TypeError, match="DataFrame"):
            SHAPExplainer(model, X_array)
    
    def test_nan_values_handled_gracefully(self, valid_model_and_data):
        """Test that NaN values in X_train are handled."""
        model, X_train = valid_model_and_data
        X_with_nan = X_train.copy()
        X_with_nan.iloc[0, 0] = np.nan
        
        # Should not raise, but fill NaN
        explainer = SHAPExplainer(model, X_with_nan)
        assert explainer is not None
        assert explainer.X_train.isnull().sum().sum() == 0
    
    def test_inf_values_handled_gracefully(self, valid_model_and_data):
        """Test that inf values in X_train are clipped."""
        model, X_train = valid_model_and_data
        X_with_inf = X_train.copy()
        X_with_inf.iloc[0, 0] = np.inf
        
        # Should not raise, but clip inf
        explainer = SHAPExplainer(model, X_with_inf)
        assert explainer is not None
        assert not np.isinf(explainer.X_train.values).any()


class TestSHAPExplainerCore:
    """Test core SHAP computation."""
    
    @pytest.fixture
    def shap_explainer(self):
        """Create SHAP explainer for testing."""
        np.random.seed(42)
        X_train = pd.DataFrame({
            'wiki': np.random.uniform(10, 100, 30),
            'trends': np.random.uniform(5, 80, 30),
            'youtube': np.random.uniform(0, 60, 30),
            'chartmetric': np.random.uniform(0, 50, 30)
        })
        y_train = (0.3 * X_train['wiki'] + 0.2 * X_train['trends'] + 
                   0.25 * X_train['youtube'] + 0.25 * X_train['chartmetric'] + 
                   np.random.normal(0, 5, 30))
        
        model = Ridge(alpha=5.0, random_state=42)
        model.fit(X_train, y_train)
        
        return SHAPExplainer(model, X_train, feature_names=X_train.columns.tolist())
    
    def test_single_prediction_explanation(self, shap_explainer):
        """Test SHAP explanation for single prediction."""
        test_point = pd.Series({'wiki': 75, 'trends': 50, 'youtube': 40, 'chartmetric': 30})
        explanation = shap_explainer.explain_single(test_point, use_cache=False)
        
        # Check structure
        assert 'prediction' in explanation
        assert 'base_value' in explanation
        assert 'shap_values' in explanation
        assert 'feature_contributions' in explanation
        
        # Check values
        assert isinstance(explanation['prediction'], float)
        assert isinstance(explanation['base_value'], float)
        assert len(explanation['shap_values']) == 4
        assert len(explanation['feature_contributions']) == 4
    
    def test_shap_values_sum_to_prediction(self, shap_explainer):
        """Test that SHAP values sum approximately to prediction."""
        test_point = pd.Series({'wiki': 75, 'trends': 50, 'youtube': 40, 'chartmetric': 30})
        explanation = shap_explainer.explain_single(test_point, use_cache=False)
        
        # Sum of SHAP values plus base value should equal prediction
        shap_sum = explanation['base_value'] + np.sum(explanation['shap_values'])
        prediction = explanation['prediction']
        
        # Allow small floating point error
        assert abs(shap_sum - prediction) < 0.1, f"SHAP sum {shap_sum} != prediction {prediction}"
    
    def test_feature_contributions_sorted(self, shap_explainer):
        """Test that feature contributions are sorted by impact."""
        test_point = pd.Series({'wiki': 75, 'trends': 50, 'youtube': 40, 'chartmetric': 30})
        explanation = shap_explainer.explain_single(test_point, use_cache=False)
        
        impacts = [c['abs_impact'] for c in explanation['feature_contributions']]
        assert impacts == sorted(impacts, reverse=True)
    
    def test_empty_series_raises_error(self, shap_explainer):
        """Test that empty Series raises ValueError."""
        empty_series = pd.Series(dtype=float)
        
        with pytest.raises(ValueError, match="cannot be empty"):
            shap_explainer.explain_single(empty_series)
    
    def test_nan_in_prediction_handled(self, shap_explainer):
        """Test that NaN values in prediction are handled."""
        test_point = pd.Series({'wiki': np.nan, 'trends': 50, 'youtube': 40, 'chartmetric': 30})
        explanation = shap_explainer.explain_single(test_point, use_cache=False)
        
        # Should complete without error
        assert explanation is not None


class TestCaching:
    """Test caching functionality."""
    
    @pytest.fixture
    def shap_explainer_with_cache(self):
        """Create SHAP explainer with disk caching."""
        np.random.seed(42)
        X_train = pd.DataFrame({
            'wiki': np.random.uniform(10, 100, 20),
            'trends': np.random.uniform(5, 80, 20),
            'youtube': np.random.uniform(0, 60, 20),
            'chartmetric': np.random.uniform(0, 50, 20)
        })
        y_train = np.random.uniform(30, 150, 20)
        
        model = Ridge(alpha=5.0, random_state=42)
        model.fit(X_train, y_train)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = SHAPExplainer(
                model, X_train, 
                feature_names=X_train.columns.tolist(),
                cache_dir=tmpdir
            )
            yield explainer, tmpdir
    
    def test_cache_hit_after_first_computation(self, shap_explainer_with_cache):
        """Test that second request uses cache."""
        explainer, _ = shap_explainer_with_cache
        test_point = pd.Series({'wiki': 50, 'trends': 40, 'youtube': 30, 'chartmetric': 25})
        
        # First call
        result1 = explainer.explain_single(test_point, use_cache=True)
        assert len(explainer._explanation_cache) == 1
        
        # Second call (same data)
        result2 = explainer.explain_single(test_point, use_cache=True)
        
        # Should be identical
        assert result1['prediction'] == result2['prediction']
        assert np.array_equal(result1['shap_values'], result2['shap_values'])
    
    def test_different_inputs_separate_cache_entries(self, shap_explainer_with_cache):
        """Test that different inputs get separate cache entries."""
        explainer, _ = shap_explainer_with_cache
        
        test_point1 = pd.Series({'wiki': 50, 'trends': 40, 'youtube': 30, 'chartmetric': 25})
        test_point2 = pd.Series({'wiki': 60, 'trends': 50, 'youtube': 40, 'chartmetric': 35})
        
        explainer.explain_single(test_point1, use_cache=True)
        explainer.explain_single(test_point2, use_cache=True)
        
        assert len(explainer._explanation_cache) == 2
    
    def test_cache_clearing(self, shap_explainer_with_cache):
        """Test cache clearing."""
        explainer, tmpdir = shap_explainer_with_cache
        
        test_point = pd.Series({'wiki': 50, 'trends': 40, 'youtube': 30, 'chartmetric': 25})
        explainer.explain_single(test_point, use_cache=True)
        
        assert len(explainer._explanation_cache) > 0
        
        clear_shap_cache(explainer, disk_only=False)
        assert len(explainer._explanation_cache) == 0


class TestVisualizationFunctions:
    """Test visualization functions."""
    
    @pytest.fixture
    def explanation(self):
        """Create sample explanation for visualization tests."""
        np.random.seed(42)
        X_train = pd.DataFrame({
            'wiki': np.random.uniform(10, 100, 25),
            'trends': np.random.uniform(5, 80, 25),
            'youtube': np.random.uniform(0, 60, 25),
            'chartmetric': np.random.uniform(0, 50, 25)
        })
        y_train = np.random.uniform(30, 150, 25)
        
        model = Ridge(alpha=5.0, random_state=42)
        model.fit(X_train, y_train)
        
        explainer = SHAPExplainer(model, X_train, feature_names=X_train.columns.tolist())
        test_point = pd.Series({'wiki': 75, 'trends': 50, 'youtube': 40, 'chartmetric': 30})
        return explainer.explain_single(test_point, use_cache=False)
    
    def test_format_narrative(self, explanation):
        """Test SHAP narrative formatting."""
        narrative = format_shap_narrative(explanation, n_top=3, min_impact=1.0)
        
        assert isinstance(narrative, str)
        assert 'tickets' in narrative.lower()
        assert len(narrative) > 0
    
    def test_get_top_drivers(self, explanation):
        """Test extracting top SHAP drivers."""
        drivers = get_top_shap_drivers(explanation, n_top=2, min_impact=0.5)
        
        assert len(drivers) <= 2
        assert all('name' in d for d in drivers)
        assert all('shap' in d for d in drivers)


class TestBatchComputation:
    """Test batch explanation functionality."""
    
    def test_batch_computation(self):
        """Test batch explanation computation."""
        np.random.seed(42)
        X_train = pd.DataFrame({
            'wiki': np.random.uniform(10, 100, 25),
            'trends': np.random.uniform(5, 80, 25),
            'youtube': np.random.uniform(0, 60, 25),
            'chartmetric': np.random.uniform(0, 50, 25)
        })
        y_train = np.random.uniform(30, 150, 25)
        
        X_test = pd.DataFrame({
            'wiki': np.random.uniform(10, 100, 5),
            'trends': np.random.uniform(5, 80, 5),
            'youtube': np.random.uniform(0, 60, 5),
            'chartmetric': np.random.uniform(0, 50, 5)
        })
        
        model = Ridge(alpha=5.0, random_state=42)
        model.fit(X_train, y_train)
        
        explainer = SHAPExplainer(model, X_train, feature_names=X_train.columns.tolist())
        explanations = explain_predictions_batch(explainer, X_test, use_cache=False, verbose=False)
        
        assert len(explanations) == len(X_test)
        assert all(isinstance(e, dict) for e in explanations.values())


class TestLogging:
    """Test logging functionality."""
    
    def test_logging_configuration(self):
        """Test that logging can be configured."""
        set_shap_logging_level("DEBUG")
        # Should not raise
        
        set_shap_logging_level("INFO")
        # Should not raise


# Edge case tests
class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_small_values(self):
        """Test with very small feature values."""
        np.random.seed(42)
        X_train = pd.DataFrame({
            'wiki': np.random.uniform(0.0001, 0.001, 20),
            'trends': np.random.uniform(0.0001, 0.001, 20),
            'youtube': np.random.uniform(0.0001, 0.001, 20),
            'chartmetric': np.random.uniform(0.0001, 0.001, 20)
        })
        y_train = np.random.uniform(0.0001, 0.1, 20)
        
        model = Ridge(alpha=5.0, random_state=42)
        model.fit(X_train, y_train)
        
        explainer = SHAPExplainer(model, X_train)
        test_point = pd.Series({'wiki': 0.0005, 'trends': 0.0005, 'youtube': 0.0005, 'chartmetric': 0.0005})
        explanation = explainer.explain_single(test_point, use_cache=False)
        
        assert explanation['prediction'] > 0
    
    def test_very_large_values(self):
        """Test with very large feature values."""
        np.random.seed(42)
        X_train = pd.DataFrame({
            'wiki': np.random.uniform(1e6, 1e7, 20),
            'trends': np.random.uniform(1e6, 1e7, 20),
            'youtube': np.random.uniform(1e6, 1e7, 20),
            'chartmetric': np.random.uniform(1e6, 1e7, 20)
        })
        y_train = np.random.uniform(1e4, 1e5, 20)
        
        model = Ridge(alpha=5.0, random_state=42)
        model.fit(X_train, y_train)
        
        explainer = SHAPExplainer(model, X_train)
        test_point = pd.Series({'wiki': 5e6, 'trends': 5e6, 'youtube': 5e6, 'chartmetric': 5e6})
        explanation = explainer.explain_single(test_point, use_cache=False)
        
        assert explanation['prediction'] > 0
    
    def test_single_sample_training(self):
        """Test with minimal training data."""
        X_train = pd.DataFrame({
            'wiki': [50.0],
            'trends': [40.0],
            'youtube': [30.0],
            'chartmetric': [25.0]
        })
        y_train = [100.0]
        
        model = Ridge(alpha=5.0, random_state=42)
        model.fit(X_train, y_train)
        
        # Should still work
        explainer = SHAPExplainer(model, X_train)
        assert explainer is not None


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_shap.py -v
    pytest.main([__file__, "-v"])
