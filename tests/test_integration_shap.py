"""
Integration tests for SHAP with streamlit_app

Tests that the enhanced SHAP implementation works correctly
with the main streamlit_app.py module.

Run with: python -m pytest tests/test_integration_shap.py -v
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.linear_model import Ridge
from ml.shap_explainer import SHAPExplainer, explain_predictions_batch, format_shap_narrative, get_top_shap_drivers
from ml.title_explanation_engine import build_title_explanation


class TestSHAPIntegration:
    """Integration tests for SHAP with title explanation."""
    
    @pytest.fixture
    def setup_models(self):
        """Setup models and data for integration testing."""
        np.random.seed(42)
        
        # Create training data (4 features: wiki, trends, youtube, chartmetric)
        X_train = pd.DataFrame(
            np.random.uniform(10, 100, (30, 4)),
            columns=['wiki_mentions', 'google_trends', 'youtube_views', 'chartmetric_score']
        )
        y_train = np.random.uniform(30, 150, 30)
        
        # Create test data
        X_test = pd.DataFrame(
            np.random.uniform(10, 100, (5, 4)),
            columns=['wiki_mentions', 'google_trends', 'youtube_views', 'chartmetric_score']
        )
        
        # Train model
        model = Ridge(alpha=5.0, random_state=42)
        model.fit(X_train, y_train)
        
        return {
            'model': model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train
        }
    
    def test_shap_with_title_explanation(self, setup_models):
        """Test SHAP integration with title explanation."""
        data = setup_models
        
        # Create SHAP explainer
        explainer = SHAPExplainer(
            data['model'],
            data['X_train'],
            feature_names=data['X_train'].columns.tolist()
        )
        
        # Test single prediction explanation
        test_row = data['X_test'].iloc[0]
        explanation = explainer.explain_single(test_row)
        
        # Verify explanation structure
        assert 'prediction' in explanation
        assert 'base_value' in explanation
        assert 'feature_contributions' in explanation
        assert len(explanation['feature_contributions']) > 0
        
        # Test with title explanation function
        narrative = format_shap_narrative(explanation)
        assert isinstance(narrative, str)
        assert len(narrative) > 0
    
    def test_batch_explanations_with_engine(self, setup_models):
        """Test batch SHAP explanations with engine."""
        data = setup_models
        
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = SHAPExplainer(
                data['model'],
                data['X_train'],
                feature_names=data['X_train'].columns.tolist(),
                cache_dir=tmpdir
            )
            
            # Get batch explanations (returns dict with index keys)
            explanations = explain_predictions_batch(
                explainer, data['X_test'], use_cache=True, verbose=False
            )
            
            # Verify batch results
            assert len(explanations) == len(data['X_test'])
            
            for i in range(len(data['X_test'])):
                exp = explanations[i]
                
                assert 'prediction' in exp
                assert 'base_value' in exp
                assert 'feature_contributions' in exp
                assert len(exp['feature_contributions']) > 0
                
                # Verify prediction matches model output
                pred = data['model'].predict(data['X_test'].iloc[[i]])[0]
                assert abs(exp['prediction'] - pred) < 0.01
    
    def test_cache_consistency(self, setup_models):
        """Test that caching doesn't affect explanation quality."""
        data = setup_models
        
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = SHAPExplainer(
                data['model'],
                data['X_train'],
                feature_names=data['X_train'].columns.tolist(),
                cache_dir=tmpdir
            )
            
            test_row = data['X_test'].iloc[0]
            
            # First explanation (populates cache)
            exp1 = explainer.explain_single(test_row, use_cache=True)
            
            # Second explanation (uses cache)
            exp2 = explainer.explain_single(test_row, use_cache=True)
            
            # Verify they're identical
            assert exp1['prediction'] == exp2['prediction']
            assert exp1['base_value'] == exp2['base_value']
            assert len(exp1['feature_contributions']) == len(exp2['feature_contributions'])
            
            for fc1, fc2 in zip(
                exp1['feature_contributions'],
                exp2['feature_contributions']
            ):
                assert fc1['name'] == fc2['name']
                assert abs(fc1['shap'] - fc2['shap']) < 1e-10
    
    def test_error_handling_in_batch(self, setup_models):
        """Test error handling during batch explanations."""
        data = setup_models
        
        explainer = SHAPExplainer(
            data['model'],
            data['X_train'],
            feature_names=data['X_train'].columns.tolist()
        )
        
        # Create test data with NaN
        X_test_with_nan = data['X_test'].copy()
        X_test_with_nan.iloc[2, 0] = np.nan
        
        # Should handle gracefully
        explanations = explain_predictions_batch(
            explainer, X_test_with_nan, use_cache=False, verbose=False
        )
        
        # Verify all explanations are valid (dict with index keys)
        assert len(explanations) == len(X_test_with_nan)
        for i in range(len(X_test_with_nan)):
            exp = explanations[i]
            assert 'prediction' in exp
            assert not np.isnan(exp['prediction'])
    
    def test_feature_importance_order(self, setup_models):
        """Test that features are ordered by importance."""
        data = setup_models
        
        explainer = SHAPExplainer(
            data['model'],
            data['X_train'],
            feature_names=data['X_train'].columns.tolist()
        )
        
        test_row = data['X_test'].iloc[0]
        explanation = explainer.explain_single(test_row)
        
        # Verify features are sorted by absolute contribution (SHAP value)
        contributions = explanation['feature_contributions']
        abs_contributions = [c['abs_impact'] for c in contributions]
        
        assert abs_contributions == sorted(abs_contributions, reverse=True)
    
    def test_top_drivers_extraction(self, setup_models):
        """Test extraction of top SHAP drivers."""
        data = setup_models
        
        explainer = SHAPExplainer(
            data['model'],
            data['X_train'],
            feature_names=data['X_train'].columns.tolist()
        )
        
        test_row = data['X_test'].iloc[0]
        explanation = explainer.explain_single(test_row)
        
        # Get top 2 drivers
        top_n = 2
        contributions = explanation['feature_contributions'][:top_n]
        
        assert len(contributions) <= top_n
        for c in contributions:
            assert 'name' in c
            assert 'shap' in c
            assert isinstance(c['name'], str)
            assert isinstance(c['shap'], (int, float))
    
    def test_prediction_accuracy(self, setup_models):
        """Test that SHAP predictions match model predictions."""
        data = setup_models
        
        explainer = SHAPExplainer(
            data['model'],
            data['X_train'],
            feature_names=data['X_train'].columns.tolist()
        )
        
        for i in range(min(5, len(data['X_test']))):
            test_row = data['X_test'].iloc[i]
            explanation = explainer.explain_single(test_row)
            
            # Get model prediction
            model_pred = data['model'].predict(data['X_test'].iloc[[i]])[0]
            
            # Get SHAP prediction
            shap_pred = explanation['prediction']
            
            # Should be very close (within floating point error)
            assert abs(model_pred - shap_pred) < 0.01, \
                f"Prediction mismatch: model={model_pred}, shap={shap_pred}"


class TestSHAPEdgeCasesIntegration:
    """Integration tests for edge cases in SHAP."""
    
    @pytest.fixture
    def setup_edge_models(self):
        """Setup models for edge case testing."""
        np.random.seed(42)
        
        # Minimal training data
        X_train = pd.DataFrame(
            [[10, 20, 30, 40], [50, 60, 70, 80]],
            columns=['f1', 'f2', 'f3', 'f4']
        )
        y_train = [100, 120]
        
        model = Ridge(alpha=5.0, random_state=42)
        model.fit(X_train, y_train)
        
        # Test data
        X_test = pd.DataFrame(
            [[15, 25, 35, 45]],
            columns=['f1', 'f2', 'f3', 'f4']
        )
        
        return {
            'model': model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train
        }
    
    def test_minimal_training_data(self, setup_edge_models):
        """Test SHAP with minimal training data (edge case)."""
        data = setup_edge_models
        
        # Should handle minimal training data
        explainer = SHAPExplainer(
            data['model'],
            data['X_train'],
            feature_names=['f1', 'f2', 'f3', 'f4']
        )
        
        test_row = data['X_test'].iloc[0]
        explanation = explainer.explain_single(test_row)
        
        # Verify valid explanation
        assert 'prediction' in explanation
        assert not np.isnan(explanation['prediction'])
        assert len(explanation['feature_contributions']) > 0
    
    def test_zero_values_in_features(self, setup_edge_models):
        """Test SHAP with zero values in features."""
        data = setup_edge_models
        
        explainer = SHAPExplainer(
            data['model'],
            data['X_train'],
            feature_names=['f1', 'f2', 'f3', 'f4']
        )
        
        # Test row with zero values
        X_test_zeros = pd.DataFrame(
            [[0, 0, 0, 0]],
            columns=['f1', 'f2', 'f3', 'f4']
        )
        
        test_row = X_test_zeros.iloc[0]
        explanation = explainer.explain_single(test_row)
        
        # Should still produce valid explanation
        assert 'prediction' in explanation
        assert isinstance(explanation['prediction'], (int, float))


class TestSHAPPerformanceIntegration:
    """Integration tests for SHAP performance."""
    
    def test_large_batch_computation(self):
        """Test SHAP with larger batch of predictions."""
        np.random.seed(42)
        
        # Create larger dataset
        X_train = pd.DataFrame(
            np.random.uniform(10, 100, (100, 4)),
            columns=['f1', 'f2', 'f3', 'f4']
        )
        y_train = np.random.uniform(30, 150, 100)
        
        X_test = pd.DataFrame(
            np.random.uniform(10, 100, (50, 4)),
            columns=['f1', 'f2', 'f3', 'f4']
        )
        
        # Train model
        model = Ridge(alpha=5.0, random_state=42)
        model.fit(X_train, y_train)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = SHAPExplainer(
                model, X_train,
                feature_names=['f1', 'f2', 'f3', 'f4'],
                cache_dir=tmpdir
            )
            
            # Batch computation (returns dict with index keys)
            explanations = explain_predictions_batch(
                explainer, X_test, use_cache=True, verbose=False
            )
            
            # Verify all predictions
            assert len(explanations) == len(X_test)
            for i in range(len(X_test)):
                exp = explanations[i]
                assert 'prediction' in exp
                assert not np.isnan(exp['prediction'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
