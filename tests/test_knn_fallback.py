"""
Tests for KNN fallback module.
"""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_baseline_data():
    """Create sample baseline data for testing."""
    return pd.DataFrame({
        "title": ["Swan Lake", "Nutcracker", "Romeo and Juliet", "Giselle", "Sleeping Beauty"],
        "wiki": [80, 95, 86, 74, 77],
        "trends": [18, 45, 32, 14, 30],
        "youtube": [71, 88, 80, 62, 70],
        "spotify": [71, 75, 80, 62, 70],
        "ticket_median": [9000, 12000, 8500, 6000, 7500],
    })


def test_knn_fallback_import():
    """Test that KNNFallback can be imported."""
    try:
        from ml.knn_fallback import KNNFallback, build_knn_index, predict_knn
        assert KNNFallback is not None
        assert build_knn_index is not None
        assert predict_knn is not None
    except ImportError as e:
        pytest.skip(f"KNN module not available: {e}")


def test_knn_fallback_build_index(sample_baseline_data):
    """Test building KNN index from baseline data."""
    try:
        from ml.knn_fallback import KNNFallback
    except ImportError:
        pytest.skip("KNN module not available")
    
    knn = KNNFallback(k=3, metric="cosine")
    knn.build_index(sample_baseline_data, outcome_col="ticket_median")
    
    assert knn._is_fitted
    assert knn._index_df is not None
    assert len(knn._index_df) == 5


def test_knn_fallback_predict_basic(sample_baseline_data):
    """Test basic KNN prediction."""
    try:
        from ml.knn_fallback import KNNFallback
    except ImportError:
        pytest.skip("KNN module not available")
    
    knn = KNNFallback(k=3, metric="cosine")
    knn.build_index(sample_baseline_data, outcome_col="ticket_median")
    
    # Test prediction with similar baseline
    new_show = {"wiki": 78, "trends": 25, "youtube": 75, "spotify": 65}
    prediction = knn.predict(new_show)
    
    # Should return a reasonable value in the range of the data
    assert isinstance(prediction, float)
    assert 5000 <= prediction <= 13000


def test_knn_fallback_predict_with_neighbors(sample_baseline_data):
    """Test prediction with neighbor info returned."""
    try:
        from ml.knn_fallback import KNNFallback
    except ImportError:
        pytest.skip("KNN module not available")
    
    knn = KNNFallback(k=3, metric="cosine")
    knn.build_index(sample_baseline_data, outcome_col="ticket_median")
    
    new_show = {"wiki": 78, "trends": 25, "youtube": 75, "spotify": 65}
    prediction, neighbors = knn.predict(new_show, return_neighbors=True)
    
    assert isinstance(prediction, float)
    assert isinstance(neighbors, pd.DataFrame)
    assert len(neighbors) == 3  # k=3
    assert "distance" in neighbors.columns
    assert "similarity" in neighbors.columns
    assert "weight" in neighbors.columns


def test_knn_fallback_predict_handles_nan():
    """Test that KNN handles NaN values gracefully."""
    try:
        from ml.knn_fallback import KNNFallback
    except ImportError:
        pytest.skip("KNN module not available")
    
    data = pd.DataFrame({
        "wiki": [80, np.nan, 86],
        "trends": [18, 45, np.nan],
        "youtube": [71, 88, 80],
        "spotify": [71, 75, 80],
        "ticket_median": [9000, 12000, 8500],
    })
    
    knn = KNNFallback(k=2, metric="cosine")
    knn.build_index(data, outcome_col="ticket_median")
    
    # Should still work with NaN in input
    prediction = knn.predict({"wiki": 75, "trends": 30, "youtube": np.nan, "spotify": 70})
    assert isinstance(prediction, float)
    assert not np.isnan(prediction)


def test_knn_fallback_empty_index():
    """Test behavior with empty data."""
    try:
        from ml.knn_fallback import KNNFallback
    except ImportError:
        pytest.skip("KNN module not available")
    
    empty_data = pd.DataFrame({
        "wiki": [],
        "trends": [],
        "youtube": [],
        "spotify": [],
        "ticket_median": [],
    })
    
    knn = KNNFallback(k=3)
    knn.build_index(empty_data, outcome_col="ticket_median")
    
    # With empty index, predict should handle gracefully (return NaN)
    # Note: _is_fitted may be False for empty data
    if knn._is_fitted:
        prediction = knn.predict({"wiki": 75, "trends": 30, "youtube": 70, "spotify": 70})
        assert np.isnan(prediction)
    else:
        # Empty data doesn't fit the model - that's OK
        pass


def test_knn_fallback_no_valid_outcomes():
    """Test behavior when all outcomes are NaN."""
    try:
        from ml.knn_fallback import KNNFallback
    except ImportError:
        pytest.skip("KNN module not available")
    
    data = pd.DataFrame({
        "wiki": [80, 90, 86],
        "trends": [18, 45, 32],
        "youtube": [71, 88, 80],
        "spotify": [71, 75, 80],
        "ticket_median": [np.nan, np.nan, np.nan],
    })
    
    knn = KNNFallback(k=2)
    # Should not error, just not fit
    knn.build_index(data, outcome_col="ticket_median")
    
    # With no valid outcomes, index won't be fitted - that's OK
    if knn._is_fitted:
        prediction = knn.predict({"wiki": 75, "trends": 30, "youtube": 70, "spotify": 70})
        assert np.isnan(prediction)
    else:
        # No valid outcomes means no index built - that's expected
        pass


def test_build_knn_index_function(sample_baseline_data):
    """Test the convenience function."""
    try:
        from ml.knn_fallback import build_knn_index
    except ImportError:
        pytest.skip("KNN module not available")
    
    knn = build_knn_index(sample_baseline_data, outcome_col="ticket_median")
    assert knn._is_fitted


def test_predict_knn_function(sample_baseline_data):
    """Test the convenience prediction function."""
    try:
        from ml.knn_fallback import build_knn_index, predict_knn
    except ImportError:
        pytest.skip("KNN module not available")
    
    knn = build_knn_index(sample_baseline_data, outcome_col="ticket_median")
    
    prediction = predict_knn(
        {"wiki": 78, "trends": 25, "youtube": 75, "spotify": 65},
        knn_index=knn,
        k=3
    )
    
    assert isinstance(prediction, float)
    assert 5000 <= prediction <= 13000
