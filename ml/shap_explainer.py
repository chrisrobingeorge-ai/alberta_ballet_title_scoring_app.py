"""
SHAP Explainer Module

Computes SHAP (SHapley Additive exPlanations) values for per-prediction explanations.
Transforms Ridge regression predictions into interpretable feature contributions.

This module enables per-show explanations like:
  "185 tickets because: Prior sales (+58), Wiki (+15), Trends (-8), YouTube (+2)"

Author: Alberta Ballet Data Science Team
Date: December 2025
"""

from typing import Dict, Tuple, Optional, Any, List
import numpy as np
import pandas as pd
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not installed. Install with: pip install shap")


class SHAPExplainer:
    """
    Wrapper for SHAP explanations on Ridge regression models.
    
    Handles:
    - Creating SHAP explainer from training data
    - Computing SHAP values for individual predictions
    - Generating structured explanation data
    - Caching for performance optimization
    """
    
    def __init__(self, model, X_train: pd.DataFrame, feature_names: Optional[List[str]] = None):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained sklearn model (typically Ridge regression)
            X_train: Training data for background (shape: n_samples x n_features)
            feature_names: Optional list of feature names. If None, uses X_train.columns
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Run: pip install shap")
        
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or list(X_train.columns)
        
        # Create explainer (uses SHAP's kernel explainer by default)
        # For Ridge regression: KernelExplainer is model-agnostic and accurate
        # Use sample of training data for speed (default 100 samples)
        n_background = min(100, len(X_train))
        background_data = shap.sample(X_train, n_background)
        
        self.explainer = shap.KernelExplainer(
            model=self.model.predict,
            data=background_data,
            link="identity"  # Linear output (ticket counts)
        )
        
        # Cache base value (expected model output)
        self.base_value = self.explainer.expected_value
    
    def explain(self, X_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute SHAP values and explanations for test data.
        
        Args:
            X_test: Test data (shape: 1 x n_features for single prediction, or n x n_features)
        
        Returns:
            Dictionary containing:
            - 'predictions': Predicted ticket counts
            - 'base_value': Global mean prediction (expected value)
            - 'shap_values': SHAP contributions per feature
            - 'feature_names': Names of features
            - 'feature_values': Actual values in X_test
        """
        # Ensure 2D
        if len(X_test.shape) == 1:
            X_test = X_test.values.reshape(1, -1)
            X_test_df = pd.DataFrame(X_test, columns=self.feature_names)
        else:
            X_test_df = X_test
        
        # Get predictions
        predictions = self.model.predict(X_test_df)
        
        # Compute SHAP values
        shap_values_obj = self.explainer.shap_values(X_test_df)
        
        return {
            'predictions': predictions,
            'base_value': self.base_value,
            'shap_values': shap_values_obj,
            'feature_names': self.feature_names,
            'feature_values': X_test_df.values,
            'X_test': X_test_df
        }
    
    def explain_single(self, X_single: pd.Series) -> Dict[str, Any]:
        """
        Compute SHAP explanation for a single prediction.
        
        Args:
            X_single: Single row (pd.Series with feature names)
        
        Returns:
            Structured explanation for that single prediction
        """
        X_df = pd.DataFrame([X_single])
        result = self.explain(X_df)
        
        # Flatten result for single prediction
        return {
            'prediction': float(result['predictions'][0]),
            'base_value': float(self.base_value),
            'shap_values': result['shap_values'][0],
            'feature_names': result['feature_names'],
            'feature_values': dict(zip(result['feature_names'], result['feature_values'][0]))
        }


def get_top_shap_drivers(
    explanation: Dict[str, Any],
    n_top: int = 5,
    min_impact: float = 0.5
) -> List[Tuple[str, float, str]]:
    """
    Extract top N feature contributions from SHAP explanation.
    
    Args:
        explanation: Output from SHAPExplainer.explain_single()
        n_top: Number of top drivers to return
        min_impact: Minimum absolute SHAP value to include (in ticket units)
    
    Returns:
        List of tuples: [(feature_name, shap_value, direction), ...]
        where direction is "up" or "down"
    """
    shap_values = explanation['shap_values']
    feature_names = explanation['feature_names']
    
    # Create ranked list
    drivers = []
    for feat_name, shap_val in zip(feature_names, shap_values):
        abs_impact = abs(shap_val)
        if abs_impact >= min_impact:
            direction = "up" if shap_val > 0 else "down"
            drivers.append((feat_name, shap_val, direction))
    
    # Sort by absolute impact
    drivers.sort(key=lambda x: abs(x[1]), reverse=True)
    
    return drivers[:n_top]


def format_shap_narrative(
    explanation: Dict[str, Any],
    n_top: int = 5,
    min_impact: float = 1.0
) -> str:
    """
    Convert SHAP explanation into human-readable narrative.
    
    Args:
        explanation: Output from SHAPExplainer.explain_single()
        n_top: Number of top factors to include
        min_impact: Minimum impact threshold (in ticket units)
    
    Returns:
        Formatted narrative string explaining the prediction
    
    Example output:
        "185 tickets because: Prior sales +58, Wiki +15, Trends -8, YouTube +2"
    """
    prediction = explanation['prediction']
    base_value = explanation['base_value']
    
    drivers = get_top_shap_drivers(explanation, n_top=n_top, min_impact=min_impact)
    
    if not drivers:
        return f"{prediction:.0f} tickets (insufficient signal to decompose)"
    
    # Format each driver
    driver_parts = []
    for feat_name, shap_val, direction in drivers:
        # Clean up feature name for readability
        display_name = feat_name.replace("_", " ").title()
        
        # Format impact
        if shap_val > 0:
            impact_str = f"+{shap_val:.0f}"
        else:
            impact_str = f"{shap_val:.0f}"
        
        driver_parts.append(f"{display_name} {impact_str}")
    
    drivers_text = ", ".join(driver_parts)
    
    return f"{prediction:.0f} tickets because: {drivers_text}"


def explain_predictions(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    verbose: bool = False
) -> Dict[int, Dict[str, Any]]:
    """
    High-level function: Create SHAP explainer and compute explanations for multiple predictions.
    
    Args:
        model: Trained sklearn regressor
        X_train: Training data (for SHAP background)
        X_test: Test data to explain
        feature_names: Optional custom feature names
        verbose: Print progress
    
    Returns:
        Dictionary mapping row index to explanation dictionaries
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP not installed. Run: pip install shap")
    
    if verbose:
        print(f"Creating SHAP explainer from {len(X_train)} training samples...")
    
    explainer = SHAPExplainer(model, X_train, feature_names=feature_names)
    
    if verbose:
        print(f"Computing SHAP values for {len(X_test)} predictions...")
    
    explanations = {}
    for idx, (_, row) in enumerate(X_test.iterrows()):
        explanations[idx] = explainer.explain_single(row)
    
    if verbose:
        print("✓ SHAP explanation complete")
    
    return explanations
