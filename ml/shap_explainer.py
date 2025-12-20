"""
SHAP Explainer Module

Computes SHAP (SHapley Additive exPlanations) values for per-prediction explanations.
Transforms Ridge regression predictions into interpretable feature contributions.

Multi-feature support: Explains individual signal contributions (wiki, trends, youtube, chartmetric)

Example output:
  "185 tickets because: 
   - Wiki search: +15 (growing public interest)
   - Prior sales: +58 (baseline expectation)
   - Google Trends: -8 (search volume declining)
   - YouTube: +2 (modest engagement)
   - Chartmetric: +0 (streaming activity stable)"

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
    
    Supports multi-feature explanations by computing SHAP values for each 
    input feature and showing its contribution to the final prediction.
    
    Handles:
    - Creating SHAP explainer from training data
    - Computing SHAP values for individual predictions
    - Generating structured explanation data
    - Caching for performance optimization
    - Multi-feature decomposition (wiki, trends, youtube, chartmetric, etc.)
    """
    
    def __init__(
        self, 
        model, 
        X_train: pd.DataFrame, 
        feature_names: Optional[List[str]] = None,
        sample_size: int = 100
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained sklearn model (typically Ridge regression)
            X_train: Training data for background (shape: n_samples x n_features)
            feature_names: Optional list of feature names. If None, uses X_train.columns
            sample_size: Number of background samples to use (max 100 for speed)
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Run: pip install shap")
        
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or list(X_train.columns)
        self.n_features = len(self.feature_names)
        
        # Create explainer (uses SHAP's kernel explainer by default)
        # For Ridge regression: KernelExplainer is model-agnostic and accurate
        n_background = min(sample_size, len(X_train))
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
            - 'shap_values': SHAP contributions per feature (n_samples x n_features)
            - 'feature_names': Names of features
            - 'feature_values': Actual values in X_test (n_samples x n_features)
            - 'X_test': Full DataFrame for reference
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
            Structured explanation for that single prediction:
            {
                'prediction': float,
                'base_value': float,
                'shap_values': np.array (1D),
                'feature_names': list,
                'feature_values': dict,
                'feature_contributions': list of dicts with keys:
                    - name: feature name
                    - value: input value
                    - shap: contribution amount
                    - direction: "up" or "down"
                    - abs_impact: absolute SHAP value
            }
        """
        X_df = pd.DataFrame([X_single])
        result = self.explain(X_df)
        
        # Build feature contributions list
        feature_contributions = []
        for fname, fval, shap_val in zip(
            result['feature_names'],
            result['feature_values'][0],
            result['shap_values'][0]
        ):
            direction = "up" if shap_val > 0 else "down"
            feature_contributions.append({
                'name': fname,
                'value': float(fval),
                'shap': float(shap_val),
                'direction': direction,
                'abs_impact': abs(float(shap_val))
            })
        
        # Sort by absolute impact (biggest contributors first)
        feature_contributions.sort(key=lambda x: x['abs_impact'], reverse=True)
        
        return {
            'prediction': float(result['predictions'][0]),
            'base_value': float(self.base_value),
            'shap_values': result['shap_values'][0],
            'feature_names': result['feature_names'],
            'feature_values': dict(zip(result['feature_names'], result['feature_values'][0])),
            'feature_contributions': feature_contributions
        }


def get_top_shap_drivers(
    explanation: Dict[str, Any],
    n_top: int = 5,
    min_impact: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Extract top N feature contributions from SHAP explanation.
    
    Args:
        explanation: Output from SHAPExplainer.explain_single()
        n_top: Number of top drivers to return
        min_impact: Minimum absolute SHAP value to include (in ticket units)
    
    Returns:
        List of dicts with keys: name, value, shap, direction, abs_impact
    """
    drivers = []
    for contrib in explanation['feature_contributions']:
        if contrib['abs_impact'] >= min_impact:
            drivers.append(contrib)
    
    return drivers[:n_top]


def format_shap_narrative(
    explanation: Dict[str, Any],
    n_top: int = 5,
    min_impact: float = 1.0,
    include_base: bool = True
) -> str:
    """
    Convert SHAP explanation into human-readable narrative.
    
    Args:
        explanation: Output from SHAPExplainer.explain_single()
        n_top: Number of top factors to include
        min_impact: Minimum impact threshold (in ticket units)
        include_base: Whether to mention base value
    
    Returns:
        Formatted narrative string explaining the prediction
    
    Example output:
        "185 tickets (base 100 + Wiki +15 + Trends -8 + YouTube +2)"
    """
    prediction = explanation['prediction']
    base_value = explanation['base_value']
    
    drivers = get_top_shap_drivers(explanation, n_top=n_top, min_impact=min_impact)
    
    if not drivers:
        return f"{prediction:.0f} tickets (insufficient signal to decompose)"
    
    # Format each driver
    driver_parts = []
    for driver in drivers:
        # Format impact with sign
        if driver['shap'] > 0:
            impact_str = f"+{driver['shap']:.0f}"
        else:
            impact_str = f"{driver['shap']:.0f}"
        
        # Clean up feature name
        display_name = driver['name'].replace("_", " ").title()
        
        driver_parts.append(f"{display_name} {impact_str}")
    
    drivers_text = " ".join(driver_parts)
    
    if include_base:
        return f"{prediction:.0f} tickets (base {base_value:.0f} {drivers_text})"
    else:
        return f"{prediction:.0f} tickets ({drivers_text})"


def build_shap_table(
    explanation: Dict[str, Any],
    n_features: int = 5
) -> pd.DataFrame:
    """
    Create a DataFrame for displaying SHAP contributions as a table.
    
    Useful for PDF tables or web display.
    
    Returns:
        DataFrame with columns: Feature, Value, Contribution, Direction
    """
    rows = []
    for contrib in explanation['feature_contributions'][:n_features]:
        rows.append({
            'Feature': contrib['name'].title(),
            'Input Value': f"{contrib['value']:.1f}",
            'SHAP Contribution': f"{contrib['shap']:+.1f}",
            'Direction': contrib['direction'].upper()
        })
    
    return pd.DataFrame(rows)


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
        if verbose and (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(X_test)} complete...")
    
    if verbose:
        print("✓ SHAP explanation complete")
    
    return explanations
