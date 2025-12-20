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


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_waterfall_plot(
    explanation: Dict[str, Any],
    max_features: int = 5,
    show: bool = False
) -> Optional[Any]:
    """
    Create SHAP waterfall plot for a single prediction.
    
    Shows how individual features contribute to the prediction, starting from 
    the base value and accumulating SHAP contributions.
    
    Args:
        explanation: Output from SHAPExplainer.explain_single()
        max_features: Maximum number of features to display
        show: Whether to display the plot immediately
    
    Returns:
        Matplotlib figure object or None if visualization unavailable
    
    Example:
        Waterfall shows: Base(36) → +Wiki(+6) → +YouTube(+4) → Final(48)
    """
    if not SHAP_AVAILABLE:
        warnings.warn("SHAP not available for visualization")
        return None
    
    try:
        import matplotlib.pyplot as plt
        
        # Extract top features
        drivers = get_top_shap_drivers(
            explanation, 
            n_top=max_features, 
            min_impact=0.1
        )
        
        if not drivers:
            warnings.warn("Insufficient signal for waterfall visualization")
            return None
        
        # Prepare data for waterfall
        values = [explanation['base_value']]
        labels = ['Base value']
        colors_list = []
        
        cumulative = explanation['base_value']
        for driver in drivers:
            values.append(driver['shap'])
            labels.append(driver['name'].title())
            colors_list.append('green' if driver['shap'] > 0 else 'red')
            cumulative += driver['shap']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot waterfall
        x_pos = np.arange(len(values))
        ax.bar(x_pos[0], values[0], color='gray', alpha=0.7, label='Base')
        
        running_total = values[0]
        for i in range(1, len(values)):
            if values[i] > 0:
                ax.bar(x_pos[i], values[i], bottom=running_total, color=colors_list[i-1], alpha=0.7)
            else:
                ax.bar(x_pos[i], values[i], bottom=running_total + values[i], color=colors_list[i-1], alpha=0.7)
            running_total += values[i]
        
        ax.axhline(y=explanation['prediction'], color='black', linestyle='--', linewidth=2, label=f'Final: {explanation["prediction"]:.0f}')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Ticket Count')
        ax.set_title(f'SHAP Waterfall: How features drive prediction')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig
    
    except Exception as e:
        warnings.warn(f"Could not create waterfall plot: {e}")
        return None


def create_force_plot_data(
    explanation: Dict[str, Any],
    max_features: int = 5
) -> Dict[str, Any]:
    """
    Prepare data for SHAP force plot visualization (text-based representation).
    
    Shows positive (pushing up) and negative (pushing down) contributions.
    
    Args:
        explanation: Output from SHAPExplainer.explain_single()
        max_features: Maximum features to include
    
    Returns:
        Dictionary with structured plot data suitable for web display or PDF
    
    Example output structure:
        {
            'base_value': 36.4,
            'prediction': 48.2,
            'positive_contributions': [
                {'feature': 'wiki', 'value': 5.8},
                {'feature': 'youtube', 'value': 3.6}
            ],
            'negative_contributions': []
        }
    """
    drivers = get_top_shap_drivers(explanation, n_top=max_features, min_impact=0.5)
    
    positive = [d for d in drivers if d['shap'] > 0]
    negative = [d for d in drivers if d['shap'] < 0]
    
    return {
        'base_value': float(explanation['base_value']),
        'prediction': float(explanation['prediction']),
        'positive_contributions': [
            {'feature': d['name'], 'contribution': float(d['shap'])}
            for d in sorted(positive, key=lambda x: x['shap'], reverse=True)
        ],
        'negative_contributions': [
            {'feature': d['name'], 'contribution': float(d['shap'])}
            for d in sorted(negative, key=lambda x: x['shap'])
        ],
        'total_positive': sum(d['shap'] for d in positive),
        'total_negative': sum(d['shap'] for d in negative)
    }


def create_bar_plot(
    explanations: List[Dict[str, Any]],
    show: bool = False
) -> Optional[Any]:
    """
    Create SHAP bar plot showing average feature importance across multiple predictions.
    
    Args:
        explanations: List of explanation dictionaries from SHAPExplainer
        show: Whether to display immediately
    
    Returns:
        Matplotlib figure object or None
    
    Useful for understanding which features drive predictions on average.
    """
    if not SHAP_AVAILABLE or not explanations:
        return None
    
    try:
        import matplotlib.pyplot as plt
        
        # Aggregate feature importance across explanations
        feature_impacts = {}
        for exp in explanations:
            for contrib in exp['feature_contributions']:
                fname = contrib['name']
                if fname not in feature_impacts:
                    feature_impacts[fname] = []
                feature_impacts[fname].append(abs(contrib['shap']))
        
        # Calculate mean absolute impact
        mean_impacts = {fname: np.mean(vals) for fname, vals in feature_impacts.items()}
        
        # Sort by importance
        sorted_features = sorted(mean_impacts.items(), key=lambda x: x[1], reverse=True)
        features = [f[0] for f in sorted_features[:8]]  # Top 8
        impacts = [f[1] for f in sorted_features[:8]]
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(features)))
        ax.barh(features, impacts, color=colors)
        ax.set_xlabel('Mean |SHAP value| (Average impact)')
        ax.set_title('Feature Importance: Average SHAP contributions')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig
    
    except Exception as e:
        warnings.warn(f"Could not create bar plot: {e}")
        return None


def create_html_force_plot(
    explanation: Dict[str, Any],
    title: str = "SHAP Force Plot"
) -> str:
    """
    Create HTML representation of force plot for web display or PDF embedding.
    
    Args:
        explanation: Output from SHAPExplainer.explain_single()
        title: Title for the visualization
    
    Returns:
        HTML string with force plot visualization
    """
    force_data = create_force_plot_data(explanation, max_features=5)
    
    # Build HTML
    html_parts = [
        f'<div style="font-family: monospace; background: #f5f5f5; padding: 15px; border-radius: 8px;">',
        f'<h3>{title}</h3>',
        f'<p style="font-size: 14px; margin: 10px 0;"><b>Prediction:</b> {force_data["prediction"]:.1f} tickets</p>',
        f'<div style="margin: 20px 0;">',
        f'<p style="font-weight: bold; color: #333;">Base Value: {force_data["base_value"]:.1f}</p>'
    ]
    
    # Positive contributions
    if force_data['positive_contributions']:
        html_parts.append('<div style="margin-left: 20px; color: green;">')
        html_parts.append('<p style="font-weight: bold;">Pushing Up:</p>')
        for item in force_data['positive_contributions']:
            html_parts.append(f'<p style="margin: 5px 0;">→ {item["feature"].title()}: +{item["contribution"]:.1f}</p>')
        html_parts.append('</div>')
    
    # Negative contributions
    if force_data['negative_contributions']:
        html_parts.append('<div style="margin-left: 20px; color: red;">')
        html_parts.append('<p style="font-weight: bold;">Pushing Down:</p>')
        for item in force_data['negative_contributions']:
            html_parts.append(f'<p style="margin: 5px 0;">← {item["feature"].title()}: {item["contribution"]:.1f}</p>')
        html_parts.append('</div>')
    
    html_parts.extend([
        '</div>',
        f'<p style="border-top: 1px solid #999; padding-top: 10px; margin-top: 15px; font-weight: bold;">',
        f'Final: {force_data["prediction"]:.1f} tickets</p>',
        '</div>'
    ])
    
    return '\n'.join(html_parts)


def save_shap_plots(
    explanation: Dict[str, Any],
    output_dir: str = "."
) -> Dict[str, str]:
    """
    Save SHAP plots to disk (for PDF embedding or web display).
    
    Args:
        explanation: Output from SHAPExplainer.explain_single()
        output_dir: Directory to save plots to
    
    Returns:
        Dictionary mapping plot type to file path
        
    Example:
        {
            'waterfall': '/path/to/waterfall.png',
            'force': '/path/to/force.html'
        }
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = {}
    
    # Save waterfall plot
    try:
        import matplotlib.pyplot as plt
        fig = create_waterfall_plot(explanation, show=False)
        if fig:
            path = os.path.join(output_dir, 'shap_waterfall.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            saved_files['waterfall'] = path
            plt.close(fig)
    except Exception as e:
        warnings.warn(f"Could not save waterfall plot: {e}")
    
    # Save force plot as HTML
    try:
        html = create_html_force_plot(explanation)
        path = os.path.join(output_dir, 'shap_force.html')
        with open(path, 'w') as f:
            f.write(html)
        saved_files['force'] = path
    except Exception as e:
        warnings.warn(f"Could not save force plot: {e}")
    
    return saved_files

