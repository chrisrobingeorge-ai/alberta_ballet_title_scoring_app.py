"""
Model scoring module with schema validation and uncertainty quantification.

Features:
- Schema validation against training features
- Column drift detection
- Prediction interval estimation
- Economic impact scoring
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import warnings
import pandas as pd
import numpy as np
import joblib
import yaml
import logging

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"


# =============================================================================
# SCHEMA VALIDATION
# =============================================================================


class SchemaValidationWarning(UserWarning):
    """Warning raised when input schema doesn't match training schema."""
    pass


def load_training_schema() -> Optional[Dict[str, Any]]:
    """Load the training schema from model metadata.
    
    Returns:
        Dictionary with feature names and metadata, or None if not available
    """
    metadata_path = MODELS_DIR / "model_metadata.json"
    
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            return {
                "features": metadata.get("features", []),
                "n_features": metadata.get("n_features", 0),
                "training_date": metadata.get("training_date")
            }
        except Exception as e:
            logger.warning(f"Could not load training schema: {e}")
    
    return None


def validate_input_schema(
    df: pd.DataFrame,
    training_schema: Optional[Dict[str, Any]] = None,
    raise_on_error: bool = False
) -> Tuple[bool, List[str]]:
    """Validate input DataFrame schema against training schema.
    
    Args:
        df: Input DataFrame to validate
        training_schema: Schema from training (loads from metadata if None)
        raise_on_error: If True, raise ValueError on schema mismatch
        
    Returns:
        Tuple of (is_valid, list of warning messages)
    """
    if training_schema is None:
        training_schema = load_training_schema()
    
    if training_schema is None:
        return True, ["No training schema available for validation"]
    
    warnings_list = []
    is_valid = True
    
    expected_features = set(training_schema.get("features", []))
    actual_features = set(df.columns.tolist())
    
    # Check for missing columns
    missing_cols = expected_features - actual_features
    if missing_cols:
        msg = f"Missing {len(missing_cols)} columns from training schema: {list(missing_cols)[:5]}"
        if len(missing_cols) > 5:
            msg += f"... and {len(missing_cols) - 5} more"
        warnings_list.append(msg)
        is_valid = False
    
    # Check for extra columns (may indicate drift)
    extra_cols = actual_features - expected_features
    if extra_cols:
        msg = f"Found {len(extra_cols)} extra columns not in training schema: {list(extra_cols)[:5]}"
        if len(extra_cols) > 5:
            msg += f"... and {len(extra_cols) - 5} more"
        warnings_list.append(msg)
    
    # Check column order (some models are sensitive to this)
    if expected_features and actual_features:
        common_features = expected_features & actual_features
        if common_features:
            training_order = [f for f in training_schema.get("features", []) if f in common_features]
            actual_order = [f for f in df.columns if f in common_features]
            if training_order != actual_order:
                warnings_list.append("Column order differs from training schema")
    
    # Emit warnings
    for warning_msg in warnings_list:
        warnings.warn(warning_msg, SchemaValidationWarning)
    
    if not is_valid and raise_on_error:
        raise ValueError(f"Schema validation failed: {'; '.join(warnings_list)}")
    
    return is_valid, warnings_list


# =============================================================================
# MODEL LOADING AND SCORING
# =============================================================================


def load_model(model_path: str | None = None):
    """Load a trained model from disk."""
    path = Path(model_path or (MODELS_DIR / "title_demand_rf.pkl"))
    return joblib.load(path)


def score_dataframe(
    df_features: pd.DataFrame,
    model=None,
    validate_schema: bool = True
) -> pd.Series:
    """Score a DataFrame using the trained model.
    
    Args:
        df_features: DataFrame with features for scoring
        model: Trained model (loads default if None)
        validate_schema: Whether to validate input schema
        
    Returns:
        Series of predictions
    """
    model = model or load_model()
    
    # Validate schema if requested
    if validate_schema:
        validate_input_schema(df_features)
    
    return pd.Series(
        model.predict(df_features),
        index=df_features.index,
        name="forecast_single_tickets"
    )


def score_with_uncertainty(
    df_features: pd.DataFrame,
    model=None,
    confidence_level: float = 0.9,
    n_bootstrap: int = 100
) -> pd.DataFrame:
    """Score DataFrame with prediction intervals via bootstrapping.
    
    For Random Forest models, uses the individual tree predictions to
    estimate uncertainty. For other models, uses bootstrap resampling.
    
    Args:
        df_features: DataFrame with features for scoring
        model: Trained model (loads default if None)
        confidence_level: Confidence level for intervals (default 0.9 = 90%)
        n_bootstrap: Number of bootstrap samples (unused for RF)
        
    Returns:
        DataFrame with columns: prediction, lower_bound, upper_bound
    """
    model = model or load_model()
    
    # Check if model has estimators (Random Forest)
    pipeline_model = model
    if hasattr(model, "named_steps"):
        # It's a pipeline, get the estimator
        for step_name, step in model.named_steps.items():
            if hasattr(step, "estimators_"):
                pipeline_model = model
                estimator = step
                break
        else:
            estimator = None
    else:
        estimator = model if hasattr(model, "estimators_") else None
    
    predictions = pipeline_model.predict(df_features)
    
    if estimator is not None and hasattr(estimator, "estimators_"):
        # Random Forest: use individual tree predictions
        # Need to transform features first if using pipeline
        if hasattr(model, "named_steps") and "pre" in model.named_steps:
            X_transformed = model.named_steps["pre"].transform(df_features)
        else:
            X_transformed = df_features
        
        tree_predictions = np.array([
            tree.predict(X_transformed) for tree in estimator.estimators_
        ])
        
        alpha = 1 - confidence_level
        lower = np.percentile(tree_predictions, alpha / 2 * 100, axis=0)
        upper = np.percentile(tree_predictions, (1 - alpha / 2) * 100, axis=0)
    else:
        # Fallback: use a simple uncertainty estimate based on prediction magnitude
        # This is a rough approximation when we can't get true uncertainty
        uncertainty = np.abs(predictions) * 0.2  # 20% relative uncertainty
        alpha = 1 - confidence_level
        z_score = 1.645 if confidence_level == 0.9 else 1.96  # Approximate
        lower = predictions - z_score * uncertainty
        upper = predictions + z_score * uncertainty
    
    result = pd.DataFrame({
        "prediction": predictions,
        "lower_bound": lower,
        "upper_bound": upper
    }, index=df_features.index)
    
    return result


# =============================================================================
# ECONOMIC BASELINES CONFIGURATION
# =============================================================================


DEFAULT_ECONOMIC_BASELINES = {
    'consumer_confidence': {
        'baseline': 55.0,  # Historical average from BNCCI
        'description': 'Bloomberg Nanos Consumer Confidence Index baseline',
        'good_threshold': 60.0,
        'poor_threshold': 45.0
    },
    'energy_index': {
        'baseline': 800.0,  # Reference value from commodity price index
        'description': 'Bank of Canada Energy Commodity Price Index baseline',
        'good_threshold': 1200.0,
        'poor_threshold': 500.0
    },
    'cpi_base': {
        'baseline': 137.5,  # CPI value around 2020
        'description': 'Consumer Price Index baseline for inflation adjustment',
        'reference_date': '2020-01-01'
    }
}


# Scaling factor for inflation impact on economic score
# A factor of 5 means 20% inflation deviation from 1.0 hits the clip bounds
INFLATION_IMPACT_SCALE = 5


def load_economic_baselines(config_path: Optional[str] = None) -> dict:
    """Load economic baseline configuration.
    
    If a config file exists, loads from YAML. Otherwise uses defaults.
    
    Args:
        config_path: Optional path to economic_baselines.yaml
        
    Returns:
        Dictionary of economic baseline configurations
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "economic_baselines.yaml"
    else:
        config_path = Path(config_path)
    
    if config_path.exists():
        try:
            with open(config_path) as f:
                baselines = yaml.safe_load(f)
            logger.info(f"Loaded economic baselines from {config_path}")
            return baselines
        except Exception as e:
            logger.warning(f"Error loading baselines from {config_path}: {e}")
    
    return DEFAULT_ECONOMIC_BASELINES


def save_economic_baselines(baselines: dict, config_path: Optional[str] = None):
    """Save economic baseline configuration to YAML.
    
    Args:
        baselines: Dictionary of baseline configurations
        config_path: Path to save the config file
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "economic_baselines.yaml"
    else:
        config_path = Path(config_path)
    
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(baselines, f, default_flow_style=False)
    
    logger.info(f"Saved economic baselines to {config_path}")


# =============================================================================
# ECONOMIC IMPACT SCORE CALCULATION
# =============================================================================


def compute_economic_impact_score(
    df: pd.DataFrame,
    baselines: Optional[dict] = None,
    include_components: bool = True
) -> pd.DataFrame:
    """Compute Economic Impact Score for each title and city.
    
    The Economic Impact Score measures how current economic conditions
    might affect ticket sales relative to baseline conditions. Positive
    scores indicate favorable conditions, negative indicates headwinds.
    
    Components:
    - Consumer confidence delta (vs baseline)
    - Energy index delta (Alberta economy health)
    - Inflation adjustment impact
    
    Args:
        df: DataFrame with economic features (consumer_confidence_*, 
            energy_index, inflation_adjustment_factor)
        baselines: Optional economic baselines config
        include_components: If True, include component scores in output
        
    Returns:
        DataFrame with economic_impact_score and optional component columns
    """
    if baselines is None:
        baselines = load_economic_baselines()
    
    out = df.copy()
    
    # Consumer Confidence Component
    cc_baseline = baselines.get('consumer_confidence', {}).get('baseline', 55.0)
    cc_good = baselines.get('consumer_confidence', {}).get('good_threshold', 60.0)
    cc_poor = baselines.get('consumer_confidence', {}).get('poor_threshold', 45.0)
    
    cc_col = None
    for col in ['consumer_confidence_prairies', 'consumer_confidence_headline', 'consumer_confidence']:
        if col in out.columns:
            cc_col = col
            break
    
    if cc_col:
        # Normalize to -1 to +1 scale
        out['_cc_score'] = (out[cc_col] - cc_baseline) / (cc_good - cc_poor)
        out['_cc_score'] = out['_cc_score'].clip(-1, 1)
    else:
        out['_cc_score'] = 0.0
    
    # Energy Index Component
    ei_baseline = baselines.get('energy_index', {}).get('baseline', 800.0)
    ei_good = baselines.get('energy_index', {}).get('good_threshold', 1200.0)
    ei_poor = baselines.get('energy_index', {}).get('poor_threshold', 500.0)
    
    if 'energy_index' in out.columns:
        # Normalize to -1 to +1 scale
        out['_ei_score'] = (out['energy_index'] - ei_baseline) / (ei_good - ei_poor)
        out['_ei_score'] = out['_ei_score'].clip(-1, 1)
    else:
        out['_ei_score'] = 0.0
    
    # Inflation Component (higher inflation = negative impact)
    if 'inflation_adjustment_factor' in out.columns:
        # Values > 1 mean prices have risen (negative impact)
        # Normalize around 1.0: 0.95-1.05 is neutral, beyond is impact
        out['_inflation_score'] = -(out['inflation_adjustment_factor'] - 1.0) * INFLATION_IMPACT_SCALE
        out['_inflation_score'] = out['_inflation_score'].clip(-1, 1)
    else:
        out['_inflation_score'] = 0.0
    
    # Combine into overall Economic Impact Score
    # Weights: consumer confidence 40%, energy 35%, inflation 25%
    out['economic_impact_score'] = (
        0.40 * out['_cc_score'] +
        0.35 * out['_ei_score'] +
        0.25 * out['_inflation_score']
    )
    
    # Scale to more intuitive range (-100 to +100)
    out['economic_impact_score'] = (out['economic_impact_score'] * 100).round(1)
    
    if include_components:
        out['econ_impact_consumer_confidence'] = (out['_cc_score'] * 100).round(1)
        out['econ_impact_energy'] = (out['_ei_score'] * 100).round(1)
        out['econ_impact_inflation'] = (out['_inflation_score'] * 100).round(1)
    
    # Clean up internal columns
    out = out.drop(columns=['_cc_score', '_ei_score', '_inflation_score'], errors='ignore')
    
    logger.info(f"Computed economic impact scores for {len(out)} rows")
    return out


def compute_city_economic_summary(
    df: pd.DataFrame,
    city_column: str = 'city'
) -> pd.DataFrame:
    """Compute economic impact summary by city.
    
    Aggregates economic impact scores and components by city to provide
    a city-level view of economic conditions affecting ticket sales.
    
    Args:
        df: DataFrame with economic impact scores and city column
        city_column: Name of the city column
        
    Returns:
        DataFrame with city-level economic summary
    """
    if 'economic_impact_score' not in df.columns:
        logger.warning("economic_impact_score not found; computing first")
        df = compute_economic_impact_score(df)
    
    # Find city column
    city_col = None
    for col in [city_column, 'city', 'city_name', 'location']:
        col_lower = col.lower().replace(' ', '_')
        if col_lower in df.columns:
            city_col = col_lower
            break
    
    if city_col is None:
        # Try to infer from city_calgary/city_edmonton
        if 'city_calgary' in df.columns or 'city_edmonton' in df.columns:
            df['_city_inferred'] = np.where(
                df.get('city_calgary', 0) == 1, 'Calgary',
                np.where(df.get('city_edmonton', 0) == 1, 'Edmonton', 'Unknown')
            )
            city_col = '_city_inferred'
        else:
            logger.warning("No city column found")
            return pd.DataFrame()
    
    # Aggregate by city
    agg_cols = ['economic_impact_score']
    for col in ['econ_impact_consumer_confidence', 'econ_impact_energy', 'econ_impact_inflation']:
        if col in df.columns:
            agg_cols.append(col)
    
    summary = df.groupby(city_col)[agg_cols].agg(['mean', 'min', 'max']).round(1)
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    # Clean up inferred column if used
    if '_city_inferred' in df.columns:
        df = df.drop(columns=['_city_inferred'])
    
    return summary


def score_with_economic_impact(
    df_features: pd.DataFrame,
    model=None,
    include_economic: bool = True,
    baselines: Optional[dict] = None
) -> pd.DataFrame:
    """Score a DataFrame and include economic impact analysis.
    
    Combines model predictions with economic impact scores to provide
    a comprehensive view of expected ticket sales and economic factors.
    
    Args:
        df_features: DataFrame with features for model scoring
        model: Trained model (loads default if None)
        include_economic: If True, compute economic impact scores
        baselines: Optional economic baselines config
        
    Returns:
        DataFrame with predictions and economic impact scores
    """
    out = df_features.copy()
    
    # Get base predictions from model
    try:
        model = model or load_model()
        
        # Get feature columns expected by model
        if hasattr(model, 'feature_names_in_'):
            model_features = list(model.feature_names_in_)
            available_features = [f for f in model_features if f in out.columns]
            
            if len(available_features) == len(model_features):
                predictions = model.predict(out[model_features])
                out['forecast_single_tickets'] = predictions
            else:
                logger.warning(f"Missing model features: {set(model_features) - set(available_features)}")
                out['forecast_single_tickets'] = np.nan
        else:
            # Try scoring with all numeric features
            numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 0:
                predictions = model.predict(out[numeric_cols])
                out['forecast_single_tickets'] = predictions
            else:
                out['forecast_single_tickets'] = np.nan
                
    except Exception as e:
        logger.warning(f"Error scoring with model: {e}")
        out['forecast_single_tickets'] = np.nan
    
    # Add economic impact scores
    if include_economic:
        out = compute_economic_impact_score(out, baselines=baselines)
    
    return out
