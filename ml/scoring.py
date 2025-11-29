from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import yaml
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# MODEL LOADING AND SCORING
# =============================================================================


def load_model(model_path: str | None = None):
    """Load a trained model from disk."""
    path = Path(model_path or (Path(__file__).parent.parent / "models" / "title_demand_rf.pkl"))
    return joblib.load(path)


def score_dataframe(df_features: pd.DataFrame, model=None) -> pd.Series:
    """Score a DataFrame using the trained model."""
    model = model or load_model()
    return pd.Series(model.predict(df_features), index=df_features.index, name="forecast_single_tickets")


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
        out['_inflation_score'] = -(out['inflation_adjustment_factor'] - 1.0) * 5
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
