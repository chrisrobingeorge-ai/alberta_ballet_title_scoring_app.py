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
import logging

import pandas as pd
import numpy as np
import joblib
import yaml

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
DATA_DIR = Path(__file__).parent.parent / "data"
ECONOMIC_BASELINES_PATH = DATA_DIR / "economic" / "economic_baselines.csv"


# =============================================================================
# SCHEMA VALIDATION
# =============================================================================


class SchemaValidationWarning(UserWarning):
    """Warning raised when input schema doesn't match training schema."""
    pass


def load_training_schema() -> Optional[Dict[str, Any]]:
    """
    Load the training schema from model metadata.

    Returns
    -------
    dict or None
        Dictionary with feature names and metadata, or None if not available.
    """
    metadata_path = MODELS_DIR / "model_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
            return {
                "features": metadata.get("features", []),
                "n_features": metadata.get("n_features", 0),
                "training_date": metadata.get("training_date"),
            }
        except Exception as exc:
            logger.warning("Could not load training schema: " + str(exc))
    return None


def validate_input_schema(
    df: pd.DataFrame,
    training_schema: Optional[Dict[str, Any]] = None,
    raise_on_error: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Validate input DataFrame schema against training schema.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to validate.
    training_schema : dict, optional
        Training schema dictionary.
    raise_on_error : bool, default False
        Whether to raise on schema mismatch.

    Returns
    -------
    (ok, messages) : (bool, list of str)
        ok is True if schema is acceptable, messages contains warnings.
    """
    messages: List[str] = []

    if training_schema is None:
        training_schema = load_training_schema()

    if training_schema is None:
        messages.append("No training schema found; skipping strict validation.")
        return True, messages

    expected_features = list(training_schema.get("features", []))
    if not expected_features:
        messages.append("Training schema has empty feature list; skipping validation.")
        return True, messages

    input_cols = list(df.columns)

    missing = [c for c in expected_features if c not in input_cols]
    extra = [c for c in input_cols if c not in expected_features]

    if missing:
        msg = "Missing expected features in input: " + ", ".join(missing)
        messages.append(msg)
    if extra:
        msg = "Extra columns in input that were not used in training: " + ", ".join(extra)
        messages.append(msg)

    ok = len(missing) == 0
    if not ok:
        text = (
            "Input DataFrame is missing one or more features that were present "
            "during training. This can degrade predictions or cause failure."
        )
        messages.append(text)

    if (not ok) and raise_on_error:
        raise SchemaValidationWarning("; ".join(messages))

    return ok, messages


def detect_column_drift(
    df_input: pd.DataFrame,
    df_training_sample: Optional[pd.DataFrame] = None,
    threshold: float = 0.2,
) -> Dict[str, Dict[str, float]]:
    """
    Simple column-level drift detection based on differences in means/stds.

    Parameters
    ----------
    df_input : pd.DataFrame
        Current scoring data.
    df_training_sample : pd.DataFrame, optional
        Sample of training data with same columns.
    threshold : float, default 0.2
        Relative change threshold to flag drift.

    Returns
    -------
    drift_report : dict
        Per-column drift metrics: mean_change, std_change, drift_flag.
    """
    if df_training_sample is None:
        return {}

    common = [c for c in df_input.columns if c in df_training_sample.columns]
    report: Dict[str, Dict[str, float]] = {}

    for col in common:
        s_now = pd.to_numeric(df_input[col], errors="coerce")
        s_train = pd.to_numeric(df_training_sample[col], errors="coerce")
        if s_now.notna().sum() == 0 or s_train.notna().sum() == 0:
            continue

        mean_now = float(s_now.mean())
        mean_train = float(s_train.mean())
        std_now = float(s_now.std(ddof=1))
        std_train = float(s_train.std(ddof=1))

        mean_change = abs(mean_now - mean_train) / (abs(mean_train) + 1e-8)
        std_change = abs(std_now - std_train) / (abs(std_train) + 1e-8)
        drift_flag = 1.0 if (mean_change > threshold or std_change > threshold) else 0.0

        report[col] = {
            "mean_change": mean_change,
            "std_change": std_change,
            "drift_flag": drift_flag,
        }

    return report


# =============================================================================
# MODEL LOADING
# =============================================================================


def _safe_load_yaml(path: Path) -> Dict[str, Any]:
    if (not path.exists()) or (not path.is_file()):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
        return content or {}
    except Exception as exc:
        logger.warning("Could not read YAML at " + str(path) + ": " + str(exc))
        return {}


def load_model(model_name: str = "model_xgb_remount_postcovid") -> Any:
    """
    Load a trained model by name from the models directory.

    Parameters
    ----------
    model_name : str
        Base model name without extension.

    Returns
    -------
    model : Any
        Deserialized model object.

    Raises
    ------
    FileNotFoundError
        If no trained model file (.joblib, .pkl, or valid XGBoost .json) is found.
        Note: A .json file containing only metadata (not a trained model) will not
        load successfully as an XGBoost model.
    """
    joblib_path = MODELS_DIR / (model_name + ".joblib")
    pkl_path = MODELS_DIR / (model_name + ".pkl")
    json_path = MODELS_DIR / (model_name + ".json")

    # Try .joblib first (most common for sklearn pipelines)
    if joblib_path.exists():
        try:
            return joblib.load(joblib_path)
        except Exception as exc:
            logger.warning(f"Could not load .joblib model: {exc}")

    # Try .pkl (older format)
    if pkl_path.exists():
        try:
            return joblib.load(pkl_path)
        except Exception as exc:
            logger.warning(f"Could not load .pkl model: {exc}")

    # Try .json as XGBoost Booster format
    if json_path.exists():
        try:
            import xgboost as xgb

            booster = xgb.Booster()
            booster.load_model(str(json_path))
            return booster
        except Exception as exc:
            # The .json file might be metadata, not an actual XGBoost model
            logger.warning(f"Could not load JSON as XGBoost model (may be metadata file): {exc}")

    raise FileNotFoundError(
        f"No trained model file found for '{model_name}'. "
        f"Checked: {joblib_path}, {pkl_path}, {json_path}. "
        f"Run `python scripts/train_safe_model.py` to train a model first."
    )


def load_feature_recipe() -> Optional[pd.DataFrame]:
    """
    Load the feature recipe used for training (if present).

    Returns
    -------
    df or None
    """
    csv_path = MODELS_DIR / "model_recipe_linear.csv"
    if not csv_path.exists():
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception as exc:
        logger.warning("Could not load feature recipe CSV: " + str(exc))
        return None


def get_feature_order() -> Optional[List[str]]:
    """
    Return the ordered list of feature names expected by the model.
    """
    df_recipe = load_feature_recipe()
    if df_recipe is None:
        return None
    cols = df_recipe.get("feature_name")
    if cols is None:
        return None
    return [str(c) for c in cols.tolist()]


def get_feature_order_from_model(model: Any) -> Optional[List[str]]:
    """
    Extract the expected feature names directly from a trained sklearn Pipeline.
    
    Parameters
    ----------
    model : Any
        A trained sklearn Pipeline with a ColumnTransformer preprocessor.
    
    Returns
    -------
    list of str or None
        Ordered list of feature names, or None if extraction fails.
    """
    try:
        # Check if it's a sklearn Pipeline with a preprocessor step
        if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
            preprocessor = model.named_steps['preprocessor']
            
            # Extract feature names from the ColumnTransformer
            if hasattr(preprocessor, 'transformers_'):
                feature_names = []
                for name, transformer, columns in preprocessor.transformers_:
                    if name != 'remainder':  # Skip remainder if present
                        feature_names.extend(columns)
                return feature_names
    except Exception as exc:
        model_type = type(model).__name__
        logger.warning(f"Could not extract feature names from {model_type} model: {exc}")
    
    return None


# =============================================================================
# CORE SCORING
# =============================================================================


def _prepare_features_for_model(df: pd.DataFrame, model: Optional[Any] = None) -> pd.DataFrame:
    """
    Prepare feature DataFrame in the correct column order and dtype for the model.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input feature DataFrame
    model : Any, optional
        The trained model (used to extract feature names if recipe not available)
    
    Returns
    -------
    pd.DataFrame
        Prepared feature DataFrame with all expected features
    """
    df_feat = df.copy()

    # Drop known label/ID columns if present
    to_drop = [
        "single_tickets_calgary",
        "single_tickets_edmonton",
        "total_single_tickets",
        "show_id",
        "run_id",
        "season",
        "label",
        "target",
    ]
    cols_to_drop = [c for c in to_drop if c in df_feat.columns]
    if cols_to_drop:
        df_feat = df_feat.drop(columns=cols_to_drop)

    # Try to get feature order from recipe file first
    feature_order = get_feature_order()
    
    # If recipe not available, try to extract from model
    if feature_order is None and model is not None:
        feature_order = get_feature_order_from_model(model)
    
    if feature_order is not None:
        # Add any missing expected features with default values
        for col in feature_order:
            if col not in df_feat.columns:
                # Use appropriate default values based on feature type
                if col.startswith('is_') or col.endswith('_flag'):
                    df_feat[col] = 0  # Binary features default to 0
                elif col in ['opening_date', 'opening_season', 'category', 'gender']:
                    df_feat[col] = 'missing'  # Categorical features
                else:
                    df_feat[col] = 0.0  # Numeric features default to 0
        df_feat = df_feat[feature_order]

    # Coerce numeric columns; non-numeric become NaN then filled with 0
    # But preserve categorical columns
    numeric_cols = [c for c in df_feat.columns if c not in ['opening_date', 'opening_season', 'category', 'gender']]
    for col in numeric_cols:
        df_feat[col] = pd.to_numeric(df_feat[col], errors='coerce').fillna(0.0)

    return df_feat


def predict_point(
    df_features: pd.DataFrame,
    model: Optional[Any] = None,
    model_name: str = "model_xgb_remount_postcovid",
) -> np.ndarray:
    """
    Predict point estimates (expected single tickets) for each row.

    For XGBoost booster: uses .predict on DMatrix
    For sklearn models: uses .predict on numpy array.
    """
    if model is None:
        model = load_model(model_name)

    df_prepared = _prepare_features_for_model(df_features, model=model)
    
    # For sklearn Pipeline, use the pipeline's predict method directly (not raw numpy)
    # This ensures the preprocessing step receives the DataFrame correctly
    if hasattr(model, 'predict') and hasattr(model, 'named_steps'):
        # sklearn Pipeline - pass the DataFrame
        preds = model.predict(df_prepared)
        return np.asarray(preds, dtype=float)
    
    # For other models, convert to numpy
    x_mat = df_prepared.values

    # XGBoost Booster API
    try:
        import xgboost as xgb

        if isinstance(model, xgb.Booster):
            dmatrix = xgb.DMatrix(x_mat)
            preds = model.predict(dmatrix)
            return np.asarray(preds, dtype=float)
    except Exception:
        pass

    # Fallback: sklearn-style API
    if hasattr(model, "predict"):
        preds = model.predict(x_mat)
        return np.asarray(preds, dtype=float)

    raise TypeError("Model of type " + str(type(model)) + " is not supported.")


def bootstrap_prediction_intervals(
    df_features: pd.DataFrame,
    model: Optional[Any] = None,
    model_name: str = "model_xgb_remount_postcovid",
    n_bootstrap: int = 200,
    confidence_level: float = 0.8,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate prediction intervals via simple residual bootstrapping.

    Returns
    -------
    mean_pred : array
        Point predictions (same as predict_point).
    lower : array
        Lower bound of prediction interval.
    upper : array
        Upper bound of prediction interval.
    """
    rng = np.random.RandomState(random_state)

    base_pred = predict_point(df_features, model=model, model_name=model_name)
    n = base_pred.shape[0]
    if n == 0:
        return base_pred, base_pred, base_pred

    # Very simple residual noise model: use a global residual std from training
    meta_path = MODELS_DIR / "model_metadata.json"
    residual_std = 0.15  # fallback relative noise
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            residual_std = float(meta.get("residual_std", residual_std))
        except Exception:
            pass

    samples = []
    for _ in range(n_bootstrap):
        noise = rng.normal(loc=0.0, scale=residual_std, size=n)
        samples.append(base_pred * (1.0 + noise))

    samples_arr = np.stack(samples, axis=0)
    alpha = 1.0 - confidence_level
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)

    lower = np.percentile(samples_arr, lower_q, axis=0)
    upper = np.percentile(samples_arr, upper_q, axis=0)

    return base_pred, lower, upper


# =============================================================================
# ECONOMIC BASELINES
# =============================================================================


def load_economic_baselines(path: Union[str, Path, None] = None) -> pd.DataFrame:
    """
    Load precomputed economic baseline factors used for title scoring.

    Parameters
    ----------
    path : str or Path, optional
        Custom path to the baselines CSV. If None, uses ECONOMIC_BASELINES_PATH.

    Returns
    -------
    pd.DataFrame
        DataFrame with economic baselines indexed by date or period.
    """
    if path is None:
        path = ECONOMIC_BASELINES_PATH

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Economic baselines file not found at {path}")

    df = pd.read_csv(path)
    return df

# =============================================================================
# ECONOMIC CONTEXT
# =============================================================================


def load_economic_config() -> Dict[str, Any]:
    """
    Load economic / macro configuration (e.g., scaling factors).
    """
    cfg_path = OUTPUTS_DIR / "economic_context.yaml"
    return _safe_load_yaml(cfg_path)


def attach_economic_context(
    df: pd.DataFrame,
    context: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Optionally attach economic or macro context columns to predictions.
    """
    if context is None:
        context = load_economic_config()

    if not context:
        return df

    df_out = df.copy()
    for key, val in context.items():
        col = "macro_" + str(key)
        if col not in df_out.columns:
            df_out[col] = val

    return df_out


# =============================================================================
# PUBLIC API
# =============================================================================


def score_with_uncertainty(
    df_features: pd.DataFrame,
    model: Optional[Any] = None,
    model_name: str = "model_xgb_remount_postcovid",
    confidence_level: float = 0.8,
    n_bootstrap: int = 200,
) -> pd.DataFrame:
    """
    Core scoring entry point with prediction intervals.

    Parameters
    ----------
    df_features : pd.DataFrame
        Features for each proposed run or scenario.
        Does NOT need label columns. Any label / ID columns present will be ignored.
    model : Any, optional
        Preloaded model; if None, loaded from disk.
    model_name : str
        Base model filename (without extension).
    confidence_level : float
        Confidence for interval, e.g. 0.8 for 80%.
    n_bootstrap : int
        Number of bootstrap samples for interval.

    Returns
    -------
    df_pred : pd.DataFrame
        Columns:
        - forecast_single_tickets
        - prediction (alias for forecast_single_tickets)
        - lower_tickets (also aliased as lower_bound)
        - upper_tickets (also aliased as upper_bound)
    """
    training_schema = load_training_schema()
    ok, msgs = validate_input_schema(df_features, training_schema, raise_on_error=False)
    for msg in msgs:
        warnings.warn(msg, SchemaValidationWarning)

    y_hat, lower, upper = bootstrap_prediction_intervals(
        df_features,
        model=model,
        model_name=model_name,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
    )

    out = pd.DataFrame(
        {
            "forecast_single_tickets": y_hat.astype(float),
            "prediction": y_hat.astype(float),  # Alias for compatibility
            "lower_tickets": lower.astype(float),
            "lower_bound": lower.astype(float),  # Alias for compatibility
            "upper_tickets": upper.astype(float),
            "upper_bound": upper.astype(float),  # Alias for compatibility
        },
        index=df_features.index,
    )
    return out


def score_runs_for_planning(
    df_runs: pd.DataFrame,
    confidence_level: float = 0.8,
    n_bootstrap: int = 200,
    model: Optional[Any] = None,
    model_name: str = "model_xgb_remount_postcovid",
    attach_context: bool = False,
    economic_context: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    High-level helper for season / run planning.

    Parameters
    ----------
    df_runs : pd.DataFrame
        DataFrame of proposed runs with the same feature columns
        as used in training (order does not matter).
    confidence_level : float
        e.g. 0.8 for an 80 percent prediction interval.
    n_bootstrap : int
        Number of bootstrap draws for the interval.
    model : Any, optional
        Pretrained model (otherwise loaded from disk).
    model_name : str
        Base model filename (without extension).
    attach_context : bool
        Whether to annotate with economic context.
    economic_context : dict, optional
        Context values.

    Returns
    -------
    result : pd.DataFrame
        Original columns plus:
        - forecast_single_tickets
        - lower_tickets_<XX>
        - upper_tickets_<XX>
        - (optional) macro_* context columns
    """
    df_features = df_runs.copy()

    # Score with uncertainty
    df_pred = score_with_uncertainty(
        df_features,
        model=model,
        model_name=model_name,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
    )

    # Merge back onto original
    result = df_runs.copy()
    result = result.join(df_pred)

    # Rename interval columns to include confidence percentage
    pct = int(confidence_level * 100.0)
    lower_col = "lower_tickets_" + str(pct)
    upper_col = "upper_tickets_" + str(pct)
    result[lower_col] = result["lower_tickets"]
    result[upper_col] = result["upper_tickets"]
    result = result.drop(columns=["lower_tickets", "upper_tickets"])

    # Optional economic context
    if attach_context:
        if economic_context is None:
            economic_context = load_economic_config()
        result = attach_economic_context(result, economic_context)

    return result


# =============================================================================
# ECONOMIC IMPACT SCORING
# =============================================================================


# Default baselines for economic indicators
DEFAULT_ECONOMIC_BASELINES: Dict[str, Any] = {
    "consumer_confidence": {
        "baseline": 55.0,
        "good_threshold": 60.0,
        "poor_threshold": 45.0,
        "weight": 0.4,
    },
    "energy_index": {
        "baseline": 800.0,
        "good_threshold": 1000.0,
        "poor_threshold": 500.0,
        "weight": 0.3,
    },
    "cpi_base": {
        "baseline": 1.0,
        "good_threshold": 0.98,  # Low inflation is good
        "poor_threshold": 1.10,  # High inflation is poor
        "weight": 0.3,
    },
}


def load_economic_baselines(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load economic baselines configuration.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to economic baselines YAML config. If None, uses defaults.

    Returns
    -------
    dict
        Baselines dictionary with consumer_confidence, energy_index, cpi_base keys.
    """
    if config_path is not None:
        cfg_path = Path(config_path)
        baselines = _safe_load_yaml(cfg_path)
        if baselines:
            return baselines
    
    # Try default config location
    default_path = OUTPUTS_DIR.parent / "config" / "economic_baselines.yaml"
    if default_path.exists():
        baselines = _safe_load_yaml(default_path)
        if baselines:
            return baselines
    
    return DEFAULT_ECONOMIC_BASELINES.copy()


def compute_economic_impact_score(
    df: pd.DataFrame,
    baselines: Optional[Dict[str, Any]] = None,
    include_components: bool = False,
) -> pd.DataFrame:
    """
    Compute economic impact score based on economic indicators.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with economic indicator columns.
    baselines : dict, optional
        Economic baselines config. If None, uses defaults.
    include_components : bool, default False
        Whether to include individual component scores.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with economic_impact_score column added.
        If include_components=True, also adds econ_impact_* columns.
    """
    if baselines is None:
        baselines = load_economic_baselines()
    
    result = df.copy()
    
    # Initialize components
    consumer_score = 0.0
    energy_score = 0.0
    inflation_score = 0.0
    
    cc_config = baselines.get("consumer_confidence", {})
    energy_config = baselines.get("energy_index", {})
    cpi_config = baselines.get("cpi_base", {})
    
    # Consumer confidence component
    cc_col = None
    for col in ["consumer_confidence_headline", "consumer_confidence_prairies"]:
        if col in result.columns:
            cc_col = col
            break
    
    if cc_col is not None:
        cc_baseline = cc_config.get("baseline", 55.0)
        cc_weight = cc_config.get("weight", 0.4)
        result["_cc_diff"] = result[cc_col] - cc_baseline
        result["econ_impact_consumer_confidence"] = result["_cc_diff"] * cc_weight
    else:
        result["econ_impact_consumer_confidence"] = 0.0
    
    # Energy index component
    if "energy_index" in result.columns:
        energy_baseline = energy_config.get("baseline", 800.0)
        energy_weight = energy_config.get("weight", 0.3)
        result["_energy_diff"] = (result["energy_index"] - energy_baseline) / 100.0
        result["econ_impact_energy"] = result["_energy_diff"] * energy_weight * 5  # Scale factor
    else:
        result["econ_impact_energy"] = 0.0
    
    # Inflation component (negative impact for high inflation)
    # Note: inflation_adjustment_factor is a ratio around 1.0 (e.g., 1.05 = 5% inflation)
    # Higher inflation (factor > 1) is bad for ticket sales
    if "inflation_adjustment_factor" in result.columns:
        # Always use 1.0 as the baseline for inflation adjustment factor (ratio)
        inflation_baseline = 1.0
        cpi_weight = cpi_config.get("weight", 0.3)
        # Higher inflation = lower score, so we subtract factor from baseline
        result["_cpi_diff"] = inflation_baseline - result["inflation_adjustment_factor"]
        # Scale: 10% inflation (factor=1.10) gives diff=-0.10, component=-3.0
        result["econ_impact_inflation"] = result["_cpi_diff"] * cpi_weight * 100
    else:
        result["econ_impact_inflation"] = 0.0
    
    # Combined score
    result["economic_impact_score"] = (
        result["econ_impact_consumer_confidence"]
        + result["econ_impact_energy"]
        + result["econ_impact_inflation"]
    )
    
    # Clip to reasonable range
    result["economic_impact_score"] = result["economic_impact_score"].clip(-100, 100)
    
    # Clean up temp columns
    result = result.drop(columns=["_cc_diff", "_energy_diff", "_cpi_diff"], errors="ignore")
    
    # Remove component columns if not requested
    if not include_components:
        result = result.drop(
            columns=[
                "econ_impact_consumer_confidence",
                "econ_impact_energy",
                "econ_impact_inflation",
            ],
            errors="ignore",
        )
    
    return result


def compute_city_economic_summary(
    df: pd.DataFrame,
    city_column: str = "city",
) -> pd.DataFrame:
    """
    Compute city-level economic summary statistics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with city column and economic_impact_score.
    city_column : str, default "city"
        Name of the city column.

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with one row per city.
    """
    if "economic_impact_score" not in df.columns:
        df = compute_economic_impact_score(df)
    
    if city_column not in df.columns:
        return pd.DataFrame()
    
    summary = df.groupby(city_column).agg({
        "economic_impact_score": ["mean", "min", "max", "count"]
    }).reset_index()
    
    # Flatten multi-level columns
    summary.columns = [
        city_column,
        "economic_impact_score_mean",
        "economic_impact_score_min",
        "economic_impact_score_max",
        "count",
    ]
    
    return summary


def score_with_economic_impact(
    df_features: pd.DataFrame,
    model: Optional[Any] = None,
    model_name: str = "model_xgb_remount_postcovid",
    confidence_level: float = 0.8,
    n_bootstrap: int = 200,
    include_economic: bool = True,
    include_components: bool = False,
) -> pd.DataFrame:
    """
    Score features with both model predictions and economic impact.

    Parameters
    ----------
    df_features : pd.DataFrame
        Features for scoring.
    model : Any, optional
        Preloaded model.
    model_name : str
        Model filename.
    confidence_level : float
        Confidence for prediction intervals.
    n_bootstrap : int
        Bootstrap samples for intervals.
    include_economic : bool, default True
        Whether to include economic impact score.
    include_components : bool, default False
        Whether to include economic component scores.

    Returns
    -------
    pd.DataFrame
        Predictions with optional economic impact scores.
    """
    # Get model predictions
    predictions = score_with_uncertainty(
        df_features,
        model=model,
        model_name=model_name,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
    )
    
    # Merge predictions with original features
    result = df_features.copy()
    result = result.join(predictions)
    
    # Add economic impact if requested
    if include_economic:
        result = compute_economic_impact_score(
            result,
            include_components=include_components,
        )
    
    return result
