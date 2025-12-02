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
    """
    json_path = MODELS_DIR / (model_name + ".json")
    pkl_path = MODELS_DIR / (model_name + ".pkl")

    if json_path.exists():
        try:
            import xgboost as xgb

            booster = xgb.Booster()
            booster.load_model(str(json_path))
            return booster
        except Exception as exc:
            logger.warning("Falling back from JSON model: " + str(exc))

    if pkl_path.exists():
        try:
            return joblib.load(pkl_path)
        except Exception as exc:
            logger.error("Could not load model pickle: " + str(exc))
            raise

    raise FileNotFoundError("No model file found for name " + model_name)


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


# =============================================================================
# CORE SCORING
# =============================================================================


def _prepare_features_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare feature DataFrame in the correct column order and dtype for the model.
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

    feature_order = get_feature_order()
    if feature_order is not None:
        # Add any missing expected features as zeros
        for col in feature_order:
            if col not in df_feat.columns:
                df_feat[col] = 0.0
        df_feat = df_feat[feature_order]

    # Coerce everything numeric; non-numeric become  then filled with 0
    df_feat = df_feat.apply(pd.to_numeric, errors="coerce").fillna(0.0)

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

    df_prepared = _prepare_features_for_model(df_features)
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
# ECONOMIC CONTEXT
# =============================================================================

# Default baselines for economic impact scoring
DEFAULT_ECONOMIC_BASELINES: Dict[str, Any] = {
    "consumer_confidence": {
        "baseline": 55.0,
        "good_threshold": 60.0,
        "poor_threshold": 50.0,
    },
    "energy_index": {
        "baseline": 800.0,
    },
    "cpi_base": {
        "baseline": 1.0,
    },
}


def load_economic_baselines() -> Dict[str, Any]:
    """
    Load economic baselines from config file or return defaults.

    Returns
    -------
    dict
        Dictionary with keys: consumer_confidence, energy_index, cpi_base.
    """
    cfg_path = OUTPUTS_DIR / "economic_baselines.yaml"
    loaded = _safe_load_yaml(cfg_path)
    if loaded and all(k in loaded for k in ["consumer_confidence", "energy_index", "cpi_base"]):
        return loaded
    return DEFAULT_ECONOMIC_BASELINES.copy()


def compute_economic_impact_score(
    df: pd.DataFrame,
    include_components: bool = False,
) -> pd.DataFrame:
    """
    Compute economic impact score for each row in the DataFrame.

    The score reflects the economic environment's impact on ticket sales:
    - Positive score: favorable economic conditions
    - Negative score: unfavorable economic conditions
    - Score bounded between -100 and 100

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with optional columns:
        - consumer_confidence_headline
        - consumer_confidence_prairies
        - energy_index
        - inflation_adjustment_factor
    include_components : bool, default False
        If True, include individual component scores as columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with economic_impact_score column added.
        If include_components is True, also adds:
        - econ_impact_consumer_confidence
        - econ_impact_energy
        - econ_impact_inflation
    """
    df_out = df.copy()
    baselines = load_economic_baselines()

    cc_baseline = baselines["consumer_confidence"]["baseline"]
    energy_baseline = baselines["energy_index"]["baseline"]
    inflation_baseline = baselines["cpi_base"]["baseline"]

    n_rows = len(df)

    # Consumer confidence component
    # Prefer prairies regional confidence for Alberta-specific analysis,
    # as it reflects the local economic sentiment more accurately than national headline
    if "consumer_confidence_prairies" in df.columns:
        cc_values = pd.to_numeric(df["consumer_confidence_prairies"], errors="coerce").fillna(cc_baseline)
    elif "consumer_confidence_headline" in df.columns:
        cc_values = pd.to_numeric(df["consumer_confidence_headline"], errors="coerce").fillna(cc_baseline)
    else:
        cc_values = pd.Series([cc_baseline] * n_rows, index=df.index)

    # Energy index component
    if "energy_index" in df.columns:
        energy_values = pd.to_numeric(df["energy_index"], errors="coerce").fillna(energy_baseline)
    else:
        energy_values = pd.Series([energy_baseline] * n_rows, index=df.index)

    # Inflation component
    if "inflation_adjustment_factor" in df.columns:
        inflation_values = pd.to_numeric(df["inflation_adjustment_factor"], errors="coerce").fillna(inflation_baseline)
    else:
        inflation_values = pd.Series([inflation_baseline] * n_rows, index=df.index)

    # Compute component scores
    # Consumer confidence: higher is better, scale relative to baseline
    # Scale: each point above/below baseline = ~2 points of score
    cc_component = (cc_values - cc_baseline) * 2.0

    # Energy index: higher is better for Alberta (oil economy)
    # Scale: percent deviation from baseline, scaled to ~30 max impact
    energy_component = ((energy_values - energy_baseline) / energy_baseline) * 30.0

    # Inflation: lower is better (1.0 = baseline, < 1.0 = deflation/low inflation)
    # Scale: each 0.01 above baseline = -1 point, below = +1 point
    inflation_component = (inflation_baseline - inflation_values) * 100.0

    # Combine components with weights
    # Consumer confidence: 40%, Energy: 30%, Inflation: 30%
    total_score = (
        0.4 * cc_component +
        0.3 * energy_component +
        0.3 * inflation_component
    )

    # Bound to [-100, 100]
    total_score = np.clip(total_score, -100, 100)

    df_out["economic_impact_score"] = total_score

    if include_components:
        df_out["econ_impact_consumer_confidence"] = cc_component
        df_out["econ_impact_energy"] = energy_component
        df_out["econ_impact_inflation"] = inflation_component

    return df_out


def compute_city_economic_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute city-level summary of economic impact scores.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'city' and 'economic_impact_score' columns.

    Returns
    -------
    pd.DataFrame
        Summary with one row per city and mean/min/max of economic_impact_score.
    """
    if "city" not in df.columns or "economic_impact_score" not in df.columns:
        return pd.DataFrame()

    summary = df.groupby("city")["economic_impact_score"].agg(
        economic_impact_score_mean="mean",
        economic_impact_score_min="min",
        economic_impact_score_max="max",
    ).reset_index()

    return summary


def score_with_economic_impact(
    df: pd.DataFrame,
    model: Optional[Any] = None,
    include_economic: bool = True,
) -> pd.DataFrame:
    """
    Score DataFrame with optional economic impact.

    Parameters
    ----------
    df : pd.DataFrame
        Input features DataFrame.
    model : Any, optional
        Pretrained model for ticket forecasting.
    include_economic : bool, default True
        Whether to compute and include economic impact score.

    Returns
    -------
    pd.DataFrame
        DataFrame with forecast_single_tickets and optionally economic_impact_score.
    """
    df_out = df.copy()

    # Try to compute forecast using model
    try:
        preds = predict_point(df, model=model)
        df_out["forecast_single_tickets"] = preds
    except Exception as exc:
        logger.warning("Could not compute model forecast: " + str(exc))
        # Return zeros as fallback
        df_out["forecast_single_tickets"] = 0.0

    # Add economic impact if requested
    if include_economic:
        df_out = compute_economic_impact_score(df_out)

    return df_out


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
        - lower_tickets
        - upper_tickets
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
            "lower_tickets": lower.astype(float),
            "upper_tickets": upper.astype(float),
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
