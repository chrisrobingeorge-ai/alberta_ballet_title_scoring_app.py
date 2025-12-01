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
            logger.warning("Could not load training schema: " + str(e))

    return None


def validate_input_schema(
    df: pd.DataFrame,
    training_schema: Optional[Dict[str, Any]] = None,
    raise_on_error: bool = False
) -> Tuple[bool, List[str]]:
    """Validate input DataFrame schema against training schema.

    Args:
        df: Input DataFrame to validate
        training_schema: Optional training schema dictionary
        raise_on_error: Whether to raise ValueError on validation failure

    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    if training_schema is None:
        training_schema = load_training_schema()

    warnings_list: List[str] = []

    if not training_schema:
        # No schema to validate against
        warnings_list.append("No training schema available for validation")
        for warning_msg in warnings_list:
            warnings.warn(warning_msg, SchemaValidationWarning)
        return True, warnings_list

    training_features = training_schema.get("features", [])
    n_features = training_schema.get("n_features", len(training_features))

    # Check number of features
    if df.shape[1] != n_features:
        msg = (
            "Input has " + str(df.shape[1]) +
            " features but training used " + str(n_features)
        )
        warnings_list.append(msg)

    # Check for missing and extra columns
    input_cols = set(df.columns.tolist())
    training_cols = set(training_features)

    missing_cols = training_cols - input_cols
    extra_cols = input_cols - training_cols

    if missing_cols:
        warnings_list.append(
            "Missing expected columns: " + ", ".join(sorted(missing_cols))
        )
    if extra_cols:
        warnings_list.append(
            "Unexpected extra columns: " + ", ".join(sorted(extra_cols))
        )

    # Check column order if we have exact feature list
    if training_features:
        training_order = list(training_features)
        actual_order = list(df.columns)
        if training_order != actual_order:
            warnings_list.append("Column order differs from training schema")

    is_valid = len(warnings_list) == 0

    # Emit warnings
    for warning_msg in warnings_list:
        warnings.warn(warning_msg, SchemaValidationWarning)

    if not is_valid and raise_on_error:
        raise ValueError("Schema validation failed: " + "; ".join(warnings_list))

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


def _get_rf_tree_predictions(model, df_features: pd.DataFrame) -> np.ndarray:
    """Get per-tree predictions for RandomForest-like models."""
    # Support sklearn RandomForest-style API
    if hasattr(model, "estimators_"):
        tree_preds = []
        for est in model.estimators_:
            tree_preds.append(est.predict(df_features))
        return np.vstack(tree_preds).T
    raise ValueError("Model does not have estimators_ attribute for RF-style trees.")


def _bootstrap_predictions(
    model,
    df_features: pd.DataFrame,
    n_bootstrap: int = 100,
    random_state: Optional[int] = None
) -> np.ndarray:
    """Bootstrap predictions for generic models without tree access."""
    rng = np.random.RandomState(random_state)
    n = len(df_features)
    boot_preds = np.zeros((n, n_bootstrap))

    for b in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        sample = df_features.iloc[idx]
        boot_preds[:, b] = model.predict(sample)

    return boot_preds


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
        confidence_level: Confidence level for prediction intervals (0-1)
        n_bootstrap: Number of bootstrap samples

    Returns:
        DataFrame with:
            - forecast_single_tickets (point estimate)
            - lower bound
            - upper bound
    """
    model = model or load_model()

    # Base predictions
    base_pred = score_dataframe(df_features, model=model, validate_schema=True)

    alpha = 1.0 - confidence_level
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)

    # Try RF-style tree predictions first
    try:
        tree_preds = _get_rf_tree_predictions(model, df_features)
        lower = np.percentile(tree_preds, lower_q, axis=1)
        upper = np.percentile(tree_preds, upper_q, axis=1)
    except Exception:
        # Fallback to bootstrap predictions
        boot_preds = _bootstrap_predictions(
            model,
            df_features,
            n_bootstrap=n_bootstrap,
            random_state=42
        )
        lower = np.percentile(boot_preds, lower_q, axis=1)
        upper = np.percentile(boot_preds, upper_q, axis=1)

    result = pd.DataFrame(
        {
            "forecast_single_tickets": base_pred.values,
            "lower_tickets": lower,
            "upper_tickets": upper,
        },
        index=df_features.index,
    )

    return result


# =============================================================================
# ECONOMIC IMPACT / STORYTELLING HOOKS (STUBS)
# =============================================================================


def load_economic_config(
    path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Load economic factor configuration (if available)."""
    cfg_path = Path(path) if path is not None else (
        Path(__file__).parent.parent / "config" / "economic_alberta.yaml"
    )
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("Could not load economic config: " + str(e))
        return {}


def attach_economic_context(
    df_scored: pd.DataFrame,
    context: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Attach lightweight economic context columns for storytelling.

    This does not affect predictions, just annotations.
    """
    if context is None:
        context = {}

    df = df_scored.copy()

    # Example: annotate whether macro sentiment is tailwind/headwind
    macro = context.get("macro_sentiment")
    if macro is not None:
        df["macro_sentiment"] = macro

    return df


# =============================================================================
# HIGH-LEVEL PLANNING API
# =============================================================================


def score_runs_for_planning(
    df_runs: pd.DataFrame,
    confidence_level: float = 0.8,
    n_bootstrap: int = 200,
    model=None,
    attach_context: bool = False,
    economic_context: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """High-level helper for season / run planning.

    Args:
        df_runs: DataFrame of proposed runs with the same feature columns
                 as used in training.
        confidence_level: e.g. 0.8 for an 80 percent prediction interval.
        n_bootstrap: number of bootstrap draws for the interval.
        model: optional pretrained model (otherwise loaded from disk).
        attach_context: whether to annotate with economic context.
        economic_context: optional dict of context values.

    Returns:
        DataFrame with original columns plus:
            - forecast_single_tickets
            - lower_tickets_<XX>
            - upper_tickets_<XX>
            - (optional) macro_sentiment or other context columns
    """
    # Ensure we are not mutating the original frame
    df_features = df_runs.copy()

    # Score with uncertainty
    df_pred = score_with_uncertainty(
        df_features,
        model=model,
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
