# validation_utils.py

from __future__ import annotations

import sys
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# PyCaret supported Python version range
PYCARET_MIN_PYTHON = (3, 9)
PYCARET_MAX_PYTHON = (3, 12)


def is_python_compatible_with_pycaret() -> bool:
    """
    Check if the current Python version is compatible with PyCaret.
    
    Returns:
        bool: True if Python version is within PyCaret's supported range (3.9-3.12).
    """
    current_version = sys.version_info[:2]
    return PYCARET_MIN_PYTHON <= current_version <= PYCARET_MAX_PYTHON


def get_pycaret_compatibility_message() -> str:
    """
    Get a user-friendly message about PyCaret Python version compatibility.
    
    Returns:
        str: Message explaining the Python version compatibility issue.
    """
    # Generate list of supported versions dynamically
    supported_versions = [
        f"{PYCARET_MIN_PYTHON[0]}.{minor}"
        for minor in range(PYCARET_MIN_PYTHON[1], PYCARET_MAX_PYTHON[1] + 1)
    ]
    
    if len(supported_versions) > 1:
        versions_str = ", ".join(supported_versions[:-1]) + f", and {supported_versions[-1]}"
    else:
        versions_str = supported_versions[0]
    
    return (
        f"PyCaret only supports Python {versions_str}. "
        f"Your Python version is {sys.version_info.major}.{sys.version_info.minor}. "
        f"To use PyCaret's Model Validation feature, please use Python {PYCARET_MAX_PYTHON[0]}.{PYCARET_MAX_PYTHON[1]} or earlier."
    )


def _check_pycaret_available():
    """
    Helper function to check if pycaret is available and raise a helpful error if not.
    
    Checks Python version compatibility (PyCaret supports Python 3.9-3.12) before
    attempting to import PyCaret.
    
    Raises:
        RuntimeError: If Python version is incompatible with PyCaret (not 3.9-3.12).
        ImportError: If pycaret is not installed with installation instructions.
    """
    # Check Python version first
    if not is_python_compatible_with_pycaret():
        raise RuntimeError(get_pycaret_compatibility_message())
    
    try:
        import pycaret
    except ImportError:
        raise ImportError(
            "PyCaret is required for this functionality. "
            "Install it with: pip install git+https://github.com/pycaret/pycaret.git@master"
        )


def load_pycaret_model(model_name: str):
    """
    Load a saved PyCaret regression model.

    Example:
        model = load_pycaret_model("title_demand_model")
    
    Note: This function requires pycaret to be installed.
    Install it with: pip install git+https://github.com/pycaret/pycaret.git@master
    
    Raises:
        ImportError: If pycaret is not installed.
    """
    _check_pycaret_available()
    from pycaret.regression import load_model
    return load_model(model_name)


def get_pycaret_predictions(
    model, feature_df: pd.DataFrame, id_cols: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Run a PyCaret model on feature_df and return a dataframe with predictions.

    Assumes the target column was removed when you trained the model.
    
    Note: This function requires pycaret to be installed.
    Install it with: pip install git+https://github.com/pycaret/pycaret.git@master
    
    Raises:
        ImportError: If pycaret is not installed.
    """
    _check_pycaret_available()
    from pycaret.regression import predict_model
    
    preds = predict_model(model, data=feature_df.copy())
    # PyCaret's predict_model usually returns 'Label' as the prediction column.
    result = preds.copy()
    if id_cols:
        # keep IDs plus label
        keep_cols = [c for c in id_cols if c in result.columns] + ["Label"]
        result = result[keep_cols]
    return result.rename(columns={"Label": "pycaret_pred"})


def compute_model_metrics(
    df: pd.DataFrame, actual_col: str, pred_col: str
) -> dict:
    """
    Compute MAE, RMSE, and R² for a given prediction column.
    """
    y_true = df[actual_col].astype(float)
    y_pred = df[pred_col].astype(float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def build_comparison_frame(
    base_df: pd.DataFrame,
    actual_col: str,
    your_pred_col: str,
    pycaret_pred_col: str = "pycaret_pred",
    id_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Build a comparison dataframe with errors and absolute error deltas
    between your model and the PyCaret model.
    """
    df = base_df.copy()

    # errors
    df["err_your"] = df[your_pred_col] - df[actual_col]
    df["err_pycaret"] = df[pycaret_pred_col] - df[actual_col]

    # absolute errors
    df["abs_err_your"] = df["err_your"].abs()
    df["abs_err_pycaret"] = df["err_pycaret"].abs()

    # which model is better for each row
    df["better_model"] = np.where(
        df["abs_err_your"] < df["abs_err_pycaret"], "Your model", "PyCaret"
    )

    # order the columns nicely if id_cols exist
    id_cols = id_cols or []
    ordered_cols = (
        id_cols
        + [actual_col, your_pred_col, pycaret_pred_col,
           "err_your", "err_pycaret",
           "abs_err_your", "abs_err_pycaret",
           "better_model"]
    )
    ordered_cols = [c for c in ordered_cols if c in df.columns]

    return df[ordered_cols]
