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
        FileNotFoundError: If the model file doesn't exist with helpful instructions.
    """
    _check_pycaret_available()
    from pycaret.regression import load_model
    
    try:
        return load_model(model_name)
    except FileNotFoundError as e:
        # Provide helpful instructions when model file is missing
        raise FileNotFoundError(
            f"Could not find PyCaret model file '{model_name}.pkl'.\n\n"
            f"To use the Model Validation feature, you need to:\n"
            f"1. Train a PyCaret regression model on your historical data\n"
            f"2. Save it using: save_model(model, '{model_name}')\n"
            f"3. Place the '{model_name}.pkl' file in the project root directory\n\n"
            f"Example training code:\n"
            f"  from pycaret.regression import setup, compare_models, save_model\n"
            f"  s = setup(data=your_df, target='actual_tickets', session_id=123)\n"
            f"  best_model = compare_models()\n"
            f"  save_model(best_model, '{model_name}')\n\n"
            f"Original error: {e}"
        ) from e


import pandas as pd  # make sure this is at the top of the file


def get_pycaret_predictions(model, feature_df: pd.DataFrame, id_cols=None) -> pd.DataFrame:
    """
    Run the PyCaret model on feature_df and return a DataFrame with predictions.

    - Keeps the original columns for display / IDs.
    - Internally renames columns (spaces -> underscores) to match how the model was trained.
    """
    from pycaret.regression import predict_model

    if model is None or feature_df is None or feature_df.empty:
        return pd.DataFrame()

    # Keep original for IDs / display
    original_df = feature_df.copy()

    # Prepare a version for PyCaret with the same naming as in train_pycaret_model.py
    py_df = feature_df.copy()
    py_df.columns = [c.replace(" ", "_") for c in py_df.columns]

    # Run model
    preds = predict_model(model, data=py_df)

    # PyCaret regression normally uses "Label" for the prediction column
    pred_col = "Label" if "Label" in preds.columns else preds.columns[-1]

    # Start from original columns, add prediction
    out = original_df.copy()
    out["PyCaret_Prediction"] = preds[pred_col].values

    # If caller only wants ID columns + prediction, trim to that
    if id_cols:
        keep = [c for c in id_cols if c in out.columns] + ["PyCaret_Prediction"]
        return out[keep]

    return out



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
