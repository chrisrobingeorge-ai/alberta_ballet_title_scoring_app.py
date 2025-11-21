# validation_utils.py

from __future__ import annotations

import sys
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────────────────────
# PyCaret version support
# ─────────────────────────────────────────────────────────────

PYCARET_MIN_PYTHON = (3, 9)
PYCARET_MAX_PYTHON = (3, 12)


def is_python_compatible_with_pycaret() -> bool:
    """
    Check if the current Python version is compatible with PyCaret.
    Returns True if Python version is within PyCaret's supported range (3.9–3.12).
    """
    current_version = sys.version_info[:2]
    return PYCARET_MIN_PYTHON <= current_version <= PYCARET_MAX_PYTHON


def get_pycaret_compatibility_message() -> str:
    """
    Get a user-friendly message about PyCaret Python version compatibility.
    """
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
        f"To use PyCaret's Model Validation feature, please use Python "
        f"{PYCARET_MAX_PYTHON[0]}.{PYCARET_MAX_PYTHON[1]} or earlier."
    )


def _check_pycaret_available():
    """
    Check Python version and PyCaret availability.
    Raises:
        RuntimeError if Python version is incompatible.
        ImportError if PyCaret is not installed.
    """
    if not is_python_compatible_with_pycaret():
        raise RuntimeError(get_pycaret_compatibility_message())

    try:
        import pycaret  # noqa: F401
    except ImportError:
        raise ImportError(
            "PyCaret is required for the Model Validation feature. "
            "Install it with:\n"
            "  pip install git+https://github.com/pycaret/pycaret.git@master"
        )


# ─────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────

def load_pycaret_model(model_name: str):
    """
    Load a saved PyCaret regression model.

    Example:
        model = load_pycaret_model("title_demand_model")
    """
    _check_pycaret_available()
    from pycaret.regression import load_model

    try:
        return load_model(model_name)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find PyCaret model file '{model_name}.pkl'.\n\n"
            f"To use the Model Validation feature, you need to:\n"
            f"1. Train a PyCaret regression model on your historical data\n"
            f"2. Save it using: save_model(model, '{model_name}')\n"
            f"3. Place the '{model_name}.pkl' file in the project root directory\n\n"
            f"Original error: {e}"
        ) from e


# ─────────────────────────────────────────────────────────────
# Prediction helper
# ─────────────────────────────────────────────────────────────

def get_pycaret_predictions(
    model,
    feature_df: pd.DataFrame,
    id_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Run the PyCaret model on feature_df and return a DataFrame with predictions.

    IMPORTANT:
    We trained the PyCaret model on four numeric feature columns (after renaming
    spaces to underscores in train_pycaret_model.py):

        - Single_Tickets_-_Calgary
        - Single_Tickets_-_Edmonton
        - Subscription_Tickets_-_Calgary
        - Subscription_Tickets_-_Edmonton

    This function:
    - Starts from whatever feature_df the validation page provides.
    - Renames columns (spaces -> underscores) to match training.
    - Constructs a PyCaret input dataframe with exactly those four columns,
      creating missing ones as zeros if necessary.
    - Runs predict_model on that clean dataframe.
    - Returns the original rows plus a PyCaret_Prediction column.
    """
    _check_pycaret_available()
    from pycaret.regression import predict_model

    if model is None or feature_df is None or feature_df.empty:
        return pd.DataFrame()

    # Keep the original for IDs / display
    original_df = feature_df.copy()

    # Work on a copy for PyCaret
    py_df = feature_df.copy()

    # Standardise column names to match training: "Single Tickets - Calgary" -> "Single_Tickets_-_Calgary"
    rename_map = {c: c.replace(" ", "_") for c in py_df.columns}
    py_df = py_df.rename(columns=rename_map)

    # Expected feature columns based on train_pycaret_model.py
    feature_cols = [
        "Single_Tickets_-_Calgary",
        "Single_Tickets_-_Edmonton",
        "Subscription_Tickets_-_Calgary",
        "Subscription_Tickets_-_Edmonton",
    ]

    # Ensure all expected feature columns exist; create missing ones as 0.0
    for col in feature_cols:
        if col not in py_df.columns:
            py_df[col] = 0.0

    # Restrict to the exact feature set PyCaret expects
    py_input = py_df[feature_cols]

    # Run model
    preds = predict_model(model, data=py_input)

    # PyCaret usually stores predictions in "Label"
    pred_col = "Label" if "Label" in preds.columns else preds.columns[-1]

    # Build output: original data + prediction
    out = original_df.copy()
    out["PyCaret_Prediction"] = preds[pred_col].values

    # If caller wants only id_cols + prediction, trim
    if id_cols:
        keep = [c for c in id_cols if c in out.columns] + ["PyCaret_Prediction"]
        return out[keep]

    return out


# ─────────────────────────────────────────────────────────────
# Metrics + comparison frame
# ─────────────────────────────────────────────────────────────

def compute_model_metrics(
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
) -> Dict[str, float]:
    """
    Compute MAE, RMSE, and R² for a given prediction column.
    """
    y_true = df[actual_col].astype(float)
    y_pred = df[pred_col].astype(float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def build_comparison_frame(
    base_df: pd.DataFrame,
    actual_col: str,
    your_pred_col: str,
    pycaret_pred_col: str = "PyCaret_Prediction",
    id_cols: Optional[List[str]] = None,
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
        df["abs_err_your"] < df["abs_err_pycaret"],
        "Your model",
        "PyCaret",
    )

    # order the columns nicely if id_cols exist
    id_cols = id_cols or []
    ordered_cols = (
        id_cols
        + [
            actual_col,
            your_pred_col,
            pycaret_pred_col,
            "err_your",
            "err_pycaret",
            "abs_err_your",
            "abs_err_pycaret",
            "better_model",
        ]
    )
    ordered_cols = [c for c in ordered_cols if c in df.columns]

    return df[ordered_cols]
