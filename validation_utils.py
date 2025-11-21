import pandas as pd  # make sure this is at the top of the file


def get_pycaret_predictions(model, feature_df: pd.DataFrame, id_cols=None) -> pd.DataFrame:
    """
    Run the PyCaret model on feature_df and return a DataFrame with predictions.

    We trained the PyCaret model on four numeric feature columns (with underscores):
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
    from pycaret.regression import predict_model

    # Basic guards
    if model is None or feature_df is None or feature_df.empty:
        return pd.DataFrame()

    # Keep a copy of the original for display/IDs
    original_df = feature_df.copy()

    # Work on a copy for PyCaret
    py_df = feature_df.copy()

    # Standardise column names: "Single Tickets - Calgary" -> "Single_Tickets_-_Calgary"
    rename_map = {c: c.replace(" ", "_") for c in py_df.columns}
    py_df = py_df.rename(columns=rename_map)

    # Expected feature columns (the ones we trained on)
    feature_cols = [
        "Single_Tickets_-_Calgary",
        "Single_Tickets_-_Edmonton",
        "Subscription_Tickets_-_Calgary",
        "Subscription_Tickets_-_Edmonton",
    ]

    # Ensure all expected feature columns exist; if any are missing, create as 0
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
