"""
Model training module with statistical rigor enhancements.

Features:
- Time-based cross-validation (TimeSeriesSplit)
- Hyperparameter tuning (RandomizedSearchCV)
- Ensemble models (Gradient Boosting)
- Feature importance tracking
- Log-transformed target support
- Model metadata versioning
"""

from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.dataset import build_dataset
from ml.time_splits import TimeSeriesCVSplitter, chronological_train_test_split, assert_chronological_split

logger = logging.getLogger(__name__)


class MissingDateColumnError(ValueError):
    """Raised when a date column is required for time-aware CV but not found.
    
    Forecasting models require temporal ordering to prevent future data leakage.
    This error is raised when attempting to train a forecasting model without
    a valid date column available for chronological splitting.
    
    Attributes:
        searched_columns: List of column names that were searched for.
        available_columns: List of columns available in the dataset.
    """
    
    def __init__(
        self,
        message: str = None,
        searched_columns: List[str] = None,
        available_columns: List[str] = None
    ):
        self.searched_columns = searched_columns or []
        self.available_columns = available_columns or []
        
        if message is None:
            message = (
                "No date column found for time-aware cross-validation. "
                "Forecasting models require a date column to ensure chronological "
                "train/test splits and prevent future data leakage. "
                f"Searched for columns: {self.searched_columns}. "
                f"Available columns: {self.available_columns[:20]}{'...' if len(self.available_columns) > 20 else ''}. "
                "Provide a valid date_column argument or ensure your dataset includes "
                "one of the expected date columns (e.g., 'end_date', 'start_date')."
            )
        super().__init__(message)


MODELS_DIR = Path(__file__).parent.parent / "models"
CONFIGS_DIR = Path(__file__).parent.parent / "configs"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
METRICS_DIR = Path(__file__).parent.parent / "metrics"

# Common date column names to search for when inferring the date column
DATE_COLUMN_CANDIDATES = ["end_date", "start_date", "date", "opening_date", "performance_date"]


def load_ml_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load ML configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default configs/ml_config.yaml
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = CONFIGS_DIR / "ml_config.yaml"
    
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    
    # Return default config if file doesn't exist
    return {
        "model": {
            "type": "random_forest",
            "enable_tuning": False,
            "cv_folds": 5,
            "random_state": 42,
            "n_iter_search": 30
        },
        "random_forest": {
            "defaults": {
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt"
            }
        },
        "target": {
            "use_log_transform": False
        },
        "cross_validation": {
            "type": "time_series",
            "n_splits": 5
        },
        "explainability": {
            "export_importances": True,
            "importance_output_path": "outputs/feature_importance.json",
            "top_n_features": 20
        },
        "versioning": {
            "enabled": True,
            "metadata_path": "models/model_metadata.json"
        }
    }


def apply_target_transform(
    y: pd.Series,
    use_log: bool = True
) -> Tuple[pd.Series, bool]:
    """Apply optional log1p transform to target variable.
    
    Args:
        y: Target series
        use_log: Whether to apply log1p transform
        
    Returns:
        Tuple of (transformed target, whether transform was applied)
    """
    if use_log:
        # Log1p handles zeros safely: log1p(x) = log(1 + x)
        return np.log1p(y), True
    return y, False


def inverse_target_transform(
    y_pred: np.ndarray,
    was_log_transformed: bool
) -> np.ndarray:
    """Reverse the target transformation for predictions.
    
    Args:
        y_pred: Predicted values (possibly in log space)
        was_log_transformed: Whether log transform was applied during training
        
    Returns:
        Predictions in original scale
    """
    if was_log_transformed:
        return np.expm1(y_pred)  # expm1(x) = exp(x) - 1, inverse of log1p
    return y_pred


def get_hyperparam_grid(
    model_type: str,
    config: Dict[str, Any]
) -> Dict[str, List[Any]]:
    """Get hyperparameter search grid for the specified model type.
    
    Args:
        model_type: "random_forest" or "gradient_boosting"
        config: ML configuration dictionary
        
    Returns:
        Dictionary of hyperparameter lists for RandomizedSearchCV
    """
    if model_type == "random_forest":
        rf_config = config.get("random_forest", {})
        return {
            "rf__n_estimators": rf_config.get("n_estimators", [100, 200, 300]),
            "rf__max_depth": rf_config.get("max_depth", [5, 10, 15, None]),
            "rf__min_samples_split": rf_config.get("min_samples_split", [2, 5, 10]),
            "rf__min_samples_leaf": rf_config.get("min_samples_leaf", [1, 2, 4]),
            "rf__max_features": rf_config.get("max_features", ["sqrt", "log2", None]),
        }
    elif model_type == "gradient_boosting":
        gb_config = config.get("gradient_boosting", {})
        return {
            "gb__n_estimators": gb_config.get("n_estimators", [100, 200, 300]),
            "gb__max_depth": gb_config.get("max_depth", [3, 5, 7]),
            "gb__learning_rate": gb_config.get("learning_rate", [0.01, 0.05, 0.1]),
            "gb__min_samples_split": gb_config.get("min_samples_split", [2, 5, 10]),
            "gb__min_samples_leaf": gb_config.get("min_samples_leaf", [1, 2, 4]),
            "gb__subsample": gb_config.get("subsample", [0.8, 0.9, 1.0]),
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_model_pipeline(
    cat_cols: List[str],
    num_cols: List[str],
    model_type: str = "random_forest",
    config: Optional[Dict[str, Any]] = None,
    random_state: int = 42
) -> Pipeline:
    """Create a sklearn pipeline with preprocessing and model.
    
    Args:
        cat_cols: List of categorical column names
        num_cols: List of numeric column names
        model_type: "random_forest" or "gradient_boosting"
        config: ML configuration dictionary
        random_state: Random seed for reproducibility
        
    Returns:
        sklearn Pipeline with preprocessor and model
    """
    config = config or load_ml_config()
    
    # Create preprocessor
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    
    # Create model based on type
    if model_type == "random_forest":
        defaults = config.get("random_forest", {}).get("defaults", {})
        model = RandomForestRegressor(
            n_estimators=defaults.get("n_estimators", 300),
            max_depth=defaults.get("max_depth"),
            min_samples_split=defaults.get("min_samples_split", 2),
            min_samples_leaf=defaults.get("min_samples_leaf", 1),
            max_features=defaults.get("max_features", "sqrt"),
            random_state=random_state,
            n_jobs=-1
        )
        model_step = ("rf", model)
    elif model_type == "gradient_boosting":
        defaults = config.get("gradient_boosting", {}).get("defaults", {})
        model = GradientBoostingRegressor(
            n_estimators=defaults.get("n_estimators", 200),
            max_depth=defaults.get("max_depth", 5),
            learning_rate=defaults.get("learning_rate", 0.1),
            min_samples_split=defaults.get("min_samples_split", 2),
            min_samples_leaf=defaults.get("min_samples_leaf", 1),
            subsample=defaults.get("subsample", 0.9),
            random_state=random_state
        )
        model_step = ("gb", model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return Pipeline([("pre", pre), model_step])


def extract_feature_importances(
    pipe: Pipeline,
    feature_names: List[str],
    cat_cols: List[str],
    model_type: str = "random_forest"
) -> Dict[str, float]:
    """Extract feature importances from the trained pipeline.
    
    Args:
        pipe: Trained sklearn Pipeline
        feature_names: Original feature names
        cat_cols: Categorical column names (for expanding one-hot features)
        model_type: Model type for accessing the correct step
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    model_key = "rf" if model_type == "random_forest" else "gb"
    model = pipe.named_steps[model_key]
    
    # Get the preprocessor
    preprocessor = pipe.named_steps["pre"]
    
    # Get feature names after transformation
    try:
        transformed_names = preprocessor.get_feature_names_out()
    except AttributeError:
        # Fallback for older sklearn versions
        transformed_names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "num":
                transformed_names.extend(cols)
            elif name == "cat" and hasattr(trans, "get_feature_names_out"):
                transformed_names.extend(trans.get_feature_names_out(cols))
    
    importances = model.feature_importances_
    
    # Map importances back to original features
    feature_importance = {}
    
    for i, name in enumerate(transformed_names):
        if i < len(importances):
            # Find original feature name (handle one-hot encoded features)
            original_name = None
            for orig_feature in feature_names:
                if name.startswith(f"cat__{orig_feature}_") or name == f"num__{orig_feature}" or name == orig_feature:
                    original_name = orig_feature
                    break
            
            if original_name:
                if original_name in feature_importance:
                    feature_importance[original_name] += importances[i]
                else:
                    feature_importance[original_name] = importances[i]
            else:
                feature_importance[name] = importances[i]
    
    # Sort by importance
    return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))


def compute_subgroup_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    df_features: pd.DataFrame,
    subgroups: List[str]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute evaluation metrics for each subgroup.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        df_features: DataFrame with subgroup columns
        subgroups: List of column names to compute metrics for
        
    Returns:
        Nested dictionary: {subgroup: {value: {metric: score}}}
    """
    results = {}
    
    for subgroup in subgroups:
        if subgroup not in df_features.columns:
            continue
        
        results[subgroup] = {}
        
        for value in df_features[subgroup].dropna().unique():
            mask = df_features[subgroup] == value
            if mask.sum() < 2:  # Need at least 2 samples
                continue
            
            y_t = y_true[mask]
            y_p = y_pred[mask]
            
            results[subgroup][str(value)] = {
                "mae": float(mean_absolute_error(y_t, y_p)),
                "rmse": float(np.sqrt(mean_squared_error(y_t, y_p))),
                "r2": float(r2_score(y_t, y_p)) if len(y_t) > 1 else 0.0,
                "n_samples": int(mask.sum())
            }
    
    return results


def save_model_metadata(
    model_path: Path,
    metrics: Dict[str, Any],
    feature_names: List[str],
    hyperparams: Dict[str, Any],
    config: Dict[str, Any]
) -> None:
    """Save model metadata for versioning and reproducibility.
    
    Args:
        model_path: Path where the model was saved
        metrics: Training/evaluation metrics
        feature_names: List of feature names used
        hyperparams: Final hyperparameters used
        config: ML configuration dictionary
    """
    versioning_config = config.get("versioning", {})
    if not versioning_config.get("enabled", True):
        return
    
    metadata_path = Path(versioning_config.get("metadata_path", "models/model_metadata.json"))
    if not metadata_path.is_absolute():
        metadata_path = Path(__file__).parent.parent / metadata_path
    
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "model_path": str(model_path),
        "training_date": datetime.now().isoformat(),
        "metrics": metrics,
        "n_features": len(feature_names),
        "features": feature_names if versioning_config.get("track_features", True) else None,
        "hyperparameters": hyperparams if versioning_config.get("track_hyperparams", True) else None,
        "config_hash": hash(str(config)) % 10**8,  # Simple hash for config tracking
    }
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Model metadata saved to {metadata_path}")


def save_feature_importances(
    importances: Dict[str, float],
    config: Dict[str, Any]
) -> None:
    """Save feature importances to JSON file.
    
    Args:
        importances: Dictionary of feature names to importance scores
        config: ML configuration dictionary
    """
    explainability_config = config.get("explainability", {})
    if not explainability_config.get("export_importances", True):
        return
    
    output_path = Path(explainability_config.get("importance_output_path", "outputs/feature_importance.json"))
    if not output_path.is_absolute():
        output_path = Path(__file__).parent.parent / output_path
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    top_n = explainability_config.get("top_n_features", 20)
    top_features = dict(list(importances.items())[:top_n])
    
    output_data = {
        "feature_importances": importances,
        "top_features": top_features,
        "generated_at": datetime.now().isoformat()
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Feature importances saved to {output_path}")


def save_evaluation_metrics(
    metrics: Dict[str, Any],
    subgroup_metrics: Dict[str, Dict[str, Dict[str, float]]],
    output_name: str = "evaluation_metrics"
) -> None:
    """Save evaluation metrics to the metrics folder.
    
    Args:
        metrics: Overall model metrics
        subgroup_metrics: Metrics broken down by subgroups
        output_name: Base name for output files
    """
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save overall metrics as JSON
    overall_path = METRICS_DIR / f"{output_name}.json"
    with open(overall_path, "w") as f:
        json.dump({
            "overall": metrics,
            "by_subgroup": subgroup_metrics,
            "generated_at": datetime.now().isoformat()
        }, f, indent=2)
    
    # Save subgroup metrics as CSV for easier analysis
    rows = []
    for subgroup, values in subgroup_metrics.items():
        for value, group_metrics in values.items():
            row = {"subgroup": subgroup, "value": value}
            row.update(group_metrics)
            rows.append(row)
    
    if rows:
        subgroup_df = pd.DataFrame(rows)
        subgroup_df.to_csv(METRICS_DIR / f"{output_name}_by_subgroup.csv", index=False)
    
    logger.info(f"Evaluation metrics saved to {METRICS_DIR}")


def train_baseline_model(
    save_path: Path | None = None,
    date_column: str | None = None,
    config_path: Path | None = None
) -> dict:
    """Train a baseline model for title demand forecasting.
    
    Uses chronological train/test split to prevent future data from leaking
    into predictions. A date column is required for forecasting tasks.
    
    Supports:
    - Time-based cross-validation
    - Hyperparameter tuning via RandomizedSearchCV
    - Log-transformed targets for skewed ticket counts
    - Feature importance tracking
    - Model metadata versioning
    
    Args:
        save_path: Optional path to save the trained model
        date_column: Name of date column for chronological split (e.g., 'end_date').
                    If None, will try to find a date column from common candidates.
        config_path: Optional path to ML config YAML file
    
    Returns:
        Dictionary with model path, training metrics, and feature importances
    
    Raises:
        MissingDateColumnError: If no date column is found. Forecasting models
            require a date column for time-aware cross-validation to prevent
            future data leakage.
    """
    # Load configuration
    config = load_ml_config(config_path)
    model_config = config.get("model", {})
    target_config = config.get("target", {})
    cv_config = config.get("cross_validation", {})
    
    # Build dataset
    X, y = build_dataset(
        theme_filters=["Production Attributes", "Historical Sales Trends", "Timing & Schedule Factors"],
        status=None,
        forecast_time_only=True,
    )
    
    # Store original feature names
    feature_names = X.columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    
    # Apply target transform if configured
    use_log_transform = target_config.get("use_log_transform", False)
    y_transformed, was_log_transformed = apply_target_transform(y, use_log_transform)
    
    # Find date column for chronological splitting
    date_col = date_column
    if date_col is None:
        for col_name in DATE_COLUMN_CANDIDATES:
            if col_name in X.columns:
                date_col = col_name
                break
    
    # Enforce time-aware CV for forecasting: raise error if no date column found
    if not date_col or date_col not in X.columns:
        raise MissingDateColumnError(
            searched_columns=list(DATE_COLUMN_CANDIDATES),
            available_columns=X.columns.tolist()
        )
    
    # Use chronological split to prevent future leakage
    combined = X.copy()
    combined["_target"] = y_transformed.values
    
    train_df, test_df = chronological_train_test_split(combined, date_col, test_ratio=0.2)
    
    # Drop the target and date column (date column is only used for splitting, not as a feature)
    Xtr = train_df.drop(columns=["_target", date_col])
    ytr = train_df["_target"]
    Xte = test_df.drop(columns=["_target", date_col])
    yte = test_df["_target"]
    
    # Update feature_names and column lists to exclude date column
    feature_names = [c for c in feature_names if c != date_col]
    cat_cols = [c for c in cat_cols if c != date_col]
    num_cols = [c for c in num_cols if c != date_col]
    
    # Verify chronological ordering
    assert_chronological_split(train_df, test_df, date_col)
    
    time_aware_split = True
    
    # Create model pipeline (after updating column lists to exclude date column)
    model_type = model_config.get("type", "random_forest")
    random_state = model_config.get("random_state", 42)
    pipe = create_model_pipeline(cat_cols, num_cols, model_type, config, random_state)
    
    # Hyperparameter tuning if enabled
    best_params = {}
    if model_config.get("enable_tuning", False):
        logger.info("Starting hyperparameter tuning...")
        
        # Use TimeSeriesSplit for CV during tuning
        n_splits = cv_config.get("n_splits", 5)
        cv = TimeSeriesSplit(n_splits=n_splits)
        
        param_grid = get_hyperparam_grid(model_type, config)
        n_iter = model_config.get("n_iter_search", 30)
        
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring="neg_mean_absolute_error",
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
        
        search.fit(Xtr, ytr)
        pipe = search.best_estimator_
        best_params = search.best_params_
        logger.info(f"Best parameters: {best_params}")
    else:
        # Fit without tuning
        pipe.fit(Xtr, ytr)
    
    # Make predictions
    preds_transformed = pipe.predict(Xte)
    
    # Inverse transform predictions for metrics
    preds = inverse_target_transform(preds_transformed, was_log_transformed)
    yte_original = inverse_target_transform(yte.values, was_log_transformed)
    
    # Calculate metrics
    mae = float(mean_absolute_error(yte_original, preds))
    rmse = float(np.sqrt(mean_squared_error(yte_original, preds)))
    r2 = float(r2_score(yte_original, preds))
    
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "features_used": feature_names,
        "time_aware_split": time_aware_split,
        "log_transformed": was_log_transformed,
        "model_type": model_type,
    }
    
    # Extract and save feature importances
    feature_importances = extract_feature_importances(pipe, feature_names, cat_cols, model_type)
    save_feature_importances(feature_importances, config)
    
    # Compute subgroup metrics if subgroup columns exist
    subgroups = config.get("evaluation", {}).get("subgroups", ["genre", "season", "city"])
    subgroup_metrics = compute_subgroup_metrics(
        pd.Series(yte_original, index=Xte.index),
        preds,
        Xte,
        subgroups
    )
    
    # Save evaluation metrics
    save_evaluation_metrics(metrics, subgroup_metrics)
    
    # Save model
    MODELS_DIR.mkdir(exist_ok=True)
    out_path = save_path or (MODELS_DIR / "title_demand_rf.pkl")
    joblib.dump(pipe, out_path)
    
    # Save model metadata
    save_model_metadata(out_path, metrics, feature_names, best_params, config)
    
    return {
        "model_path": str(out_path),
        "metrics": metrics,
        "feature_importances": feature_importances,
        "best_params": best_params,
        "subgroup_metrics": subgroup_metrics
    }


def train_with_cross_validation(
    config_path: Path | None = None,
    date_column: str | None = None
) -> Dict[str, Any]:
    """Train model with time-series cross-validation for robust evaluation.
    
    Uses walk-forward validation to ensure no future data leakage.
    A date column is required for time-aware cross-validation.
    
    Args:
        config_path: Optional path to ML config YAML file
        date_column: Name of date column for chronological splits.
                    If None, will try to find a date column from common candidates.
        
    Returns:
        Dictionary with CV scores and final model metrics
    
    Raises:
        MissingDateColumnError: If no date column is found. Forecasting models
            require a date column for time-aware cross-validation to prevent
            future data leakage.
    """
    config = load_ml_config(config_path)
    cv_config = config.get("cross_validation", {})
    model_config = config.get("model", {})
    target_config = config.get("target", {})
    
    # Build dataset
    X, y = build_dataset(
        theme_filters=["Production Attributes", "Historical Sales Trends", "Timing & Schedule Factors"],
        status=None,
        forecast_time_only=True,
    )
    
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    
    # Apply target transform
    use_log_transform = target_config.get("use_log_transform", False)
    y_transformed, was_log_transformed = apply_target_transform(y, use_log_transform)
    
    # Find date column
    date_col = date_column
    if date_col is None:
        for col_name in DATE_COLUMN_CANDIDATES:
            if col_name in X.columns:
                date_col = col_name
                break
    
    # Enforce time-aware CV for forecasting: raise error if no date column found
    if not date_col or date_col not in X.columns:
        raise MissingDateColumnError(
            searched_columns=list(DATE_COLUMN_CANDIDATES),
            available_columns=X.columns.tolist()
        )
    
    # Exclude date column from features (it's only used for CV splitting)
    cat_cols = [c for c in cat_cols if c != date_col]
    num_cols = [c for c in num_cols if c != date_col]
    X_features = X.drop(columns=[date_col])
    
    # Create CV splitter with time-based splitting
    n_splits = cv_config.get("n_splits", 5)
    cv = TimeSeriesCVSplitter(n_splits=n_splits, date_column=date_col)
    
    # Cross-validation loop
    fold_scores = []
    model_type = model_config.get("type", "random_forest")
    random_state = model_config.get("random_state", 42)
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        # Use X_features (without date column) for model training/prediction
        Xtr = X_features.iloc[train_idx] if hasattr(X_features, "iloc") else X_features[train_idx]
        Xte = X_features.iloc[test_idx] if hasattr(X_features, "iloc") else X_features[test_idx]
        ytr = y_transformed.iloc[train_idx] if hasattr(y_transformed, "iloc") else y_transformed[train_idx]
        yte = y_transformed.iloc[test_idx] if hasattr(y_transformed, "iloc") else y_transformed[test_idx]
        
        # Create and train model
        pipe = create_model_pipeline(cat_cols, num_cols, model_type, config, random_state)
        pipe.fit(Xtr, ytr)
        
        # Predict and evaluate
        preds_transformed = pipe.predict(Xte)
        preds = inverse_target_transform(preds_transformed, was_log_transformed)
        yte_original = inverse_target_transform(np.array(yte), was_log_transformed)
        
        fold_metrics = {
            "fold": fold_idx + 1,
            "mae": float(mean_absolute_error(yte_original, preds)),
            "rmse": float(np.sqrt(mean_squared_error(yte_original, preds))),
            "r2": float(r2_score(yte_original, preds)),
            "n_train": len(train_idx),
            "n_test": len(test_idx)
        }
        fold_scores.append(fold_metrics)
        logger.info(f"Fold {fold_idx + 1}: MAE={fold_metrics['mae']:.2f}, R²={fold_metrics['r2']:.3f}")
    
    # Aggregate CV scores
    cv_results = {
        "folds": fold_scores,
        "mean_mae": float(np.mean([f["mae"] for f in fold_scores])),
        "std_mae": float(np.std([f["mae"] for f in fold_scores])),
        "mean_r2": float(np.mean([f["r2"] for f in fold_scores])),
        "std_r2": float(np.std([f["r2"] for f in fold_scores])),
        "n_folds": n_splits,
        "cv_type": cv_config.get("type", "time_series")
    }
    
    # Save CV results
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    cv_path = METRICS_DIR / "cv_results.json"
    with open(cv_path, "w") as f:
        json.dump(cv_results, f, indent=2)
    
    logger.info(f"CV Results: Mean MAE={cv_results['mean_mae']:.2f} ± {cv_results['std_mae']:.2f}")
    
    return cv_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train title demand forecasting model")
    parser.add_argument("--config", type=str, help="Path to ML config YAML file")
    parser.add_argument("--cv-only", action="store_true", help="Run cross-validation only")
    parser.add_argument("--output", type=str, help="Output path for model")
    
    args = parser.parse_args()
    
    config_path = Path(args.config) if args.config else None
    
    if args.cv_only:
        results = train_with_cross_validation(config_path)
        print(f"\nCV Results: Mean MAE={results['mean_mae']:.2f}, Mean R²={results['mean_r2']:.3f}")
    else:
        output_path = Path(args.output) if args.output else None
        results = train_baseline_model(save_path=output_path, config_path=config_path)
        print(f"\nModel saved to: {results['model_path']}")
        print(f"Metrics: MAE={results['metrics']['mae']:.2f}, R²={results['metrics']['r2']:.3f}")
