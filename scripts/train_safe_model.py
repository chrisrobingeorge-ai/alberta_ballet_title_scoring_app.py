#!/usr/bin/env python3
"""
Train Safe Model Script - PRIMARY RECOMMENDED PIPELINE

This script trains a leak-free XGBoost/LightGBM model on the modelling dataset.
It includes:
- Safety assertions to prevent training on current-run ticket columns
- Time-aware cross-validation
- Feature importance (permutation and SHAP if available)
- Model artifact saving with metadata

**This is the canonical, recommended ML path for Alberta Ballet ticket prediction.**

The safe modelling dataset pipeline consists of:
1. ``python scripts/build_modelling_dataset.py`` - Build leak-free dataset
2. ``python scripts/train_safe_model.py --tune`` (this script) - Train model with time-aware CV
3. ``python scripts/backtest_timeaware.py`` - Evaluate prediction methods

**Note:** The legacy baseline pipeline (ml/dataset.py + ml/training.py) is deprecated
and should not be used for production. It has known leakage risks.

Features Used (from modelling_dataset.csv)
------------------------------------------

**Baseline Signals (forecast-time safe):**
- wiki, trends, youtube, spotify

**Categorical Features:**
- category, gender

**Lagged Historical Features (from prior seasons - safe):**
- prior_total_tickets, prior_run_count, ticket_median_prior

**Remount Features (derived from prior runs - safe):**
- years_since_last_run, is_remount_recent, is_remount_medium, run_count_prior

**Seasonality & Date Features (based on planned timing - safe):**
- month_of_opening, holiday_flag
- opening_year, opening_month, opening_day_of_week, opening_week_of_year
- opening_quarter, opening_season
- opening_is_winter, opening_is_spring, opening_is_summer, opening_is_autumn
- opening_is_holiday_season, opening_is_weekend, run_duration_days

**External Economic Features (macro context known at forecast time - safe):**
- consumer_confidence_prairies (Nanos consumer confidence for Prairies region)
- energy_index (BoC commodity price index for energy)
- inflation_adjustment_factor (CPI-based inflation adjustment)
- city_median_household_income (Census-based household income)

**Audience Analytics Features (derived from historical category engagement - safe):**
- aud__engagement_factor (Live Analytics category engagement factor)

**Research Features (donor research data by year - safe):**
- res__arts_share_giving (Nanos arts donor research share of giving)

Forbidden Features (data leakage prevention)
--------------------------------------------
The following patterns are FORBIDDEN and will cause training to abort:
- ``single_tickets`` or ``single tickets`` (current-run ticket sales)
- ``total_tickets`` or ``total tickets`` (current-run totals)
- ``yourmodel_*`` (external model predictions)
- ``*_tickets_calgary``, ``*_tickets_edmonton`` (city-level current-run sales)

Usage:
    python scripts/train_safe_model.py [options]

Outputs:
    - models/model_xgb_remount_postcovid.joblib: Trained pipeline
    - models/model_xgb_remount_postcovid.json: Training metadata
    - results/feature_importances.csv: Feature importance scores
    - results/shap/ (if --save-shap): SHAP analysis outputs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

# Suppress warnings during training
warnings.filterwarnings("ignore", category=UserWarning)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Core ML imports
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GroupKFold, TimeSeriesSplit, cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Import version metadata utilities from ml.training
from ml.training import get_git_commit_hash, get_file_hash, get_dataset_shape, MissingDateColumnError

# Import data quality utilities
from data.quality import check_feature_ranges, DataQualityWarning

# XGBoost (required)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available")

# LightGBM (optional)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# SHAP (optional)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ============================================================================
# Safety checks
# ============================================================================

FORBIDDEN_FEATURE_PATTERNS = [
    "single_tickets",
    "single tickets",
    "total_tickets",
    "total tickets",
    "total_single_tickets",
    "total single tickets",
    "yourmodel_",
    "_tickets_-_",
    "_tickets_calgary",
    "_tickets_edmonton",
]

ALLOWED_COLUMNS = {
    "prior_total_tickets",
    "mean_last_3_seasons",
    "ticket_median_prior",
    "ticket_mean_prior",
    "ticket_std_prior",
    "years_since_last_ticket",
    "ticket_index_deseason",
}


def is_forbidden_feature(col_name: str) -> bool:
    """Check if column name matches forbidden patterns."""
    col_lower = col_name.lower().strip()
    if col_lower in ALLOWED_COLUMNS:
        return False
    for pattern in FORBIDDEN_FEATURE_PATTERNS:
        if pattern in col_lower:
            return True
    return False


def assert_safe_features(feature_cols: List[str]) -> None:
    """Assert that no forbidden columns are in the feature set."""
    forbidden = [col for col in feature_cols if is_forbidden_feature(col)]
    if forbidden:
        raise AssertionError(
            f"DATA LEAKAGE DETECTED!\n"
            f"Forbidden current-run ticket columns found in features:\n"
            f"  {forbidden}\n"
            f"Training aborted. Fix the dataset before retraining."
        )


# ============================================================================
# Training functions
# ============================================================================

def load_dataset(path: str = "data/modelling_dataset.csv") -> pd.DataFrame:
    """Load the modelling dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Modelling dataset not found at {path}\n"
            f"Run scripts/build_modelling_dataset.py first to create it."
        )
    return pd.read_csv(path)


def prepare_features(
    df: pd.DataFrame,
    target_col: str = "target_ticket_median",
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Prepare features and target for training.
    
    Returns:
        Tuple of (X, y, numeric_cols, categorical_cols)
    """
    exclude_cols = exclude_cols or []
    
    # Default columns to exclude from features
    default_exclude = {
        "title", "canonical_title", target_col,
        "show_title", "show_title_id"
    }
    exclude_set = default_exclude | set(exclude_cols)
    
    # Get feature columns
    feature_cols = [c for c in df.columns if c not in exclude_set]
    
    # Safety check
    assert_safe_features(feature_cols)
    
    # Split by dtype
    numeric_cols = []
    categorical_cols = []
    
    for col in feature_cols:
        if df[col].dtype in ["object", "category", "bool"]:
            categorical_cols.append(col)
        elif df[col].dtype in ["int64", "int32", "float64", "float32"]:
            numeric_cols.append(col)
    
    # Prepare X and y
    X = df[feature_cols].copy()
    
    if target_col in df.columns:
        y = df[target_col].copy()
        # Log transform target for better predictions
        y = np.log1p(y.clip(lower=0))
    else:
        # No target - return for inference
        y = None
    
    return X, y, numeric_cols, categorical_cols


def build_preprocessing_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> ColumnTransformer:
    """Build sklearn preprocessing pipeline."""
    
    transformers = []
    
    if numeric_cols:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", numeric_pipeline, numeric_cols))
    
    if categorical_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            # Use sparse_output (sklearn 1.2+) with fallback to sparse for older versions
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipeline, categorical_cols))
    
    return ColumnTransformer(transformers=transformers, remainder="passthrough")


def get_model(
    model_type: str = "xgboost",
    tune: bool = False,
    seed: int = 42
):
    """Get the model instance."""
    
    if model_type == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        if tune:
            # Light tuning - would use GridSearchCV in production
            return xgb.XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                random_state=seed,
                n_jobs=-1,
                verbosity=0
            )
        else:
            return xgb.XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=seed,
                n_jobs=-1,
                verbosity=0
            )
    
    elif model_type == "lightgbm":
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")
        
        if tune:
            return lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                n_jobs=-1,
                verbose=-1
            )
        else:
            return lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=seed,
                n_jobs=-1,
                verbose=-1
            )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_cv_splitter(
    df: pd.DataFrame,
    n_splits: int = 5,
    group_col: Optional[str] = None
):
    """Get appropriate cross-validation splitter.
    
    Enforces time-aware splitting to prevent future data leakage:
    1. If date column found, use TimeSeriesCVSplitter (strictest)
    2. If season/year column found, use TimeSeriesSplit
    3. If group column found, use GroupKFold (weaker guarantee, with warning)
    4. Raise MissingDateColumnError - NO random split fallback allowed
    
    Raises:
        MissingDateColumnError: If no date/time column is found for time-aware CV.
            Forecasting models require temporal ordering to prevent future data leakage.
    """
    # First, try to find a date column for strict chronological splitting
    date_cols = ["end_date", "start_date", "date", "opening_date", "performance_date"]
    for col in date_cols:
        if col in df.columns:
            try:
                from ml.time_splits import TimeSeriesCVSplitter
                return TimeSeriesCVSplitter(n_splits=n_splits, date_column=col)
            except ImportError:
                pass  # Fall through to TimeSeriesSplit
    
    # Try time-aware if we have a season/year column
    time_cols = ["season", "ref_year", "year", "fiscal_year"]
    for col in time_cols:
        if col in df.columns:
            # Use TimeSeriesSplit
            return TimeSeriesSplit(n_splits=n_splits)
    
    # Try group-based if we have a grouping column
    if group_col and group_col in df.columns:
        import warnings
        warnings.warn(
            f"Using GroupKFold with group column '{group_col}'. "
            "This does not guarantee chronological ordering within groups. "
            "Consider adding a date column for strict time-aware splitting.",
            UserWarning
        )
        return GroupKFold(n_splits=n_splits)
    
    # Raise error instead of falling back to random splits
    # Random splits are NOT allowed for forecasting models due to data leakage risk
    raise MissingDateColumnError(
        searched_columns=list(date_cols) + list(time_cols),
        available_columns=df.columns.tolist()
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    # Inverse log transform for interpretable metrics
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)
    
    return {
        "mae": float(mean_absolute_error(y_true_orig, y_pred_orig)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))),
        "r2": float(r2_score(y_true_orig, y_pred_orig)),
        "mae_log": float(mean_absolute_error(y_true, y_pred)),
        "rmse_log": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def train_model(
    dataset_path: str = "data/modelling_dataset.csv",
    model_type: str = "xgboost",
    target_col: str = "target_ticket_median",
    model_output_path: str = "models/model_xgb_remount_postcovid.joblib",
    metadata_output_path: str = "models/model_xgb_remount_postcovid.json",
    importance_output_path: str = "results/feature_importances.csv",
    tune: bool = False,
    save_shap: bool = False,
    seed: int = 42,
    verbose: bool = True,
    date_column: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train the model pipeline.
    
    Returns:
        Dictionary with training results and metadata
    """
    results: Dict[str, Any] = {
        "success": False,
        "training_date": datetime.now().isoformat(),
        "seed": seed,
        "model_type": model_type,
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("Training Safe Model")
        print("=" * 60)
    
    # 1. Load data
    if verbose:
        print("\n1. Loading dataset...")
    
    df = load_dataset(dataset_path)
    if verbose:
        print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Filter to rows with valid target
    if target_col in df.columns:
        df = df[df[target_col].notna() & (df[target_col] > 0)]
        if verbose:
            print(f"   After filtering for valid target: {len(df)} rows")
    else:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    if len(df) < 10:
        raise ValueError(f"Not enough training data: {len(df)} rows (need at least 10)")
    
    # 1b. Data quality check - validate feature ranges before training
    if verbose:
        print("\n   Running data quality checks...")
    
    violations = check_feature_ranges(df)
    if violations:
        if verbose:
            print(f"   Warning: Found {len(violations)} features with out-of-range values")
            for feature_name, details in list(violations.items())[:5]:
                print(f"     - {feature_name}: {details['total_violations']} violations "
                      f"({details['violation_rate']:.1%})")
            if len(violations) > 5:
                print(f"     ... and {len(violations) - 5} more")
        results["data_quality_violations"] = violations
    else:
        if verbose:
            print("   Data quality check passed: all features within expected ranges")
        results["data_quality_violations"] = {}
    
    # 2. Prepare features
    if verbose:
        print("\n2. Preparing features...")
    
    X, y, numeric_cols, categorical_cols = prepare_features(df, target_col)
    
    if verbose:
        print(f"   Numeric features ({len(numeric_cols)}): {numeric_cols}")
        print(f"   Categorical features ({len(categorical_cols)}): {categorical_cols}")
    
    results["features"] = {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "total": len(numeric_cols) + len(categorical_cols)
    }
    
    # 3. Build pipeline
    if verbose:
        print("\n3. Building preprocessing pipeline...")
    
    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    model = get_model(model_type, tune, seed)
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    # 4. Cross-validation
    if verbose:
        print("\n4. Running cross-validation...")
    
    cv = get_cv_splitter(df, n_splits=min(5, len(df) // 2), date_column=date_column)
    
    # Manual CV for detailed metrics
    cv_scores = {"mae": [], "rmse": [], "r2": []}
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Clone pipeline for each fold
        fold_pipeline = Pipeline([
            ("preprocessor", build_preprocessing_pipeline(numeric_cols, categorical_cols)),
            ("model", get_model(model_type, tune, seed))
        ])
        fold_pipeline.fit(X_train, y_train)
        y_pred = fold_pipeline.predict(X_val)
        
        fold_metrics = compute_metrics(y_val.values, y_pred)
        cv_scores["mae"].append(fold_metrics["mae"])
        cv_scores["rmse"].append(fold_metrics["rmse"])
        cv_scores["r2"].append(fold_metrics["r2"])
        
        if verbose:
            print(f"   Fold {fold+1}: MAE={fold_metrics['mae']:.0f}, "
                  f"RMSE={fold_metrics['rmse']:.0f}, R²={fold_metrics['r2']:.3f}")
    
    cv_results = {
        "mae_mean": float(np.mean(cv_scores["mae"])),
        "mae_std": float(np.std(cv_scores["mae"])),
        "rmse_mean": float(np.mean(cv_scores["rmse"])),
        "rmse_std": float(np.std(cv_scores["rmse"])),
        "r2_mean": float(np.mean(cv_scores["r2"])),
        "r2_std": float(np.std(cv_scores["r2"])),
        "n_folds": len(cv_scores["mae"])
    }
    results["cv_metrics"] = cv_results
    
    if verbose:
        print(f"\n   CV Summary:")
        print(f"   MAE: {cv_results['mae_mean']:.0f} ± {cv_results['mae_std']:.0f}")
        print(f"   RMSE: {cv_results['rmse_mean']:.0f} ± {cv_results['rmse_std']:.0f}")
        print(f"   R²: {cv_results['r2_mean']:.3f} ± {cv_results['r2_std']:.3f}")
    
    # 5. Train final model on all data
    if verbose:
        print("\n5. Training final model on all data...")
    
    pipeline.fit(X, y)
    
    # Compute training metrics
    y_pred_train = pipeline.predict(X)
    train_metrics = compute_metrics(y.values, y_pred_train)
    results["train_metrics"] = train_metrics
    
    if verbose:
        print(f"   Training MAE: {train_metrics['mae']:.0f}")
        print(f"   Training R²: {train_metrics['r2']:.3f}")
    
    # 6. Compute feature importance
    if verbose:
        print("\n6. Computing feature importance...")
    
    # Get feature names after preprocessing
    try:
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    except Exception:
        feature_names = numeric_cols + categorical_cols
    
    # Get model feature importances
    model_instance = pipeline.named_steps["model"]
    if hasattr(model_instance, "feature_importances_"):
        importances = model_instance.feature_importances_
        
        importance_df = pd.DataFrame({
            "feature": feature_names[:len(importances)],
            "importance": importances
        }).sort_values("importance", ascending=False)
        
        if verbose:
            print("   Top 10 features:")
            for _, row in importance_df.head(10).iterrows():
                print(f"     {row['feature']}: {row['importance']:.4f}")
        
        # Save importance
        os.makedirs(os.path.dirname(importance_output_path) or ".", exist_ok=True)
        importance_df.to_csv(importance_output_path, index=False)
        results["feature_importances_path"] = importance_output_path
    
    # 7. SHAP analysis (optional)
    if save_shap:
        if verbose:
            print("\n7. Computing SHAP values...")
        
        if not SHAP_AVAILABLE:
            print("   Warning: SHAP not installed. Skipping SHAP analysis.")
        else:
            try:
                # Transform data for SHAP
                X_transformed = pipeline.named_steps["preprocessor"].transform(X)
                
                # Create explainer
                explainer = shap.TreeExplainer(model_instance)
                shap_values = explainer.shap_values(X_transformed)
                
                # Save outputs
                shap_dir = "results/shap"
                os.makedirs(shap_dir, exist_ok=True)
                
                # Save SHAP values
                shap_df = pd.DataFrame(
                    shap_values,
                    columns=feature_names[:shap_values.shape[1]]
                )
                shap_df.to_parquet(f"{shap_dir}/shap_values.parquet", index=False)
                
                # Save summary plot
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                
                shap.summary_plot(shap_values, X_transformed, 
                                 feature_names=list(feature_names[:shap_values.shape[1]]),
                                 show=False)
                plt.tight_layout()
                plt.savefig(f"{shap_dir}/shap_summary.png", dpi=150, bbox_inches="tight")
                plt.close()
                
                results["shap_outputs"] = shap_dir
                if verbose:
                    print(f"   Saved SHAP outputs to {shap_dir}/")
                
            except Exception as e:
                print(f"   Warning: SHAP analysis failed: {e}")
    
    # 8. Save model and metadata
    if verbose:
        print("\n8. Saving model and metadata...")
    
    os.makedirs(os.path.dirname(model_output_path) or ".", exist_ok=True)
    
    # Save pipeline
    joblib.dump(pipeline, model_output_path)
    results["model_path"] = model_output_path
    
    if verbose:
        print(f"   Saved model to {model_output_path}")
    
    # Save metadata
    metadata = {
        "training_date": results["training_date"],
        "model_type": model_type,
        "seed": seed,
        "tuned": tune,
        "n_samples": len(df),
        "features": results["features"],
        "cv_metrics": cv_results,
        "train_metrics": train_metrics,
        "target_column": target_col,
        "dataset_path": dataset_path,
        # Version metadata
        "git_commit_hash": get_git_commit_hash(),
        "data_file_hash": get_file_hash(dataset_path),
        "dataset_shape": get_dataset_shape(df),
    }
    
    with open(metadata_output_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    results["metadata_path"] = metadata_output_path
    
    if verbose:
        print(f"   Saved metadata to {metadata_output_path}")
    
    results["success"] = True
    
    if verbose:
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
    
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train leak-free ticket demand model"
    )
    parser.add_argument(
        "--dataset",
        default="data/modelling_dataset.csv",
        help="Path to modelling dataset CSV"
    )
    parser.add_argument(
        "--model-type",
        choices=["xgboost", "lightgbm"],
        default="xgboost",
        help="Model type to train"
    )
    parser.add_argument(
        "--target",
        default="target_ticket_median",
        help="Target column name"
    )
    parser.add_argument(
        "--output",
        default="models/model_xgb_remount_postcovid.joblib",
        help="Output path for model"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable light hyperparameter tuning"
    )
    parser.add_argument(
        "--save-shap",
        action="store_true",
        help="Compute and save SHAP explanations"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--date_column",
        type=str, default=None,
        help="Name of date column to use for time-aware CV (e.g., 'opening_date')")

    args = parser.parse_args()
    
    try:
        results = train_model(
            dataset_path=args.dataset,
            model_type=args.model_type,
            target_col=args.target,
            model_output_path=args.output,
            tune=args.tune,
            save_shap=args.save_shap,
            seed=args.seed,
            verbose=not args.quiet,
            date_column=args.date_column  # ✅ Add this line
        )
        
        if not results["success"]:
            sys.exit(1)
            
    except AssertionError as e:
        print(f"\nFATAL ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
