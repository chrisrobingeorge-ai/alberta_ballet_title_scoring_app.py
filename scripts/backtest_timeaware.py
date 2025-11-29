#!/usr/bin/env python3
"""
Time-Aware Backtesting Script

This script performs rolling/seasonal holdout evaluation to compare different
prediction methods:
1. Heuristic/composite score from the Streamlit app
2. k-NN similarity fallback
3. Baseline-only supervised model
4. Full supervised model with all features

Usage:
    python scripts/backtest_timeaware.py [options]

Outputs:
    - results/backtest_summary.json: Summary metrics per method
    - results/backtest_comparison.csv: Row-level predictions
    - results/plots/mae_by_method.png: MAE comparison chart
    - results/plots/mae_by_category.png: MAE by category chart
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

# Suppress warnings
warnings.filterwarnings("ignore")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Core imports
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Time-aware splitting
try:
    from ml.time_splits import (
        TimeSeriesCVSplitter,
        assert_chronological_split,
    )
    TIME_SPLITS_AVAILABLE = True
except ImportError:
    TIME_SPLITS_AVAILABLE = False

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# KNN fallback
try:
    from ml.knn_fallback import KNNFallback
    KNN_AVAILABLE = True
except ImportError:
    KNN_AVAILABLE = False


def load_modelling_dataset(path: str = "data/modelling_dataset.csv") -> pd.DataFrame:
    """Load the modelling dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics."""
    # Handle NaN predictions
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "n_samples": 0}
    
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "n_samples": int(len(y_true))
    }


def heuristic_prediction(row: pd.Series) -> float:
    """
    Compute heuristic/composite prediction based on baseline signals.
    
    This mimics the Streamlit app's composite scoring approach.
    """
    # Get baseline signals
    wiki = float(row.get("wiki", 50) or 50)
    trends = float(row.get("trends", 50) or 50)
    youtube = float(row.get("youtube", 50) or 50)
    spotify = float(row.get("spotify", 50) or 50)
    
    # Compute familiarity and motivation (simplified)
    familiarity = 0.55 * wiki + 0.30 * trends + 0.15 * spotify
    motivation = 0.45 * youtube + 0.25 * trends + 0.15 * spotify + 0.15 * wiki
    
    # Signal average
    signal_only = (familiarity + motivation) / 2
    
    # Use prior ticket median if available, else estimate from signal
    prior_median = float(row.get("ticket_median_prior", 0) or 0)
    
    if prior_median > 0:
        # Blend signal and history
        return 0.5 * prior_median + 0.5 * (signal_only * 50)  # Scale signal to ticket range
    else:
        # Pure signal-based estimate
        return signal_only * 50  # Rough scaling


def train_baseline_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series
) -> Pipeline:
    """Train a baseline model using only baseline signal features."""
    baseline_cols = ["wiki", "trends", "youtube", "spotify"]
    available_cols = [c for c in baseline_cols if c in X_train.columns]
    
    if not available_cols:
        return None
    
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    if XGBOOST_AVAILABLE:
        model = xgb.XGBRegressor(
            n_estimators=50, max_depth=2, random_state=42, verbosity=0
        )
    else:
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
    
    pipeline = Pipeline([
        ("pre", preprocessor),
        ("model", model)
    ])
    
    X_baseline = X_train[available_cols].copy()
    pipeline.fit(X_baseline, y_train)
    
    return pipeline


def train_full_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series
) -> Pipeline:
    """Train a full model using all available features."""
    # Identify feature types
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "bool"]).columns.tolist()
    
    transformers = []
    
    if numeric_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipeline, numeric_cols))
    
    if categorical_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipeline, categorical_cols))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    
    if XGBOOST_AVAILABLE:
        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=3, random_state=42, verbosity=0
        )
    else:
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
    
    pipeline = Pipeline([
        ("pre", preprocessor),
        ("model", model)
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline


def run_backtest(
    dataset_path: str = "data/modelling_dataset.csv",
    target_col: str = "target_ticket_median",
    n_folds: int = 5,
    output_dir: str = "results",
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run time-aware backtesting.
    
    Returns:
        Dictionary with backtest results
    """
    results = {
        "created_at": datetime.now().isoformat(),
        "seed": seed,
        "n_folds": n_folds,
        "methods": {},
        "by_category": {},
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("Time-Aware Backtesting")
        print("=" * 60)
    
    # 1. Load data
    if verbose:
        print("\n1. Loading dataset...")
    
    df = load_modelling_dataset(dataset_path)
    
    # Filter to rows with valid target
    df = df[df[target_col].notna() & (df[target_col] > 0)].copy()
    
    if verbose:
        print(f"   Loaded {len(df)} rows with valid target")
    
    if len(df) < 10:
        raise ValueError(f"Not enough data for backtesting: {len(df)} rows")
    
    # 2. Prepare features
    exclude_cols = {"title", "canonical_title", target_col, "show_title", "show_title_id"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Keep category for stratified analysis
    category_col = "category" if "category" in df.columns else None
    categories = df[category_col].values if category_col else None
    
    # 3. Initialize storage for predictions
    all_predictions = {
        "fold_id": [],
        "actual": [],
        "heuristic": [],
        "knn": [],
        "baseline_model": [],
        "full_model": [],
    }
    if category_col:
        all_predictions["category"] = []
    
    # 4. Run backtesting
    if verbose:
        print("\n2. Running cross-validation...")
    
    # Detect date column for time-aware splitting
    date_col = None
    for col_name in ["end_date", "start_date", "date", "opening_date", "performance_date"]:
        if col_name in df.columns:
            date_col = col_name
            break
    
    # Use time-aware CV if date column available, else fall back to GroupKFold/KFold
    if date_col and TIME_SPLITS_AVAILABLE:
        if verbose:
            print(f"   Using time-aware CV with date column: {date_col}")
        cv = TimeSeriesCVSplitter(n_splits=n_folds, date_column=date_col)
        # Add date column to X temporarily for splitting
        X_with_date = X.copy()
        X_with_date[date_col] = df[date_col].values
        cv_iter = cv.split(X_with_date)
    elif "canonical_title" in df.columns:
        if verbose:
            print("   Using GroupKFold by title (no date column found)")
            print("   WARNING: GroupKFold may allow future data to leak into training!")
        groups = df["canonical_title"].values
        cv = GroupKFold(n_splits=min(n_folds, len(df["canonical_title"].unique())))
        cv_iter = cv.split(X, y, groups)
    else:
        if verbose:
            print("   Using KFold with shuffle (no date or group column found)")
            print("   WARNING: Random KFold may allow future data to leak into training!")
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        cv_iter = cv.split(X, y)
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv_iter):
        if verbose:
            print(f"\n   Fold {fold_idx + 1}/{n_folds}...")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Heuristic predictions
        heuristic_preds = X_test.apply(heuristic_prediction, axis=1).values
        
        # KNN predictions
        if KNN_AVAILABLE:
            # Build KNN index from training data
            train_df = pd.concat([X_train, y_train], axis=1)
            knn = KNNFallback(k=5, metric="cosine")
            try:
                knn.build_index(train_df, outcome_col=target_col)
                knn_preds = X_test.apply(lambda row: knn.predict(row), axis=1).values
            except Exception as e:
                if verbose:
                    print(f"      KNN failed: {e}")
                knn_preds = np.full(len(X_test), np.nan)
        else:
            knn_preds = np.full(len(X_test), np.nan)
        
        # Baseline model (signals only)
        baseline_cols = ["wiki", "trends", "youtube", "spotify"]
        baseline_cols = [c for c in baseline_cols if c in X_train.columns]
        if baseline_cols:
            try:
                baseline_model = train_baseline_model(X_train, y_train)
                baseline_preds = baseline_model.predict(X_test[baseline_cols])
            except Exception as e:
                if verbose:
                    print(f"      Baseline model failed: {e}")
                baseline_preds = np.full(len(X_test), np.nan)
        else:
            baseline_preds = np.full(len(X_test), np.nan)
        
        # Full model
        try:
            full_model = train_full_model(X_train, y_train)
            full_preds = full_model.predict(X_test)
        except Exception as e:
            if verbose:
                print(f"      Full model failed: {e}")
            full_preds = np.full(len(X_test), np.nan)
        
        # Store predictions
        for i, idx in enumerate(test_idx):
            all_predictions["fold_id"].append(fold_idx)
            all_predictions["actual"].append(float(y_test.iloc[i]))
            all_predictions["heuristic"].append(float(heuristic_preds[i]))
            all_predictions["knn"].append(float(knn_preds[i]) if not np.isnan(knn_preds[i]) else None)
            all_predictions["baseline_model"].append(float(baseline_preds[i]) if not np.isnan(baseline_preds[i]) else None)
            all_predictions["full_model"].append(float(full_preds[i]) if not np.isnan(full_preds[i]) else None)
            if category_col:
                all_predictions["category"].append(categories[idx])
        
        if verbose:
            # Print fold metrics for full model
            fold_metrics = compute_metrics(
                np.array(y_test), np.array(full_preds)
            )
            print(f"      Full model MAE: {fold_metrics['mae']:.0f}")
    
    # 5. Compute overall metrics
    if verbose:
        print("\n3. Computing overall metrics...")
    
    pred_df = pd.DataFrame(all_predictions)
    
    methods = ["heuristic", "knn", "baseline_model", "full_model"]
    method_metrics = {}
    
    for method in methods:
        if method not in pred_df.columns:
            continue
        
        # Filter valid predictions
        mask = pred_df[method].notna()
        if mask.sum() == 0:
            continue
        
        y_true = pred_df.loc[mask, "actual"].values
        y_pred = pred_df.loc[mask, method].values.astype(float)
        
        metrics = compute_metrics(y_true, y_pred)
        method_metrics[method] = metrics
        
        if verbose:
            print(f"   {method}: MAE={metrics['mae']:.0f}, "
                  f"RMSE={metrics['rmse']:.0f}, R²={metrics['r2']:.3f}")
    
    results["methods"] = method_metrics
    
    # 6. Compute metrics by category
    if category_col and category_col in pred_df.columns:
        if verbose:
            print("\n4. Computing metrics by category...")
        
        by_category = {}
        for cat in pred_df[category_col].dropna().unique():
            cat_mask = pred_df[category_col] == cat
            cat_metrics = {}
            
            for method in methods:
                if method not in pred_df.columns:
                    continue
                
                valid_mask = cat_mask & pred_df[method].notna()
                if valid_mask.sum() < 2:
                    continue
                
                y_true = pred_df.loc[valid_mask, "actual"].values
                y_pred = pred_df.loc[valid_mask, method].values.astype(float)
                
                cat_metrics[method] = compute_metrics(y_true, y_pred)
            
            if cat_metrics:
                by_category[cat] = cat_metrics
        
        results["by_category"] = by_category
    
    # 7. Save outputs
    if verbose:
        print("\n5. Saving outputs...")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)
    
    # Save summary JSON
    summary_path = f"{output_dir}/backtest_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    if verbose:
        print(f"   Saved summary to {summary_path}")
    
    # Save comparison CSV
    comparison_path = f"{output_dir}/backtest_comparison.csv"
    pred_df.to_csv(comparison_path, index=False)
    if verbose:
        print(f"   Saved comparisons to {comparison_path}")
    
    # 8. Generate plots
    if verbose:
        print("\n6. Generating plots...")
    
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        # MAE by method bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        method_names = list(method_metrics.keys())
        maes = [method_metrics[m]["mae"] for m in method_names]
        
        bars = ax.bar(method_names, maes, color=["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"])
        ax.set_ylabel("Mean Absolute Error (tickets)")
        ax.set_title("Backtest: MAE by Prediction Method")
        ax.set_ylim(0, max(maes) * 1.2 if maes else 1)
        
        # Add value labels on bars
        for bar, mae in zip(bars, maes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(maes)*0.02,
                   f"{mae:.0f}", ha="center", va="bottom", fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/mae_by_method.png", dpi=150)
        plt.close()
        
        if verbose:
            print(f"   Saved mae_by_method.png")
        
        # MAE by category (if available)
        if results["by_category"]:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            cats = list(results["by_category"].keys())
            x = np.arange(len(cats))
            width = 0.2
            
            for i, method in enumerate(["heuristic", "full_model"]):
                maes = []
                for cat in cats:
                    if method in results["by_category"].get(cat, {}):
                        maes.append(results["by_category"][cat][method]["mae"])
                    else:
                        maes.append(0)
                ax.bar(x + i*width, maes, width, label=method)
            
            ax.set_xlabel("Category")
            ax.set_ylabel("MAE")
            ax.set_title("Backtest: MAE by Category")
            ax.set_xticks(x + width/2)
            ax.set_xticklabels(cats, rotation=45, ha="right")
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/plots/mae_by_category.png", dpi=150)
            plt.close()
            
            if verbose:
                print(f"   Saved mae_by_category.png")
        
    except Exception as e:
        if verbose:
            print(f"   Warning: Could not generate plots: {e}")
    
    results["output_files"] = {
        "summary": summary_path,
        "comparison": comparison_path,
        "plots_dir": f"{output_dir}/plots"
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("Backtest Complete!")
        print("=" * 60)
    
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run time-aware backtesting for ticket demand models"
    )
    parser.add_argument(
        "--dataset",
        default="data/modelling_dataset.csv",
        help="Path to modelling dataset"
    )
    parser.add_argument(
        "--target",
        default="target_ticket_median",
        help="Target column name"
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of CV folds"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output"
    )
    
    args = parser.parse_args()
    
    try:
        run_backtest(
            dataset_path=args.dataset,
            target_col=args.target,
            n_folds=args.folds,
            output_dir=args.output_dir,
            seed=args.seed,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
