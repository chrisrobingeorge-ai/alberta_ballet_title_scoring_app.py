#!/usr/bin/env python3
"""
Calibration Script

This script fits calibration parameters to adjust model predictions to better
match actual outcomes. Supports global, per-category, and per-remount-bin
calibration modes.

Usage:
    python scripts/calibrate_predictions.py [options]

Outputs:
    - models/calibration.json: Calibration parameters
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def fit_linear_calibration(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Fit linear calibration: actual ≈ alpha * pred + beta
    
    Returns:
        Dict with alpha, beta, r2, n_samples
    """
    # Filter valid pairs
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) < 3:
        return {"alpha": 1.0, "beta": 0.0, "r2": np.nan, "n_samples": len(y_true)}
    
    # Fit linear regression: y_true = alpha * y_pred + beta
    coeffs = np.polyfit(y_pred, y_true, 1)
    alpha, beta = float(coeffs[0]), float(coeffs[1])
    
    # Compute R²
    y_calibrated = alpha * y_pred + beta
    ss_res = np.sum((y_true - y_calibrated) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    return {
        "alpha": alpha,
        "beta": beta,
        "r2": float(r2),
        "n_samples": int(len(y_true))
    }


def calibrate_predictions(
    predictions: np.ndarray,
    alpha: float,
    beta: float
) -> np.ndarray:
    """Apply linear calibration to predictions."""
    return alpha * np.array(predictions) + beta


def fit_calibration(
    backtest_path: str = "results/backtest_comparison.csv",
    mode: str = "global",
    prediction_col: str = "full_model",
    actual_col: str = "actual",
    output_path: str = "models/calibration.json",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Fit calibration parameters from backtest results.
    
    Args:
        backtest_path: Path to backtest comparison CSV
        mode: Calibration mode ('global', 'per_category', 'by_remount_bin')
        prediction_col: Column with predictions to calibrate
        actual_col: Column with actual values
        output_path: Path to save calibration parameters
        verbose: Print progress
        
    Returns:
        Calibration parameters dictionary
    """
    results = {
        "created_at": datetime.now().isoformat(),
        "mode": mode,
        "prediction_column": prediction_col,
        "parameters": {}
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("Fitting Calibration Parameters")
        print("=" * 60)
    
    # Load backtest results
    if not os.path.exists(backtest_path):
        raise FileNotFoundError(f"Backtest results not found: {backtest_path}")
    
    df = pd.read_csv(backtest_path)
    
    if prediction_col not in df.columns:
        raise ValueError(f"Prediction column '{prediction_col}' not in backtest results")
    if actual_col not in df.columns:
        raise ValueError(f"Actual column '{actual_col}' not in backtest results")
    
    # Filter to valid rows
    df = df[df[prediction_col].notna() & df[actual_col].notna()].copy()
    
    if verbose:
        print(f"\nLoaded {len(df)} valid predictions from backtest")
    
    if mode == "global":
        # Single calibration for all predictions
        y_true = df[actual_col].values
        y_pred = df[prediction_col].values
        
        params = fit_linear_calibration(y_true, y_pred)
        results["parameters"]["global"] = params
        
        if verbose:
            print(f"\nGlobal calibration:")
            print(f"  alpha = {params['alpha']:.4f}")
            print(f"  beta = {params['beta']:.2f}")
            print(f"  R² = {params['r2']:.4f}")
            print(f"  n = {params['n_samples']}")
    
    elif mode == "per_category":
        # Calibration per category
        if "category" not in df.columns:
            raise ValueError("Category column required for per_category mode")
        
        category_params = {}
        
        for cat in df["category"].unique():
            cat_df = df[df["category"] == cat]
            y_true = cat_df[actual_col].values
            y_pred = cat_df[prediction_col].values
            
            params = fit_linear_calibration(y_true, y_pred)
            category_params[cat] = params
            
            if verbose:
                print(f"\n{cat}: alpha={params['alpha']:.3f}, "
                      f"beta={params['beta']:.1f}, n={params['n_samples']}")
        
        results["parameters"] = category_params
    
    elif mode == "by_remount_bin":
        # Calibration by remount timing bins
        if "years_since_last_run" not in df.columns:
            if verbose:
                print("Warning: years_since_last_run not available, using global calibration")
            mode = "global"
            y_true = df[actual_col].values
            y_pred = df[prediction_col].values
            results["parameters"]["global"] = fit_linear_calibration(y_true, y_pred)
        else:
            # Create bins
            df["remount_bin"] = pd.cut(
                df["years_since_last_run"],
                bins=[-np.inf, 2, 4, np.inf],
                labels=["recent_0-2y", "medium_2-4y", "old_4y+"]
            )
            
            bin_params = {}
            for bin_name in df["remount_bin"].unique():
                if pd.isna(bin_name):
                    continue
                    
                bin_df = df[df["remount_bin"] == bin_name]
                y_true = bin_df[actual_col].values
                y_pred = bin_df[prediction_col].values
                
                params = fit_linear_calibration(y_true, y_pred)
                bin_params[str(bin_name)] = params
                
                if verbose:
                    print(f"\n{bin_name}: alpha={params['alpha']:.3f}, "
                          f"beta={params['beta']:.1f}, n={params['n_samples']}")
            
            results["parameters"] = bin_params
    
    else:
        raise ValueError(f"Unknown calibration mode: {mode}")
    
    # Save calibration parameters
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    if verbose:
        print(f"\n✓ Saved calibration to {output_path}")
    
    return results


def apply_calibration_to_csv(
    input_csv: str,
    output_csv: str,
    calibration_path: str = "models/calibration.json",
    prediction_col: str = "predicted",
    category_col: Optional[str] = "category",
    remount_col: Optional[str] = "years_since_last_run",
    verbose: bool = True
) -> None:
    """
    Apply calibration to a predictions CSV.
    
    Args:
        input_csv: Path to input predictions CSV
        output_csv: Path to output calibrated CSV
        calibration_path: Path to calibration parameters JSON
        prediction_col: Column with predictions to calibrate
        category_col: Category column (for per_category mode)
        remount_col: Remount years column (for by_remount_bin mode)
        verbose: Print progress
    """
    # Load calibration
    with open(calibration_path, "r") as f:
        calibration = json.load(f)
    
    mode = calibration.get("mode", "global")
    params = calibration.get("parameters", {})
    
    # Load predictions
    df = pd.read_csv(input_csv)
    
    if prediction_col not in df.columns:
        raise ValueError(f"Prediction column '{prediction_col}' not found")
    
    predictions = df[prediction_col].values
    calibrated = np.zeros_like(predictions, dtype=float)
    
    if mode == "global":
        p = params.get("global", {"alpha": 1.0, "beta": 0.0})
        calibrated = calibrate_predictions(predictions, p["alpha"], p["beta"])
    
    elif mode == "per_category":
        if category_col not in df.columns:
            raise ValueError(f"Category column '{category_col}' required for per_category mode")
        
        for cat in df[category_col].unique():
            mask = df[category_col] == cat
            p = params.get(cat, {"alpha": 1.0, "beta": 0.0})
            calibrated[mask] = calibrate_predictions(
                predictions[mask], p["alpha"], p["beta"]
            )
    
    elif mode == "by_remount_bin":
        if remount_col not in df.columns:
            # Fall back to global
            p = list(params.values())[0] if params else {"alpha": 1.0, "beta": 0.0}
            calibrated = calibrate_predictions(predictions, p["alpha"], p["beta"])
        else:
            df["_remount_bin"] = pd.cut(
                df[remount_col],
                bins=[-np.inf, 2, 4, np.inf],
                labels=["recent_0-2y", "medium_2-4y", "old_4y+"]
            )
            
            for bin_name in df["_remount_bin"].unique():
                if pd.isna(bin_name):
                    continue
                mask = df["_remount_bin"] == bin_name
                p = params.get(str(bin_name), {"alpha": 1.0, "beta": 0.0})
                calibrated[mask] = calibrate_predictions(
                    predictions[mask], p["alpha"], p["beta"]
                )
            
            df = df.drop(columns=["_remount_bin"])
    
    # Add calibrated column
    df["calibrated_prediction"] = calibrated
    
    # Save
    df.to_csv(output_csv, index=False)
    
    if verbose:
        print(f"✓ Saved calibrated predictions to {output_csv}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fit and apply calibration to predictions"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Fit command
    fit_parser = subparsers.add_parser("fit", help="Fit calibration parameters")
    fit_parser.add_argument(
        "--backtest",
        default="results/backtest_comparison.csv",
        help="Path to backtest comparison CSV"
    )
    fit_parser.add_argument(
        "--mode",
        choices=["global", "per_category", "by_remount_bin"],
        default="global",
        help="Calibration mode"
    )
    fit_parser.add_argument(
        "--prediction-col",
        default="full_model",
        help="Prediction column name"
    )
    fit_parser.add_argument(
        "--output",
        default="models/calibration.json",
        help="Output path for calibration parameters"
    )
    
    # Apply command
    apply_parser = subparsers.add_parser("apply", help="Apply calibration to predictions")
    apply_parser.add_argument(
        "--input",
        required=True,
        help="Input predictions CSV"
    )
    apply_parser.add_argument(
        "--output",
        required=True,
        help="Output calibrated CSV"
    )
    apply_parser.add_argument(
        "--calibration",
        default="models/calibration.json",
        help="Path to calibration parameters"
    )
    apply_parser.add_argument(
        "--prediction-col",
        default="predicted",
        help="Prediction column name"
    )
    
    args = parser.parse_args()
    
    if args.command == "fit":
        fit_calibration(
            backtest_path=args.backtest,
            mode=args.mode,
            prediction_col=args.prediction_col,
            output_path=args.output
        )
    
    elif args.command == "apply":
        apply_calibration_to_csv(
            input_csv=args.input,
            output_csv=args.output,
            calibration_path=args.calibration,
            prediction_col=args.prediction_col
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
