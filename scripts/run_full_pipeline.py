#!/usr/bin/env python3
"""
Full Modelling Pipeline Script

This script runs the complete ML pipeline end-to-end:
1. Build the modelling dataset (leak-free)
2. Run time-aware backtesting
3. Train the final safe model

All outputs are organized in a timestamped directory under results/.

Usage:
    python scripts/run_full_pipeline.py [options]

    # Run with default options
    python scripts/run_full_pipeline.py

    # Enable hyperparameter tuning
    python scripts/run_full_pipeline.py --tune

    # Include SHAP explanations
    python scripts/run_full_pipeline.py --save-shap

    # Quiet mode (less output)
    python scripts/run_full_pipeline.py --quiet

Outputs:
    results/<timestamp>/
    ├── modelling_dataset.csv          # Leak-free training data
    ├── modelling_dataset_report.json  # Dataset diagnostics
    ├── backtest_summary.json          # Method comparison metrics
    ├── backtest_comparison.csv        # Row-level predictions
    ├── feature_importances.csv        # Feature importance scores
    ├── plots/
    │   ├── mae_by_method.png          # MAE comparison chart
    │   └── mae_by_category.png        # Category breakdown
    ├── shap/ (if --save-shap)
    │   ├── shap_summary.png           # SHAP summary plot
    │   └── shap_values.parquet        # Raw SHAP values
    └── pipeline_summary.json          # Overall pipeline run summary

    models/
    ├── model_xgb_remount_postcovid.joblib  # Trained model pipeline
    └── model_xgb_remount_postcovid.json    # Model metadata
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_full_pipeline(
    output_base_dir: str = "results",
    history_path: str = "data/productions/history_city_sales.csv",
    baselines_path: str = "data/productions/baselines.csv",
    past_runs_path: str = "data/productions/past_runs.csv",
    tune: bool = False,
    save_shap: bool = False,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the complete modelling pipeline.

    This function orchestrates:
    1. Building the leak-free modelling dataset
    2. Running time-aware backtesting
    3. Training the final safe model

    Args:
        output_base_dir: Base directory for outputs (default: results/)
        history_path: Path to historical sales CSV
        baselines_path: Path to baselines CSV
        past_runs_path: Path to past runs CSV
        tune: Enable hyperparameter tuning
        save_shap: Compute and save SHAP explanations
        seed: Random seed for reproducibility
        verbose: Print progress messages

    Returns:
        Dictionary with pipeline execution results and output paths
    """
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    # Create timestamped output directory
    run_output_dir = os.path.join(output_base_dir, timestamp)
    os.makedirs(run_output_dir, exist_ok=True)
    os.makedirs(os.path.join(run_output_dir, "plots"), exist_ok=True)

    results: Dict[str, Any] = {
        "timestamp": timestamp,
        "start_time": start_time.isoformat(),
        "success": False,
        "steps": {},
        "output_dir": run_output_dir,
        "errors": [],
    }

    if verbose:
        print("\n" + "=" * 70)
        print("  FULL MODELLING PIPELINE")
        print("  " + "=" * 66)
        print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Output directory: {run_output_dir}")
        print("=" * 70)

    # =========================================================================
    # Step 1: Build Modelling Dataset
    # =========================================================================
    if verbose:
        print("\n" + "-" * 70)
        print("  STEP 1: Build Modelling Dataset")
        print("-" * 70)

    try:
        from scripts.build_modelling_dataset import build_modelling_dataset

        dataset_output_path = os.path.join(run_output_dir, "modelling_dataset.csv")
        diagnostics_output_path = os.path.join(
            run_output_dir, "modelling_dataset_report.json"
        )

        df = build_modelling_dataset(
            history_path=history_path,
            baselines_path=baselines_path,
            past_runs_path=past_runs_path,
            output_path=dataset_output_path,
            diagnostics_path=diagnostics_output_path,
            verbose=verbose,
        )

        # Also copy to standard location for other tools
        standard_dataset_path = "data/modelling_dataset.csv"
        shutil.copy2(dataset_output_path, standard_dataset_path)

        results["steps"]["build_dataset"] = {
            "success": True,
            "rows": len(df),
            "columns": len(df.columns),
            "output_path": dataset_output_path,
            "standard_path": standard_dataset_path,
        }

        if verbose:
            print(f"\n  ✓ Dataset built: {len(df)} rows, {len(df.columns)} columns")

    except Exception as e:
        error_msg = f"Step 1 (Build Dataset) failed: {e}"
        results["errors"].append(error_msg)
        results["steps"]["build_dataset"] = {"success": False, "error": str(e)}
        if verbose:
            print(f"\n  ✗ ERROR: {error_msg}")
        # Cannot continue without dataset
        _save_summary(results, run_output_dir)
        return results

    # =========================================================================
    # Step 2: Run Time-Aware Backtesting
    # =========================================================================
    if verbose:
        print("\n" + "-" * 70)
        print("  STEP 2: Run Time-Aware Backtesting")
        print("-" * 70)

    try:
        from scripts.backtest_timeaware import run_backtest

        backtest_results = run_backtest(
            dataset_path=dataset_output_path,
            target_col="target_ticket_median",
            n_folds=5,
            output_dir=run_output_dir,
            seed=seed,
            verbose=verbose,
        )

        results["steps"]["backtest"] = {
            "success": True,
            "methods": backtest_results.get("methods", {}),
            "output_files": backtest_results.get("output_files", {}),
        }

        if verbose:
            print("\n  ✓ Backtesting complete")
            methods = backtest_results.get("methods", {})
            if "full_model" in methods:
                fm = methods["full_model"]
                print(
                    f"    Full model: MAE={fm['mae']:.0f}, R²={fm['r2']:.3f}"
                )

    except Exception as e:
        error_msg = f"Step 2 (Backtest) failed: {e}"
        results["errors"].append(error_msg)
        results["steps"]["backtest"] = {"success": False, "error": str(e)}
        if verbose:
            print(f"\n  ✗ WARNING: {error_msg}")
            print("    Continuing to Step 3...")

    # =========================================================================
    # Step 3: Train Final Safe Model
    # =========================================================================
    if verbose:
        print("\n" + "-" * 70)
        print("  STEP 3: Train Final Safe Model")
        print("-" * 70)

    try:
        from scripts.train_safe_model import train_model

        model_output_path = "models/model_xgb_remount_postcovid.joblib"
        metadata_output_path = "models/model_xgb_remount_postcovid.json"
        importance_output_path = os.path.join(
            run_output_dir, "feature_importances.csv"
        )

        train_results = train_model(
            dataset_path=dataset_output_path,
            model_type="xgboost",
            target_col="target_ticket_median",
            model_output_path=model_output_path,
            metadata_output_path=metadata_output_path,
            importance_output_path=importance_output_path,
            tune=tune,
            save_shap=save_shap,
            seed=seed,
            verbose=verbose,
        )

        # Move SHAP outputs to run directory if generated
        if save_shap and os.path.exists("results/shap"):
            shap_dest = os.path.join(run_output_dir, "shap")
            if os.path.exists(shap_dest):
                shutil.rmtree(shap_dest)
            shutil.move("results/shap", shap_dest)

        results["steps"]["train_model"] = {
            "success": train_results.get("success", False),
            "model_path": model_output_path,
            "metadata_path": metadata_output_path,
            "cv_metrics": train_results.get("cv_metrics", {}),
            "train_metrics": train_results.get("train_metrics", {}),
        }

        if verbose and train_results.get("success"):
            cv = train_results.get("cv_metrics", {})
            print("\n  ✓ Model trained successfully")
            if cv:
                print(
                    f"    CV MAE: {cv.get('mae_mean', 0):.0f} ± "
                    f"{cv.get('mae_std', 0):.0f}"
                )
                print(
                    f"    CV R²: {cv.get('r2_mean', 0):.3f} ± "
                    f"{cv.get('r2_std', 0):.3f}"
                )

    except Exception as e:
        error_msg = f"Step 3 (Train Model) failed: {e}"
        results["errors"].append(error_msg)
        results["steps"]["train_model"] = {"success": False, "error": str(e)}
        if verbose:
            print(f"\n  ✗ ERROR: {error_msg}")

    # =========================================================================
    # Finalize
    # =========================================================================
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    results["end_time"] = end_time.isoformat()
    results["duration_seconds"] = duration
    results["success"] = all(
        step.get("success", False) for step in results["steps"].values()
    )

    # Save pipeline summary
    _save_summary(results, run_output_dir)

    if verbose:
        print("\n" + "=" * 70)
        print("  PIPELINE COMPLETE")
        print("=" * 70)
        print(f"  Duration: {duration:.1f} seconds")
        print(f"  Status: {'✓ SUCCESS' if results['success'] else '✗ FAILED'}")
        print(f"  Output directory: {run_output_dir}")

        if results["errors"]:
            print(f"\n  Errors encountered:")
            for err in results["errors"]:
                print(f"    - {err}")

        print("\n  Output files:")
        for root, dirs, files in os.walk(run_output_dir):
            level = root.replace(run_output_dir, "").count(os.sep)
            indent = "    " + "  " * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = "    " + "  " * (level + 1)
            for file in files:
                print(f"{subindent}{file}")

        print("=" * 70 + "\n")

    return results


def _save_summary(results: Dict[str, Any], output_dir: str) -> None:
    """Save pipeline summary to JSON file."""
    summary_path = os.path.join(output_dir, "pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run the complete modelling pipeline end-to-end",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default options
  python scripts/run_full_pipeline.py

  # Enable hyperparameter tuning
  python scripts/run_full_pipeline.py --tune

  # Include SHAP explanations
  python scripts/run_full_pipeline.py --save-shap

  # Specify custom data paths
  python scripts/run_full_pipeline.py --history data/custom_history.csv
""",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Base output directory (default: results/)",
    )
    parser.add_argument(
        "--history",
        default="data/productions/history_city_sales.csv",
        help="Path to historical sales CSV",
    )
    parser.add_argument(
        "--baselines",
        default="data/productions/baselines.csv",
        help="Path to baselines CSV",
    )
    parser.add_argument(
        "--past-runs",
        default="data/productions/past_runs.csv",
        help="Path to past runs CSV",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning for model training",
    )
    parser.add_argument(
        "--save-shap",
        action="store_true",
        help="Compute and save SHAP explanations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    try:
        results = run_full_pipeline(
            output_base_dir=args.output_dir,
            history_path=args.history,
            baselines_path=args.baselines,
            past_runs_path=args.past_runs,
            tune=args.tune,
            save_shap=args.save_shap,
            seed=args.seed,
            verbose=not args.quiet,
        )

        if not results["success"]:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
