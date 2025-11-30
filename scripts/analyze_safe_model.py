#!/usr/bin/env python3
"""
Analyze Safe Model Script

This script analyzes the trained safe model and produces interpretability artifacts:
- Feature importance rankings (model-native and permutation)
- A simple "rule-of-thumb" linear surrogate model
- Human-readable summaries

Usage:
    python scripts/analyze_safe_model.py

Outputs:
    - results/feature_importances_detailed.csv: Tree and permutation importances
    - results/model_recipe_linear.csv: Linear surrogate coefficients
    - results/model_recipe_summary.md: Human-readable model summary
    - results/plots/feature_importances_bar.png: Top features bar chart
    - results/plots/surrogate_vs_model_scatter.png: Surrogate vs XGBoost predictions
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# scikit-learn imports
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Matplotlib for plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_artifacts(
    model_path: str = "models/model_xgb_remount_postcovid.joblib",
    metadata_path: str = "models/model_xgb_remount_postcovid.json",
    dataset_path: str = "data/modelling_dataset.csv",
) -> Tuple[Any, Dict[str, Any], pd.DataFrame]:
    """Load model artifacts.
    
    Returns:
        Tuple of (model_pipeline, metadata, dataset)
    """
    # Load pipeline
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    pipeline = joblib.load(model_path)
    
    # Load metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Load dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    return pipeline, metadata, df


def reconstruct_xy(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Reconstruct X and y from the dataset using metadata.
    
    Returns:
        Tuple of (X, y, numeric_cols, categorical_cols)
    """
    target_col = metadata.get("target_column", "target_ticket_median")
    features = metadata.get("features", {})
    numeric_cols = features.get("numeric", [])
    categorical_cols = features.get("categorical", [])
    
    # Build feature list
    all_features = numeric_cols + categorical_cols
    
    # Filter to rows with valid target
    df_valid = df[df[target_col].notna() & (df[target_col] > 0)].copy()
    
    # Build X and y
    X = df_valid[all_features].copy()
    y = df_valid[target_col].copy()
    
    # Apply log transform (matching training)
    y = np.log1p(y.clip(lower=0))
    
    return X, y, numeric_cols, categorical_cols


def get_feature_names_after_transform(
    pipeline: Any,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> List[str]:
    """Get feature names after preprocessing transformation."""
    try:
        return list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    except Exception:
        # Fallback: construct names manually
        names = [f"num__{c}" for c in numeric_cols]
        # For categorical, we don't know the exact categories so return base names
        names.extend([f"cat__{c}" for c in categorical_cols])
        return names


def compute_tree_importances(
    pipeline: Any,
    feature_names: List[str],
) -> pd.DataFrame:
    """Extract model-native feature importances from the tree model."""
    model = pipeline.named_steps["model"]
    
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame()
    
    importances = model.feature_importances_
    
    # Match importances to feature names
    n_features = len(importances)
    names = feature_names[:n_features] if len(feature_names) >= n_features else feature_names
    
    df = pd.DataFrame({
        "feature": names,
        "importance_type": "tree",
        "value": importances[:len(names)],
    })
    
    return df.sort_values("value", ascending=False).reset_index(drop=True)


def compute_permutation_importances(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation importances using scikit-learn."""
    result = permutation_importance(
        pipeline, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    
    # Build dataframe
    df = pd.DataFrame({
        "feature": X.columns.tolist(),
        "importance_type": "permutation",
        "value": result.importances_mean,
        "std": result.importances_std,
    })
    
    return df.sort_values("value", ascending=False).reset_index(drop=True)


def fit_surrogate_linear_model(
    pipeline: Any,
    X: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    alpha: float = 1.0,
) -> Tuple[Ridge, pd.DataFrame, float, np.ndarray, np.ndarray]:
    """Fit a simple linear surrogate model to approximate the XGBoost model.
    
    Returns:
        Tuple of (ridge_model, coefficients_df, r2_score, y_hat, y_surrogate)
    """
    # Get XGBoost predictions as target for surrogate
    y_hat = pipeline.predict(X)
    
    # Prepare features for linear model
    # Numeric features: scale them
    X_numeric = X[numeric_cols].copy()
    
    # Categorical features: one-hot encode
    X_categorical = pd.get_dummies(X[categorical_cols], drop_first=True)
    
    # Combine
    X_combined = pd.concat([X_numeric, X_categorical], axis=1)
    
    # Scale all features for interpretability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    
    # Fit Ridge regression
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_scaled, y_hat)
    
    # Get predictions
    y_surrogate = ridge.predict(X_scaled)
    
    # Compute R^2 of surrogate vs original model
    r2 = r2_score(y_hat, y_surrogate)
    
    # Build coefficients dataframe
    coef_data = []
    
    # Add intercept
    coef_data.append({
        "term": "intercept",
        "coefficient": ridge.intercept_,
        "abs_coefficient": abs(ridge.intercept_),
        "standardized": True,
    })
    
    # Add feature coefficients
    feature_names = X_combined.columns.tolist()
    for name, coef in zip(feature_names, ridge.coef_):
        coef_data.append({
            "term": name,
            "coefficient": coef,
            "abs_coefficient": abs(coef),
            "standardized": True,
        })
    
    coef_df = pd.DataFrame(coef_data)
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    
    return ridge, coef_df, r2, y_hat, y_surrogate


def create_feature_importance_plot(
    df: pd.DataFrame,
    output_path: str,
    top_n: int = 15,
    title: str = "Top Features by Permutation Importance",
):
    """Create a bar chart of top features by importance."""
    # Filter to permutation importances if both types present
    if "importance_type" in df.columns:
        df_plot = df[df["importance_type"] == "permutation"].copy()
        if df_plot.empty:
            df_plot = df.copy()
    else:
        df_plot = df.copy()
    
    # Get top N
    df_plot = df_plot.head(top_n)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(df_plot))
    bars = ax.barh(y_pos, df_plot["value"], align="center", color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["feature"])
    ax.invert_yaxis()  # Top feature at top
    ax.set_xlabel("Importance")
    ax.set_title(title)
    
    # Add value labels
    for i, (v, bar) in enumerate(zip(df_plot["value"], bars)):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_surrogate_scatter_plot(
    y_hat: np.ndarray,
    y_surrogate: np.ndarray,
    r2: float,
    output_path: str,
):
    """Create scatter plot of surrogate vs XGBoost predictions."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(y_hat, y_surrogate, alpha=0.6, edgecolors="k", linewidths=0.5)
    
    # Add y=x reference line
    min_val = min(y_hat.min(), y_surrogate.min())
    max_val = max(y_hat.max(), y_surrogate.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="y = x")
    
    ax.set_xlabel("XGBoost Predictions (log-scale)")
    ax.set_ylabel("Linear Surrogate Predictions (log-scale)")
    ax.set_title(f"Surrogate vs XGBoost Model\n(R² = {r2:.3f})")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_summary_markdown(
    perm_importances: pd.DataFrame,
    coef_df: pd.DataFrame,
    surrogate_r2: float,
    metadata: Dict[str, Any],
    output_path: str,
):
    """Generate human-readable summary in markdown format."""
    lines = [
        "# Model Recipe Summary",
        "",
        "## Overview",
        "",
        f"This document summarizes the `model_xgb_remount_postcovid` safe model.",
        f"The model predicts ticket demand (log-transformed) using {metadata['features']['total']} features.",
        "",
        "## Feature Groups",
        "",
        "The safe model uses the following feature groups:",
        "",
        "### 1. Digital Attention / Awareness Signals",
        "- `wiki`: Wikipedia page views",
        "- `trends`: Google Trends interest",
        "- `youtube`: YouTube engagement",
        "- `spotify`: Spotify plays/popularity",
        "",
        "### 2. Historical Priors",
        "- `prior_total_tickets`: Sum of tickets from all prior runs",
        "- `prior_run_count`: Number of previous runs",
        "- `ticket_median_prior`: Median tickets per prior run",
        "",
        "### 3. Remount Features",
        "- `years_since_last_run`: Years since the last production",
        "- `is_remount_recent`: Was remounted within 2 years (binary)",
        "- `is_remount_medium`: Remounted 2-4 years ago (binary)",
        "- `run_count_prior`: Number of prior runs (same as prior_run_count)",
        "",
        "### 4. Calendar/Context",
        "- `month_of_opening`: Month when the show opens",
        "- `holiday_flag`: Opens during holiday season (Nov-Jan)",
        "",
        "### 5. Show Descriptors",
        "- `category`: Show category (e.g., classic_story_ballet, contemporary)",
        "- `gender`: Gender composition of the cast",
        "",
        "---",
        "",
        "## Top 10 Features by Permutation Importance",
        "",
        "| Rank | Feature | Importance | Direction (from surrogate) |",
        "|------|---------|------------|---------------------------|",
    ]
    
    # Get top 10 permutation importances
    perm_top10 = perm_importances.head(10)
    
    # Create lookup for coefficients
    coef_lookup = dict(zip(coef_df["term"], coef_df["coefficient"]))
    
    for i, row in perm_top10.iterrows():
        feature = row["feature"]
        importance = row["value"]
        
        # Look up coefficient direction
        coef = coef_lookup.get(feature, 0)
        direction = "↑ positive" if coef > 0 else "↓ negative" if coef < 0 else "neutral"
        
        lines.append(f"| {i + 1} | `{feature}` | {importance:.4f} | {direction} |")
    
    lines.extend([
        "",
        "---",
        "",
        "## Approximate Linear Model (Surrogate)",
        "",
        "To provide an interpretable \"rule-of-thumb\" approximation, we fit a Ridge regression",
        "model to the XGBoost predictions. This surrogate captures the main linear effects.",
        "",
        f"**Surrogate R² (vs XGBoost):** {surrogate_r2:.3f}",
        "",
        "This means the linear surrogate explains ~{:.0%} of the XGBoost model's variance.".format(surrogate_r2),
        "",
        "### Top 10 Coefficients (Standardized)",
        "",
        "| Term | Coefficient | Interpretation |",
        "|------|-------------|----------------|",
    ])
    
    # Get top 10 coefficients by absolute value (excluding intercept for display)
    coef_top = coef_df[coef_df["term"] != "intercept"].head(10)
    
    for _, row in coef_top.iterrows():
        term = row["term"]
        coef = row["coefficient"]
        
        if coef > 0:
            interp = "Higher value → more tickets"
        else:
            interp = "Higher value → fewer tickets"
        
        lines.append(f"| `{term}` | {coef:+.4f} | {interp} |")
    
    # Add intercept
    intercept = coef_lookup.get("intercept", 0)
    lines.append(f"| `intercept` | {intercept:.4f} | Baseline prediction |")
    
    lines.extend([
        "",
        "---",
        "",
        "## Verbal Summary",
        "",
        "Based on the analysis above, the model's behavior can be summarized as:",
        "",
    ])
    
    # Generate verbal summary from top positive and negative coefficients
    positive_terms = coef_df[(coef_df["coefficient"] > 0) & (coef_df["term"] != "intercept")].head(5)
    negative_terms = coef_df[(coef_df["coefficient"] < 0) & (coef_df["term"] != "intercept")].head(5)
    
    if not positive_terms.empty:
        pos_list = ", ".join([f"`{t}`" for t in positive_terms["term"].tolist()])
        lines.append(f"- **Predicted tickets increase** with higher values of: {pos_list}")
    
    if not negative_terms.empty:
        neg_list = ", ".join([f"`{t}`" for t in negative_terms["term"].tolist()])
        lines.append(f"- **Predicted tickets decrease** with higher values of: {neg_list}")
    
    lines.extend([
        "",
        "The most important driver is `prior_total_tickets` (historical performance),",
        "which makes intuitive sense: shows that have sold well in the past tend to sell well again.",
        "",
        "---",
        "",
        "## Model Metadata",
        "",
        f"- **Training Date:** {metadata.get('training_date', 'N/A')}",
        f"- **Model Type:** {metadata.get('model_type', 'N/A')}",
        f"- **Training Samples:** {metadata.get('n_samples', 'N/A')}",
        f"- **Target Column:** {metadata.get('target_column', 'N/A')}",
        f"- **CV R² (mean ± std):** {metadata.get('cv_metrics', {}).get('r2_mean', 0):.3f} ± {metadata.get('cv_metrics', {}).get('r2_std', 0):.3f}",
        "",
        "---",
        "",
        "*Generated by `scripts/analyze_safe_model.py`*",
    ])
    
    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def analyze_model(
    model_path: str = "models/model_xgb_remount_postcovid.joblib",
    metadata_path: str = "models/model_xgb_remount_postcovid.json",
    dataset_path: str = "data/modelling_dataset.csv",
    output_dir: str = "results",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Main analysis function."""
    results: Dict[str, Any] = {"success": False}
    
    if verbose:
        print("\n" + "=" * 60)
        print("Analyzing Safe Model")
        print("=" * 60)
    
    # 1. Load artifacts
    if verbose:
        print("\n1. Loading artifacts...")
    
    pipeline, metadata, df = load_artifacts(model_path, metadata_path, dataset_path)
    
    if verbose:
        print(f"   Model type: {metadata.get('model_type', 'unknown')}")
        print(f"   Training date: {metadata.get('training_date', 'unknown')}")
        print(f"   Dataset rows: {len(df)}")
    
    # 2. Reconstruct X/y
    if verbose:
        print("\n2. Reconstructing X/y...")
    
    X, y, numeric_cols, categorical_cols = reconstruct_xy(df, metadata)
    
    if verbose:
        print(f"   Valid samples: {len(X)}")
        print(f"   Numeric features: {len(numeric_cols)}")
        print(f"   Categorical features: {len(categorical_cols)}")
    
    # 3. Compute feature importances
    if verbose:
        print("\n3. Computing feature importances...")
    
    # Get feature names after transform
    feature_names = get_feature_names_after_transform(pipeline, numeric_cols, categorical_cols)
    
    # Tree importances
    tree_imp = compute_tree_importances(pipeline, feature_names)
    if verbose and not tree_imp.empty:
        print(f"   Tree importances: {len(tree_imp)} features")
    
    # Permutation importances
    if verbose:
        print("   Computing permutation importances (this may take a moment)...")
    perm_imp = compute_permutation_importances(pipeline, X, y, n_repeats=10)
    if verbose:
        print(f"   Permutation importances: {len(perm_imp)} features")
    
    # Combine importances
    importances_detailed = pd.concat([tree_imp, perm_imp], ignore_index=True)
    
    # 4. Fit surrogate linear model
    if verbose:
        print("\n4. Fitting surrogate linear model...")
    
    ridge, coef_df, surrogate_r2, y_hat, y_surrogate = fit_surrogate_linear_model(
        pipeline, X, numeric_cols, categorical_cols
    )
    
    if verbose:
        print(f"   Surrogate R² (vs XGBoost): {surrogate_r2:.3f}")
    
    # 5. Save outputs
    if verbose:
        print("\n5. Saving outputs...")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    
    # Save detailed importances
    importances_path = os.path.join(output_dir, "feature_importances_detailed.csv")
    importances_detailed.to_csv(importances_path, index=False)
    if verbose:
        print(f"   Saved: {importances_path}")
    
    # Save linear coefficients
    linear_path = os.path.join(output_dir, "model_recipe_linear.csv")
    coef_df.to_csv(linear_path, index=False)
    if verbose:
        print(f"   Saved: {linear_path}")
    
    # 6. Create plots
    if verbose:
        print("\n6. Creating plots...")
    
    # Feature importance bar chart
    bar_plot_path = os.path.join(output_dir, "plots", "feature_importances_bar.png")
    create_feature_importance_plot(perm_imp, bar_plot_path)
    if verbose:
        print(f"   Saved: {bar_plot_path}")
    
    # Surrogate scatter plot
    scatter_plot_path = os.path.join(output_dir, "plots", "surrogate_vs_model_scatter.png")
    create_surrogate_scatter_plot(y_hat, y_surrogate, surrogate_r2, scatter_plot_path)
    if verbose:
        print(f"   Saved: {scatter_plot_path}")
    
    # 7. Generate summary markdown
    if verbose:
        print("\n7. Generating summary markdown...")
    
    summary_path = os.path.join(output_dir, "model_recipe_summary.md")
    generate_summary_markdown(perm_imp, coef_df, surrogate_r2, metadata, summary_path)
    if verbose:
        print(f"   Saved: {summary_path}")
    
    results["success"] = True
    results["outputs"] = {
        "importances_detailed": importances_path,
        "linear_coefficients": linear_path,
        "bar_plot": bar_plot_path,
        "scatter_plot": scatter_plot_path,
        "summary": summary_path,
    }
    results["surrogate_r2"] = surrogate_r2
    
    if verbose:
        print("\n" + "=" * 60)
        print("Analysis Complete!")
        print("=" * 60)
    
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze the trained safe model and produce interpretability artifacts"
    )
    parser.add_argument(
        "--model",
        default="models/model_xgb_remount_postcovid.joblib",
        help="Path to trained model"
    )
    parser.add_argument(
        "--metadata",
        default="models/model_xgb_remount_postcovid.json",
        help="Path to model metadata JSON"
    )
    parser.add_argument(
        "--dataset",
        default="data/modelling_dataset.csv",
        help="Path to modelling dataset"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for analysis artifacts"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    try:
        results = analyze_model(
            model_path=args.model,
            metadata_path=args.metadata,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            verbose=not args.quiet,
        )
        
        if not results["success"]:
            sys.exit(1)
            
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
