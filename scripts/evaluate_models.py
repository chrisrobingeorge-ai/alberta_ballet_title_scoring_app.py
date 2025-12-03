#!/usr/bin/env python3
"""
Evaluate baseline models for ticket sales prediction.

This script loads historical ticket sales data, performs a chronological train/test split,
and compares a mean baseline against a linear regression model using title features.

Usage:
    python scripts/evaluate_models.py
"""

import sys
from pathlib import Path

# Add project root to path to enable imports from data.loader
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

from data.loader import load_history_sales
from features.title_features import add_title_features


def main():
    """Main evaluation workflow."""
    print("=" * 70)
    print("MODEL COMPARISON: BASELINE VS LINEAR REGRESSION")
    print("=" * 70)
    print()
    
    # Load historical sales data
    print("Loading historical sales data...")
    df = load_history_sales()
    
    if df.empty:
        print("ERROR: No data loaded. Please check data/productions/history_city_sales.csv")
        sys.exit(1)
    
    print(f"Loaded {len(df)} records")
    
    # Verify required columns
    if "single_tickets" not in df.columns:
        print("ERROR: 'single_tickets' column not found in data")
        sys.exit(1)
    
    if "start_date" not in df.columns:
        print("ERROR: 'start_date' column not found in data")
        sys.exit(1)
    
    # Add title features
    print("Generating title features...")
    df = add_title_features(df)
    print()
    
    # Sort chronologically by start_date
    df = df.sort_values("start_date").reset_index(drop=True)
    
    # Split: 80% train (older shows), 20% test (newer shows)
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Train set: {len(train_df)} records (oldest to {train_df['start_date'].max()})")
    print(f"Test set:  {len(test_df)} records ({test_df['start_date'].min()} to {test_df['start_date'].max()})")
    print()
    
    # Extract target variable
    y_train = train_df["single_tickets"].values
    y_test = test_df["single_tickets"].values
    
    # Define feature columns for Linear Regression
    feature_cols = ['is_benchmark_classic', 'title_word_count']
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    
    # ========================================
    # Model 1: Mean Baseline
    # ========================================
    mean_baseline = np.mean(y_train)
    y_pred_baseline = np.full(len(y_test), mean_baseline)
    mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
    
    # ========================================
    # Model 2: Linear Regression
    # ========================================
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    
    # Calculate improvement
    improvement = mae_baseline - mae_lr
    improvement_pct = (improvement / mae_baseline) * 100
    
    # Print results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"{'Model':<30} {'MAE (tickets)':<20} {'Improvement':<20}")
    print("-" * 70)
    print(f"{'Mean Baseline':<30} {mae_baseline:>15.2f}     {'-':<20}")
    print(f"{'Linear Regression':<30} {mae_lr:>15.2f}     {improvement:>+8.2f} ({improvement_pct:>+6.1f}%)")
    print("=" * 70)
    print()
    
    # Model coefficients
    print("Linear Regression Coefficients:")
    print(f"  Intercept:           {lr_model.intercept_:>10.2f}")
    for i, col in enumerate(feature_cols):
        print(f"  {col:<20} {lr_model.coef_[i]:>10.2f}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
