#!/usr/bin/env python3
"""
Validation script for newly added features.

Checks:
1. Presence in modelling dataset
2. Usage in model training (via feature importance file)
3. Directionality and correlations with target

This is a READ-ONLY diagnostic script.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Define the features to validate
ECONOMIC_FEATURES = [
    'consumer_confidence_prairies',
    'energy_index', 
    'inflation_adjustment_factor',
    'city_median_household_income'
]

LIVE_ANALYTICS_FEATURES = ['aud__engagement_factor']
RESEARCH_FEATURES = ['res__arts_share_giving']

ALL_NEW_FEATURES = ECONOMIC_FEATURES + LIVE_ANALYTICS_FEATURES + RESEARCH_FEATURES

def main():
    print("=" * 80)
    print("VALIDATION REPORT: Newly Added Features")
    print("=" * 80)
    
    # ========================================================================
    # 1. Check presence in modelling dataset
    # ========================================================================
    print("\n1. PRESENCE IN MODELLING DATASET")
    print("-" * 80)
    
    dataset_path = Path("data/modelling_dataset.csv")
    if not dataset_path.exists():
        print(f"❌ ERROR: Dataset not found at {dataset_path}")
        return
    
    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Target column: target_ticket_median")
    
    presence_results = {}
    for feat in ALL_NEW_FEATURES:
        if feat in df.columns:
            non_null = df[feat].notna().sum()
            total = len(df)
            pct = 100 * non_null / total
            print(f"  ✓ {feat}: {non_null}/{total} rows ({pct:.1f}% coverage)")
            presence_results[feat] = True
        else:
            print(f"  ✗ {feat}: NOT FOUND")
            presence_results[feat] = False
    
    # ========================================================================
    # 2. Check usage in model training
    # ========================================================================
    print("\n2. USAGE IN MODEL TRAINING")
    print("-" * 80)
    
    importance_path = Path("results/feature_importances.csv")
    if not importance_path.exists():
        print(f"❌ WARNING: Feature importances not found at {importance_path}")
        print("   Model may not have been trained yet.")
        importances_df = None
    else:
        importances_df = pd.read_csv(importance_path)
        print(f"Feature importances file contains {len(importances_df)} features")
        
        usage_results = {}
        for feat in ALL_NEW_FEATURES:
            # Look for the feature with num__ prefix (from preprocessing)
            preprocessed_name = f"num__{feat}"
            if preprocessed_name in importances_df['feature'].values:
                importance = importances_df.loc[
                    importances_df['feature'] == preprocessed_name, 
                    'importance'
                ].iloc[0]
                print(f"  ✓ {feat}: Used in training (importance={importance:.6f})")
                usage_results[feat] = True
            else:
                print(f"  ✗ {feat}: NOT FOUND in feature importances")
                usage_results[feat] = False
    
    # ========================================================================
    # 3. Feature importances table
    # ========================================================================
    print("\n3. FEATURE IMPORTANCES (NEW FEATURES ONLY)")
    print("-" * 80)
    
    if importances_df is not None:
        # Extract new features and sort by importance
        new_feat_importances = []
        for feat in ALL_NEW_FEATURES:
            preprocessed_name = f"num__{feat}"
            if preprocessed_name in importances_df['feature'].values:
                importance = importances_df.loc[
                    importances_df['feature'] == preprocessed_name, 
                    'importance'
                ].iloc[0]
                new_feat_importances.append({
                    'feature': feat,
                    'importance': importance
                })
        
        if new_feat_importances:
            importance_table = pd.DataFrame(new_feat_importances)
            importance_table = importance_table.sort_values('importance', ascending=False)
            
            print("\nFeature                              Importance    Rank")
            print("-" * 80)
            for idx, row in importance_table.iterrows():
                # Get rank among ALL features
                preprocessed_name = f"num__{row['feature']}"
                rank = (importances_df['importance'] > row['importance']).sum() + 1
                total = len(importances_df)
                print(f"{row['feature']:35s} {row['importance']:10.6f}    {rank:3d}/{total}")
    
    # ========================================================================
    # 4. Directionality sanity check
    # ========================================================================
    print("\n4. DIRECTIONALITY SANITY CHECK")
    print("-" * 80)
    print("Correlations with target (target_ticket_median):\n")
    
    # Remove rows with missing target
    df_valid = df[df['target_ticket_median'].notna()].copy()
    target = df_valid['target_ticket_median']
    
    directionality_results = []
    
    for feat in ALL_NEW_FEATURES:
        if feat not in df_valid.columns:
            continue
        
        feature_series = df_valid[feat]
        
        # Skip if all values are the same
        if feature_series.nunique() <= 1:
            print(f"{feat:35s} - No variance (all values identical)")
            directionality_results.append({
                'feature': feat,
                'correlation': np.nan,
                'mean_low': np.nan,
                'mean_high': np.nan,
                'note': 'No variance'
            })
            continue
        
        # Compute correlation
        valid_mask = feature_series.notna() & target.notna()
        if valid_mask.sum() < 2:
            print(f"{feat:35s} - Insufficient data")
            continue
        
        corr = np.corrcoef(
            feature_series[valid_mask], 
            target[valid_mask]
        )[0, 1]
        
        # Split into low/high groups at median
        median_val = feature_series.median()
        low_mask = feature_series <= median_val
        high_mask = feature_series > median_val
        
        mean_target_low = target[low_mask].mean()
        mean_target_high = target[high_mask].mean()
        
        # Determine sign
        sign = "positive ↑" if corr > 0 else "negative ↓"
        
        print(f"{feat:35s} correlation={corr:+.4f} ({sign})")
        print(f"  {'':35s} Low half mean target: {mean_target_low:6.0f}")
        print(f"  {'':35s} High half mean target: {mean_target_high:6.0f}")
        
        directionality_results.append({
            'feature': feat,
            'correlation': corr,
            'mean_low': mean_target_low,
            'mean_high': mean_target_high,
            'sign': sign
        })
    
    # ========================================================================
    # 5. Summary and interpretation
    # ========================================================================
    print("\n5. INTERPRETATION NOTES")
    print("-" * 80)
    
    # Economic features
    print("\nEconomic Features (econ__*):")
    for feat in ECONOMIC_FEATURES:
        result = next((r for r in directionality_results if r['feature'] == feat), None)
        if result is None or pd.isna(result['correlation']):
            print(f"  {feat}: Insufficient variance or data")
        else:
            corr = result['correlation']
            sign = result['sign']
            
            if 'consumer_confidence' in feat:
                expected = "positive (higher confidence → more attendance)"
                sensible = "✓" if corr > 0 else "⚠"
                print(f"  {feat}: {sign} - {sensible} {expected}")
            elif 'energy_index' in feat:
                expected = "mixed (high energy prices may reduce disposable income)"
                print(f"  {feat}: {sign} - {expected}")
            elif 'inflation_adjustment' in feat:
                expected = "positive (accounts for price level changes)"
                sensible = "✓" if corr > 0 else "⚠"
                print(f"  {feat}: {sign} - {sensible} {expected}")
            elif 'median_household_income' in feat:
                expected = "positive (higher income → more arts spending)"
                sensible = "✓" if corr > 0 else "⚠"
                print(f"  {feat}: {sign} - {sensible} {expected}")
    
    # Engagement factor
    print("\nLive Analytics Features:")
    result = next((r for r in directionality_results if r['feature'] == 'aud__engagement_factor'), None)
    if result and not pd.isna(result['correlation']):
        corr = result['correlation']
        sign = result['sign']
        expected = "positive (higher engagement → more tickets)"
        sensible = "✓" if corr > 0 else "⚠"
        print(f"  aud__engagement_factor: {sign} - {sensible} {expected}")
    
    # Research features
    print("\nResearch Features (res__*):")
    result = next((r for r in directionality_results if r['feature'] == 'res__arts_share_giving'), None)
    if result and not pd.isna(result['correlation']):
        corr = result['correlation']
        sign = result['sign']
        expected = "positive (higher arts giving → more cultural engagement)"
        sensible = "✓" if corr > 0 else "?"
        print(f"  res__arts_share_giving: {sign} - {sensible} {expected}")
    else:
        print(f"  res__arts_share_giving: No variance detected (limited year coverage)")
    
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
