#!/usr/bin/env python3
"""
Quick test to verify economic features are integrated into the modelling dataset.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd


def test_economic_features_in_dataset():
    """Test that economic features appear in the modelling dataset."""
    
    dataset_path = "data/modelling_dataset.csv"
    
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"❌ Dataset not found at {dataset_path}")
        print("   Run: python scripts/build_modelling_dataset.py")
        return False
    
    expected_econ_features = [
        'consumer_confidence_prairies',
        'energy_index',
        'inflation_adjustment_factor',
        'city_median_household_income'
    ]
    
    present = [f for f in expected_econ_features if f in df.columns]
    missing = [f for f in expected_econ_features if f not in df.columns]
    
    print("\n" + "=" * 60)
    print("Economic Feature Integration Test")
    print("=" * 60)
    print(f"Dataset path: {dataset_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"\nExpected economic features: {len(expected_econ_features)}")
    print(f"Present: {len(present)}")
    print(f"Missing: {len(missing)}")
    
    if present:
        print("\n✓ PRESENT FEATURES:")
        for feat in present:
            non_null = df[feat].notna().sum()
            pct_non_null = 100 * non_null / len(df)
            print(f"  - {feat}: {non_null}/{len(df)} non-null ({pct_non_null:.1f}%)")
    
    if missing:
        print("\n✗ MISSING FEATURES:")
        for feat in missing:
            print(f"  - {feat}")
    
    print("\nAll dataset columns:")
    for col in df.columns:
        print(f"  - {col}")
    
    success = len(present) == len(expected_econ_features)
    
    if success:
        print("\n" + "=" * 60)
        print("✓ SUCCESS: All economic features are present!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ FAILURE: Some economic features are missing")
        print("=" * 60)
    
    return success


if __name__ == "__main__":
    success = test_economic_features_in_dataset()
    sys.exit(0 if success else 1)
