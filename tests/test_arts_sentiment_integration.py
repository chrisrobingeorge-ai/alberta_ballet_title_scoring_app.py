#!/usr/bin/env python3
"""
Test script to verify nanos_arts_donors.csv integration with economic_features.py
"""

import pandas as pd
import sys
from features.economic_features import (
    load_arts_sentiment,
    add_arts_sentiment_feature,
    add_economic_features,
    get_feature_names
)


def test_load_arts_sentiment():
    """Test loading the arts sentiment data."""
    print("=" * 70)
    print("TEST 1: Loading Arts Sentiment Data")
    print("=" * 70)
    
    arts_data = load_arts_sentiment()
    
    if arts_data.empty:
        print("❌ FAILED: Arts sentiment data is empty or file not found")
        return False
    
    print(f"✓ Successfully loaded {len(arts_data)} year(s) of arts sentiment data")
    print("\nData:")
    print("-" * 70)
    print(arts_data.to_string(index=False))
    
    # Validate data
    print("\n" + "=" * 70)
    print("Data Validation:")
    print("=" * 70)
    
    checks = []
    checks.append(("Has 'year' column", 'year' in arts_data.columns))
    checks.append(("Has 'arts_sentiment' column", 'arts_sentiment' in arts_data.columns))
    checks.append(("All years are numeric", arts_data['year'].dtype in ['int64', 'Int64']))
    checks.append(("All sentiment values are numeric", pd.api.types.is_numeric_dtype(arts_data['arts_sentiment'])))
    checks.append(("No NaN years", arts_data['year'].notna().all()))
    checks.append(("No NaN sentiments", arts_data['arts_sentiment'].notna().all()))
    
    # Check value range (should be percentages, roughly 0-100)
    if not arts_data.empty:
        min_val = arts_data['arts_sentiment'].min()
        max_val = arts_data['arts_sentiment'].max()
        checks.append(("Values in reasonable range (0-100)", min_val >= 0 and max_val <= 100))
        print(f"\nSentiment Range: {min_val:.1f}% - {max_val:.1f}%")
        print(f"Median Sentiment: {arts_data['arts_sentiment'].median():.1f}%")
    
    for check_name, passed in checks:
        status = "✓" if passed else "❌"
        print(f"  {status} {check_name}")
    
    all_passed = all(c[1] for c in checks)
    print("=" * 70)
    
    return all_passed


def test_add_arts_sentiment_feature():
    """Test adding arts sentiment feature to a DataFrame."""
    print("\n" + "=" * 70)
    print("TEST 2: Adding Arts Sentiment Feature to DataFrame")
    print("=" * 70)
    
    # Create sample data with various years
    sample_data = pd.DataFrame({
        'show_title': [
            'The Nutcracker 2023',
            'Swan Lake 2024',
            'Cinderella 2025',
            'Romeo and Juliet 2020',  # Before data range
            'Contemporary 2026'  # After data range (should forward-fill)
        ],
        'start_date': [
            '2023-12-15',
            '2024-03-20',
            '2025-11-10',
            '2020-02-14',
            '2026-05-01'
        ]
    })
    
    print("\nInput DataFrame:")
    print("-" * 70)
    print(sample_data.to_string(index=False))
    
    # Add feature
    result = add_arts_sentiment_feature(sample_data)
    
    print("\n" + "=" * 70)
    print("Output DataFrame with Arts Sentiment:")
    print("=" * 70)
    
    display_cols = ['show_title', 'start_date', 'Econ_ArtsSentiment']
    print(result[display_cols].to_string(index=False))
    
    # Validate
    print("\n" + "=" * 70)
    print("Feature Validation:")
    print("=" * 70)
    
    checks = []
    checks.append(("Econ_ArtsSentiment column exists", 'Econ_ArtsSentiment' in result.columns))
    checks.append(("All values are numeric", pd.api.types.is_numeric_dtype(result['Econ_ArtsSentiment'])))
    checks.append(("No NaN values", result['Econ_ArtsSentiment'].notna().all()))
    
    # Check that 2023-2025 shows have expected values
    if 'Econ_ArtsSentiment' in result.columns:
        # Get values by matching the show title (since rows may be reordered)
        val_2023 = result[result['show_title'].str.contains('2023')]['Econ_ArtsSentiment'].iloc[0]
        val_2024 = result[result['show_title'].str.contains('2024')]['Econ_ArtsSentiment'].iloc[0]
        val_2025 = result[result['show_title'].str.contains('2025')]['Econ_ArtsSentiment'].iloc[0]
        
        print(f"\nYear-specific values:")
        print(f"  2023 show: {val_2023:.1f}%")
        print(f"  2024 show: {val_2024:.1f}%")
        print(f"  2025 show: {val_2025:.1f}%")
        
        # 2024 and 2025 should have the same value (both 12% per CSV)
        checks.append(("2024 and 2025 have same value", abs(val_2024 - val_2025) < 0.01))
        
        # 2023 should be 11% (different from 2024/2025)
        checks.append(("2023 has expected value (11%)", abs(val_2023 - 11.0) < 0.01))
        checks.append(("2024 has expected value (12%)", abs(val_2024 - 12.0) < 0.01))
    
    for check_name, passed in checks:
        status = "✓" if passed else "❌"
        print(f"  {status} {check_name}")
    
    all_passed = all(c[1] for c in checks)
    print("=" * 70)
    
    return all_passed


def test_full_economic_features():
    """Test that arts sentiment integrates with full economic features."""
    print("\n" + "=" * 70)
    print("TEST 3: Full Economic Features Integration")
    print("=" * 70)
    
    sample_data = pd.DataFrame({
        'show_title': ['Test Show'],
        'start_date': ['2024-06-15']
    })
    
    print("\nInput DataFrame:")
    print(sample_data.to_string(index=False))
    
    # Add all economic features
    result = add_economic_features(sample_data)
    
    print("\n" + "=" * 70)
    print("Output with All Economic Features:")
    print("=" * 70)
    
    feature_cols = get_feature_names()
    print(f"\nExpected features: {feature_cols}")
    
    print("\nFeature values:")
    for col in feature_cols:
        if col in result.columns:
            val = result[col].iloc[0]
            print(f"  {col:25s}: {val:>10.3f}")
    
    # Validate
    print("\n" + "=" * 70)
    print("Integration Validation:")
    print("=" * 70)
    
    checks = []
    for feat in feature_cols:
        checks.append((f"{feat} exists", feat in result.columns))
        if feat in result.columns:
            checks.append((f"{feat} is numeric", pd.api.types.is_numeric_dtype(result[feat])))
    
    checks.append(("get_feature_names includes Econ_ArtsSentiment", 
                   'Econ_ArtsSentiment' in feature_cols))
    
    for check_name, passed in checks:
        status = "✓" if passed else "❌"
        print(f"  {status} {check_name}")
    
    all_passed = all(c[1] for c in checks)
    print("=" * 70)
    
    return all_passed


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("ARTS SENTIMENT INTEGRATION TEST SUITE")
    print("=" * 70 + "\n")
    
    test1_passed = test_load_arts_sentiment()
    test2_passed = test_add_arts_sentiment_feature()
    test3_passed = test_full_economic_features()
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  Test 1 (Load Data):        {'✓ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"  Test 2 (Add Feature):      {'✓ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"  Test 3 (Integration):      {'✓ PASSED' if test3_passed else '❌ FAILED'}")
    print("=" * 70)
    
    if test1_passed and test2_passed and test3_passed:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
