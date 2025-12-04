#!/usr/bin/env python3
"""
Test script to verify live_analytics.csv integration with title_features.py
"""

import pandas as pd
import sys
from features.title_features import load_live_analytics_mapping, add_title_features

def test_load_mapping():
    """Test loading the live analytics mapping."""
    print("=" * 70)
    print("TEST 1: Loading Live Analytics Mapping")
    print("=" * 70)
    
    mapping = load_live_analytics_mapping()
    
    if not mapping:
        print("❌ FAILED: Mapping is empty or file not found")
        return False
    
    print(f"✓ Successfully loaded {len(mapping)} categories")
    print("\nCategory -> Customer Count Mapping:")
    print("-" * 70)
    for category, count in sorted(mapping.items(), key=lambda x: x[1], reverse=True):
        print(f"  {category:40s} : {count:>10,.0f}")
    
    # Calculate statistics
    values = list(mapping.values())
    print("\n" + "=" * 70)
    print("Statistics:")
    print(f"  Min:    {min(values):>10,.0f}")
    print(f"  Max:    {max(values):>10,.0f}")
    print(f"  Median: {pd.Series(values).median():>10,.0f}")
    print(f"  Mean:   {pd.Series(values).mean():>10,.0f}")
    print("=" * 70)
    
    return True


def test_add_features():
    """Test adding features to a sample DataFrame."""
    print("\n" + "=" * 70)
    print("TEST 2: Adding Features to Sample DataFrame")
    print("=" * 70)
    
    # Create sample data with various categories
    sample_data = pd.DataFrame({
        'show_title': [
            'The Nutcracker',
            'Swan Lake',
            'Contemporary Works',
            'Romeo and Juliet',
            'Unknown Ballet'
        ],
        'Category': [
            'family_classic',
            'classic_romance, classic_comedy, romantic_comedy',
            'contemporary',
            'romantic_tragedy',
            'unknown_category'  # This should fall back to median
        ]
    })
    
    print("\nInput DataFrame:")
    print(sample_data.to_string(index=False))
    
    # Add features
    result = add_title_features(sample_data)
    
    # Display results
    print("\n" + "=" * 70)
    print("Output DataFrame with New Features:")
    print("=" * 70)
    
    display_cols = [
        'show_title',
        'Category',
        'is_benchmark_classic',
        'title_word_count',
        'LA_AddressableMarket',
        'LA_AddressableMarket_Norm'
    ]
    
    print(result[display_cols].to_string(index=False))
    
    # Verify features were added
    print("\n" + "=" * 70)
    print("Feature Validation:")
    print("=" * 70)
    
    checks = []
    checks.append(('is_benchmark_classic exists', 'is_benchmark_classic' in result.columns))
    checks.append(('title_word_count exists', 'title_word_count' in result.columns))
    checks.append(('LA_AddressableMarket exists', 'LA_AddressableMarket' in result.columns))
    checks.append(('LA_AddressableMarket_Norm exists', 'LA_AddressableMarket_Norm' in result.columns))
    
    # Check if benchmark classics are correctly identified
    checks.append(('Swan Lake is benchmark', result.loc[1, 'is_benchmark_classic'] == 1))
    checks.append(('Romeo and Juliet is benchmark', result.loc[3, 'is_benchmark_classic'] == 1))
    checks.append(('Contemporary Works is not benchmark', result.loc[2, 'is_benchmark_classic'] == 0))
    
    # Check normalization is in [0, 1] range
    if not result['LA_AddressableMarket_Norm'].isna().all():
        norm_in_range = (result['LA_AddressableMarket_Norm'] >= 0).all() and \
                       (result['LA_AddressableMarket_Norm'] <= 1).all()
        checks.append(('Normalized values in [0,1]', norm_in_range))
    
    for check_name, passed in checks:
        status = "✓" if passed else "❌"
        print(f"  {status} {check_name}")
    
    all_passed = all(c[1] for c in checks)
    
    print("=" * 70)
    if all_passed:
        print("✓ All validation checks passed!")
    else:
        print("❌ Some validation checks failed!")
    
    return all_passed


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("LIVE ANALYTICS INTEGRATION TEST SUITE")
    print("=" * 70 + "\n")
    
    test1_passed = test_load_mapping()
    test2_passed = test_add_features()
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  Test 1 (Load Mapping):  {'✓ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"  Test 2 (Add Features):  {'✓ PASSED' if test2_passed else '❌ FAILED'}")
    print("=" * 70)
    
    if test1_passed and test2_passed:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
