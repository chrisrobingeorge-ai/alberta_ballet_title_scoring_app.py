"""
Verify that the scoring fix produces consistent results with baselines.csv.

This script tests the updated normalize_with_reference() function to ensure
it produces scores aligned with the baseline distribution.
"""

import sys
from pathlib import Path

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import numpy as np


def load_baselines():
    """Load the baseline scores from CSV."""
    csv_path = _REPO_ROOT / "data" / "productions" / "baselines.csv"
    return pd.read_csv(csv_path)


def load_reference_distribution():
    """Load reference distribution (copy of function from helper)."""
    try:
        baselines_path = _REPO_ROOT / "data" / "productions" / "baselines.csv"
        if not baselines_path.exists():
            return None
        df = pd.read_csv(baselines_path)
        return df[['title', 'wiki', 'trends', 'youtube', 'spotify']].copy()
    except Exception as exc:
        print(f"Could not load reference distribution: {exc}")
        return None


def normalize_with_reference(values, signal_name, reference_df):
    """Copy of the normalize_with_reference function from title_scoring_helper.py."""
    if not values:
        return []
    
    # Clean input values
    v = [0.0 if (val is None or pd.isna(val)) else float(val) for val in values]
    
    # Get reference values
    if reference_df is not None and signal_name in reference_df.columns:
        ref_values = reference_df[signal_name].dropna().tolist()
        
        if ref_values:
            # Get min and max from reference distribution
            ref_min = min(ref_values)
            ref_max = max(ref_values)
            
            if ref_max > ref_min:
                # Normalize using reference distribution bounds
                normalized = []
                for val in v:
                    # Clip to reference bounds to avoid values outside 0-100
                    clipped = max(ref_min, min(ref_max, val))
                    norm_val = 100.0 * (clipped - ref_min) / (ref_max - ref_min)
                    normalized.append(norm_val)
                return normalized
    
    # Fallback
    return [50.0 for _ in v]


def test_reference_normalization():
    """Test that reference normalization produces consistent results."""
    print("="*80)
    print("VERIFICATION: Reference-Based Normalization")
    print("="*80)
    
    # Load data
    df = load_baselines()
    reference_df = load_reference_distribution()
    
    if reference_df is None:
        print("\n\u274c ERROR: Could not load reference distribution")
        return False
    
    print(f"\nLoaded {len(df)} titles from baselines.csv")
    print(f"Reference distribution contains {len(reference_df)} titles")
    
    # Test 1: Scores at extremes should map to 0 and 100
    print("\n" + "-"*80)
    print("TEST 1: Extreme values map correctly")
    print("-"*80)
    
    all_passed = True
    for signal in ['wiki', 'trends', 'youtube', 'spotify']:
        ref_vals = reference_df[signal].dropna().tolist()
        ref_min, ref_max = min(ref_vals), max(ref_vals)
        
        # Test minimum value
        test_vals = [ref_min]
        normalized = normalize_with_reference(test_vals, signal, reference_df)
        expected = 0.0
        actual = normalized[0]
        passed = abs(actual - expected) < 0.01
        
        status = '✓' if passed else '❌'
        print(f"{signal:8s} min: expected={expected:6.2f}, actual={actual:6.2f} {status}")
        all_passed = all_passed and passed
        
        # Test maximum value
        test_vals = [ref_max]
        normalized = normalize_with_reference(test_vals, signal, reference_df)
        expected = 100.0
        actual = normalized[0]
        passed = abs(actual - expected) < 0.01
        
        status = '✓' if passed else '❌'
        print(f"{signal:8s} max: expected={expected:6.2f}, actual={actual:6.2f} {status}")
        all_passed = all_passed and passed
    
    # Test 2: Re-normalizing baseline values should give same results
    print("\n" + "-"*80)
    print("TEST 2: Re-normalizing baseline values")
    print("-"*80)
    print("(Should produce identical or very close scores)")
    
    test_titles = ["Cinderella", "Swan Lake", "Giselle", "The Nutcracker"]
    test_data = df[df['title'].isin(test_titles)].copy()
    
    if len(test_data) == 0:
        print("\u274c No test titles found in baselines.csv")
        return False
    
    max_error = 0.0
    for signal in ['wiki', 'trends', 'youtube', 'spotify']:
        original_scores = test_data[signal].tolist()
        renormalized = normalize_with_reference(original_scores, signal, reference_df)
        
        for i, (orig, renorm) in enumerate(zip(original_scores, renormalized)):
            error = abs(renorm - orig)
            max_error = max(max_error, error)
            status = '✓' if error < 1.0 else '\u274c'
            if error >= 1.0:  # Only show errors
                title_str = test_data.iloc[i]['title']
                print(f"  {title_str:30s} {signal:8s}: {orig:6.2f} → {renorm:6.2f} (error: {error:+.2f}) {status}")
    
    print(f"\nMaximum error: {max_error:.2f}")
    if max_error < 1.0:
        print("✓ All scores within 1.0 points (acceptable)")
        all_passed = all_passed and True
    else:
        print("\u274c Some scores differ by more than 1.0 point")
        all_passed = False
    
    # Test 3: Batch independence
    print("\n" + "-"*80)
    print("TEST 3: Batch independence")
    print("-"*80)
    print("(Same title should get same score regardless of other titles in batch)")
    
    test_title = "Cinderella"
    test_row = df[df['title'] == test_title]
    
    if len(test_row) == 0:
        print(f"\u274c {test_title} not found in baselines.csv")
        return False
    
    # Score Cinderella alone
    batch1_scores = {}
    for signal in ['wiki', 'trends', 'youtube', 'spotify']:
        val = test_row[signal].values[0]
        normalized = normalize_with_reference([val], signal, reference_df)
        batch1_scores[signal] = normalized[0]
    
    # Score Cinderella with other titles
    other_titles = ["Giselle", "Swan Lake", "The Nutcracker"]
    batch2_df = df[df['title'].isin([test_title] + other_titles)].copy()
    batch2_scores = {}
    
    for signal in ['wiki', 'trends', 'youtube', 'spotify']:
        vals = batch2_df[signal].tolist()
        normalized = normalize_with_reference(vals, signal, reference_df)
        # Get Cinderella's score (it's the first one since we filtered for it first)
        cinderella_idx = batch2_df[batch2_df['title'] == test_title].index[0]
        batch2_idx = batch2_df.index.tolist().index(cinderella_idx)
        batch2_scores[signal] = normalized[batch2_idx]
    
    print(f"\nScoring '{test_title}':")
    print(f"  {'Signal':8s} | Alone    | With Others | Difference")
    print("  " + "-"*50)
    
    max_diff = 0.0
    for signal in ['wiki', 'trends', 'youtube', 'spotify']:
        alone = batch1_scores[signal]
        together = batch2_scores[signal]
        diff = abs(together - alone)
        max_diff = max(max_diff, diff)
        status = '✓' if diff < 0.01 else '❌'
        print(f"  {signal:8s} | {alone:6.2f}   | {together:6.2f}      | {diff:+6.2f} {status}")
    
    print(f"\nMaximum difference: {max_diff:.2f}")
    if max_diff < 0.01:
        print("✓ Scores are batch-independent (as expected)")
        all_passed = all_passed and True
    else:
        print("❌ Scores vary by batch composition (unexpected)")
        all_passed = False
    
    # Summary
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("="*80)
        print("\nThe reference-based normalization is working correctly.")
        print("Scores should now match baselines.csv when using the same raw values.")
    else:
        print("❌ SOME TESTS FAILED")
        print("="*80)
        print("\nPlease review the test output above to identify issues.")
    
    return all_passed


def main():
    """Run verification tests."""
    success = test_reference_normalization()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
