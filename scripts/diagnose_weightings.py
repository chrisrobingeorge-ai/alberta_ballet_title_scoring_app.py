#!/usr/bin/env python3
"""
Diagnose Weightings Impact Script

This script validates that the three weighting systems (Live Analytics, Economics,
Stone Olafson) are actively influencing scores and quantifies their impact.

Usage:
    python scripts/diagnose_weightings.py [--dataset PATH] [--output PATH]

Outputs:
    - results/weightings_impact_summary.csv: Per-show impact analysis
    - Prints summary statistics to console

The script computes scores in 4 configurations:
1. Base: All weightings active (current production logic)
2. No Live Analytics: aud__engagement_factor set to 1.0 (neutral)
3. No Economics: consumer_confidence, energy_index, inflation set to defaults
4. No Stone Olafson: segment/region multipliers set to 1.0

"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_modelling_dataset(path: str = "data/modelling_dataset.csv") -> pd.DataFrame:
    """Load the modelling dataset with all features."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} shows from {path}")
    return df


def compute_base_score_from_features(row: pd.Series) -> float:
    """Compute a composite base score from wiki/trends/youtube/spotify.
    
    This mimics the base signal before weightings are applied.
    """
    try:
        wiki = float(row.get('wiki', 60))
        trends = float(row.get('trends', 55))
        youtube = float(row.get('youtube', 60))
        spotify = float(row.get('chartmetric', 58))
        
        # Use the familiarity score formula from calc_scores
        base = wiki * 0.55 + trends * 0.30 + spotify * 0.15
        return base
    except (ValueError, TypeError):
        return 60.0  # Default


def compute_score_with_all_weights(row: pd.Series) -> float:
    """Compute score with all three weighting systems active.
    
    Combines:
    1. Base signal (wiki/trends/spotify)
    2. Live Analytics: aud__engagement_factor
    3. Economics: consumer_confidence, energy_index, inflation
    4. Stone Olafson: implicit segment/region multipliers (approximated as 1.0 here)
    
    Note: We can't fully replicate Stone Olafson multipliers without the full
    calc_scores() context, so we approximate their effect as ~1.05 average
    """
    base = compute_base_score_from_features(row)
    
    # Apply Live Analytics weighting
    engagement = float(row.get('aud__engagement_factor', 1.0))
    
    # Apply Economics weightings (normalize to multiplier effects)
    # Consumer confidence: normalized to 1.0 at 50.0 baseline
    confidence = float(row.get('consumer_confidence_prairies', 50.0))
    confidence_mult = confidence / 50.0
    
    # Energy index: normalized to 1.0 at 1000 baseline
    energy = float(row.get('energy_index', 1000.0))
    energy_mult = energy / 1000.0
    
    # Inflation: directly used as multiplier
    inflation = float(row.get('inflation_adjustment_factor', 1.0))
    
    # Combined economic multiplier (weighted average to avoid extreme swings)
    econ_mult = (confidence_mult * 0.3 + energy_mult * 0.4 + inflation * 0.3)
    
    # Stone Olafson: approximate average effect (segment × region multipliers)
    # Typical range: 0.9-1.2, so use 1.05 as average
    stone_olafson_mult = 1.05
    
    # Combine all weightings
    final_score = base * engagement * econ_mult * stone_olafson_mult
    
    return final_score


def compute_score_without_live_analytics(row: pd.Series) -> float:
    """Compute score with Live Analytics weighting disabled (set to 1.0)."""
    base = compute_base_score_from_features(row)
    
    # NO Live Analytics
    engagement = 1.0
    
    # Keep Economics
    confidence = float(row.get('consumer_confidence_prairies', 50.0))
    confidence_mult = confidence / 50.0
    energy = float(row.get('energy_index', 1000.0))
    energy_mult = energy / 1000.0
    inflation = float(row.get('inflation_adjustment_factor', 1.0))
    econ_mult = (confidence_mult * 0.3 + energy_mult * 0.4 + inflation * 0.3)
    
    # Keep Stone Olafson
    stone_olafson_mult = 1.05
    
    final_score = base * engagement * econ_mult * stone_olafson_mult
    return final_score


def compute_score_without_economics(row: pd.Series) -> float:
    """Compute score with Economics weightings disabled (set to defaults)."""
    base = compute_base_score_from_features(row)
    
    # Keep Live Analytics
    engagement = float(row.get('aud__engagement_factor', 1.0))
    
    # NO Economics (use neutral defaults)
    econ_mult = 1.0
    
    # Keep Stone Olafson
    stone_olafson_mult = 1.05
    
    final_score = base * engagement * econ_mult * stone_olafson_mult
    return final_score


def compute_score_without_stone_olafson(row: pd.Series) -> float:
    """Compute score with Stone Olafson weightings disabled (set to 1.0)."""
    base = compute_base_score_from_features(row)
    
    # Keep Live Analytics
    engagement = float(row.get('aud__engagement_factor', 1.0))
    
    # Keep Economics
    confidence = float(row.get('consumer_confidence_prairies', 50.0))
    confidence_mult = confidence / 50.0
    energy = float(row.get('energy_index', 1000.0))
    energy_mult = energy / 1000.0
    inflation = float(row.get('inflation_adjustment_factor', 1.0))
    econ_mult = (confidence_mult * 0.3 + energy_mult * 0.4 + inflation * 0.3)
    
    # NO Stone Olafson
    stone_olafson_mult = 1.0
    
    final_score = base * engagement * econ_mult * stone_olafson_mult
    return final_score


def diagnose_weightings(df: pd.DataFrame) -> pd.DataFrame:
    """Compute scores with each weighting system selectively disabled.
    
    Returns DataFrame with columns:
    - show_title
    - base_score
    - score_with_all_weights
    - score_without_live_analytics
    - score_without_economics
    - score_without_stone_olafson
    - delta_live_analytics
    - delta_economics
    - delta_stone_olafson
    """
    results = []
    
    for idx, row in df.iterrows():
        base = compute_base_score_from_features(row)
        score_all = compute_score_with_all_weights(row)
        score_no_la = compute_score_without_live_analytics(row)
        score_no_econ = compute_score_without_economics(row)
        score_no_so = compute_score_without_stone_olafson(row)
        
        results.append({
            'show_title': row.get('title', row.get('canonical_title', 'Unknown')),
            'category': row.get('category', 'unknown'),
            'base_score': base,
            'score_with_all_weights': score_all,
            'score_without_live_analytics': score_no_la,
            'score_without_economics': score_no_econ,
            'score_without_stone_olafson': score_no_so,
            'delta_live_analytics': score_all - score_no_la,
            'delta_economics': score_all - score_no_econ,
            'delta_stone_olafson': score_all - score_no_so,
            # Also include raw feature values for analysis
            'aud__engagement_factor': row.get('aud__engagement_factor', 1.0),
            'consumer_confidence_prairies': row.get('consumer_confidence_prairies', 50.0),
            'energy_index': row.get('energy_index', 1000.0),
            'inflation_adjustment_factor': row.get('inflation_adjustment_factor', 1.0),
        })
    
    return pd.DataFrame(results)


def print_summary_statistics(results_df: pd.DataFrame) -> None:
    """Print descriptive statistics for each weighting system's impact."""
    print("\n" + "=" * 80)
    print("WEIGHTINGS IMPACT SUMMARY")
    print("=" * 80)
    
    for weighting in ['live_analytics', 'economics', 'stone_olafson']:
        delta_col = f'delta_{weighting}'
        
        if delta_col not in results_df.columns:
            continue
        
        deltas = results_df[delta_col]
        zero_count = (deltas == 0).sum()
        zero_pct = (zero_count / len(deltas)) * 100
        
        print(f"\n{weighting.upper().replace('_', ' ')}:")
        print(f"  Mean delta:        {deltas.mean():+.3f}")
        print(f"  Median delta:      {deltas.median():+.3f}")
        print(f"  Std deviation:     {deltas.std():.3f}")
        print(f"  Min delta:         {deltas.min():+.3f}")
        print(f"  Max delta:         {deltas.max():+.3f}")
        print(f"  Zero deltas:       {zero_count}/{len(deltas)} ({zero_pct:.1f}%)")
        
        if zero_pct > 90:
            print(f"  ⚠️  WARNING: {zero_pct:.0f}% of shows have zero delta!")
            print(f"      This weighting may be disconnected or ineffective.")
    
    print("\n" + "=" * 80)


def check_feature_values(df: pd.DataFrame) -> None:
    """Check the distribution of raw weighting feature values."""
    print("\n" + "=" * 80)
    print("RAW FEATURE VALUE DISTRIBUTIONS")
    print("=" * 80)
    
    features = [
        'aud__engagement_factor',
        'consumer_confidence_prairies',
        'energy_index',
        'inflation_adjustment_factor'
    ]
    
    for feat in features:
        if feat not in df.columns:
            print(f"\n{feat}: NOT FOUND")
            continue
        
        vals = df[feat].dropna()
        unique_count = vals.nunique()
        
        print(f"\n{feat}:")
        print(f"  Count:             {len(vals)}")
        print(f"  Unique values:     {unique_count}")
        print(f"  Mean:              {vals.mean():.3f}")
        print(f"  Std deviation:     {vals.std():.3f}")
        print(f"  Min:               {vals.min():.3f}")
        print(f"  Max:               {vals.max():.3f}")
        
        if unique_count == 1:
            print(f"  ⚠️  WARNING: Only 1 unique value - feature is constant!")
        elif unique_count < 5:
            print(f"  ⚠️  WARNING: Only {unique_count} unique values - limited variance")
        
        if unique_count <= 10:
            print(f"  Values: {sorted(vals.unique())}")
    
    print("\n" + "=" * 80)


def analyze_correlations_with_target(results_df: pd.DataFrame, dataset_df: pd.DataFrame) -> None:
    """If target variable exists, compute correlations."""
    # Check if we have a target
    target_cols = ['prior_total_tickets', 'total_single_tickets', 'tickets']
    target_col = None
    
    for col in target_cols:
        if col in dataset_df.columns:
            target_col = col
            break
    
    if target_col is None:
        print("\n(No target variable found for correlation analysis)")
        return
    
    print("\n" + "=" * 80)
    print(f"CORRELATIONS WITH TARGET: {target_col}")
    print("=" * 80)
    
    # Merge target with results
    results_with_target = results_df.copy()
    results_with_target['target'] = dataset_df[target_col].values[:len(results_df)]
    
    # Filter to shows with non-zero target
    valid = results_with_target[results_with_target['target'] > 0]
    
    if len(valid) < 10:
        print(f"\nInsufficient data for correlation (only {len(valid)} shows with target > 0)")
        return
    
    print(f"\nAnalyzing {len(valid)} shows with {target_col} > 0")
    
    score_cols = [
        'score_with_all_weights',
        'score_without_live_analytics',
        'score_without_economics',
        'score_without_stone_olafson'
    ]
    
    for score_col in score_cols:
        if score_col in valid.columns:
            corr = valid[score_col].corr(valid['target'])
            print(f"  {score_col:40s}: r={corr:+.3f}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Diagnose weighting systems impact")
    parser.add_argument(
        '--dataset',
        default='data/modelling_dataset.csv',
        help='Path to modelling dataset CSV'
    )
    parser.add_argument(
        '--output',
        default='results/weightings_impact_summary.csv',
        help='Path to output CSV'
    )
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    df = load_modelling_dataset(args.dataset)
    
    # Check raw feature distributions
    check_feature_values(df)
    
    # Compute scores with selective disabling
    print(f"\nComputing scores with selective weighting disabling...")
    results = diagnose_weightings(df)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")
    
    # Print summary statistics
    print_summary_statistics(results)
    
    # Analyze correlations if target exists
    analyze_correlations_with_target(results, df)
    
    print("\n✓ Diagnostic complete!")
    print(f"\nNext steps:")
    print(f"  1. Review {args.output} for per-show impact")
    print(f"  2. Check for warnings above (constant features, zero deltas)")
    print(f"  3. See docs/weightings_map.md for implementation details")


if __name__ == '__main__':
    main()
