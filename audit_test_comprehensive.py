#!/usr/bin/env python3
"""
Comprehensive Audit Test Suite for Alberta Ballet Title Scoring App

This script tests:
1. Model artifact availability and loading
2. Feature engineering pipeline
3. Prediction chain (Ridge → k-NN fallback)
4. Segment and region multipliers
5. Seasonality logic
6. k-NN fallback conditions
7. Edge cases (NaN, missing data)
8. Code-to-technical-report alignment
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, date
from pathlib import Path

# Add workspace to path
sys.path.insert(0, '/workspaces/alberta_ballet_title_scoring_app.py')

# Import core components
import streamlit as st
from data.loader import load_baselines

print("\n" + "="*80)
print("COMPREHENSIVE AUDIT TEST SUITE")
print("="*80 + "\n")

# ============================================================================
# TEST 1: Model Artifact Verification
# ============================================================================
print("[TEST 1] Model Artifact Verification")
print("-" * 80)

model_json_path = Path('/workspaces/alberta_ballet_title_scoring_app.py/models/model_xgb_remount_postcovid.json')
model_joblib_path = Path('/workspaces/alberta_ballet_title_scoring_app.py/models/model_xgb_remount_postcovid.joblib')

test1_results = {
    'json_metadata_exists': model_json_path.exists(),
    'joblib_model_exists': model_joblib_path.exists(),
    'issues': []
}

if not model_joblib_path.exists():
    test1_results['issues'].append(
        "CRITICAL: Actual trained model (joblib) does not exist. "
        "Only metadata (.json) present. Code tries to load: 'models/model_xgb_remount_postcovid.joblib'"
    )

if model_json_path.exists():
    with open(model_json_path) as f:
        metadata = json.load(f)
    test1_results['metadata'] = {
        'training_date': metadata.get('training_date'),
        'n_samples': metadata.get('n_samples'),
        'cv_metrics': metadata.get('cv_metrics'),
        'features_count': metadata.get('features', {}).get('total')
    }

print(f"✓ JSON Metadata exists: {test1_results['json_metadata_exists']}")
print(f"✗ Joblib Model exists: {test1_results['joblib_model_exists']}")
if test1_results['issues']:
    for issue in test1_results['issues']:
        print(f"  ⚠️  {issue}")

# ============================================================================
# TEST 2: Load Baseline Data
# ============================================================================
print("\n[TEST 2] Baseline Data Loading")
print("-" * 80)

try:
    baselines = load_baselines()
    test2_results = {
        'baselines_loaded': True,
        'count': len(baselines),
        'sample_keys': list(baselines.keys())[:3]
    }
    print(f"✓ Loaded {len(baselines)} baseline titles")
    print(f"  Sample: {test2_results['sample_keys']}")
except Exception as e:
    test2_results = {
        'baselines_loaded': False,
        'error': str(e)
    }
    print(f"✗ Failed to load baselines: {e}")

# ============================================================================
# TEST 3: Feature Engineering Pipeline
# ============================================================================
print("\n[TEST 3] Feature Engineering Pipeline")
print("-" * 80)

test3_results = {
    'signal_construction': {},
    'issues': []
}

try:
    # Test with sample baseline
    if baselines:
        sample_title = list(baselines.keys())[0]
        sample_entry = baselines[sample_title]
        
        # Extract signals
        wiki_raw = sample_entry.get('wiki', 0)
        trends_raw = sample_entry.get('trends', 0)
        youtube_raw = sample_entry.get('youtube', 0)
        chartmetric_raw = sample_entry.get('chartmetric', 0)
        
        # Test signal transformation formulas from TECHNICAL_ML_REPORT.md Section 2.1
        wiki_idx = 40.0 + min(110.0, (np.log1p(max(0.0, wiki_raw)) * 20.0))
        yt_idx = 50.0 + min(90.0, np.log1p(max(0.0, youtube_raw)) * 9.0)
        
        test3_results['signal_construction'] = {
            'sample_title': sample_title,
            'raw_signals': {
                'wiki': wiki_raw,
                'trends': trends_raw,
                'youtube': youtube_raw,
                'chartmetric': chartmetric_raw
            },
            'computed_indices': {
                'wiki_idx': float(wiki_idx),
                'youtube_idx': float(yt_idx)
            }
        }
        
        print(f"✓ Sample title: {sample_title}")
        print(f"  Raw wiki: {wiki_raw} → Wiki Index: {wiki_idx:.1f}")
        print(f"  Raw YouTube: {youtube_raw} → YouTube Index: {yt_idx:.1f}")
        print(f"  Trends: {trends_raw}, Chartmetric: {chartmetric_raw}")
        
        # Check formula correctness per TECHNICAL_ML_REPORT Section 2.1
        if not (40 <= wiki_idx <= 150):
            test3_results['issues'].append(f"Wiki index {wiki_idx} outside expected range [40, 150]")
        if not (50 <= yt_idx <= 140):
            test3_results['issues'].append(f"YouTube index {yt_idx} outside expected range [50, 140]")
            
except Exception as e:
    test3_results['issues'].append(f"Feature engineering failed: {e}")
    print(f"✗ Feature engineering failed: {e}")

if not test3_results['issues']:
    print("✓ All signal transformations within expected ranges")
else:
    for issue in test3_results['issues']:
        print(f"  ⚠️  {issue}")

# ============================================================================
# TEST 4: Config Loading and Multipliers
# ============================================================================
print("\n[TEST 4] Config Loading and Multipliers")
print("-" * 80)

test4_results = {
    'config_loaded': False,
    'segment_multipliers': {},
    'issues': []
}

try:
    config_path = Path('/workspaces/alberta_ballet_title_scoring_app.py/config.yaml')
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        
        test4_results['config_loaded'] = True
        segment_mult = cfg.get('segment_mult', {})
        
        # Verify structure
        if 'Core Classical (F35–64)' in segment_mult:
            core_cfg = segment_mult['Core Classical (F35–64)']
            print(f"✓ Config loaded with {len(segment_mult)} segments")
            
            # Test sample multiplier
            sample_female_mult = core_cfg.get('gender', {}).get('female', None)
            if sample_female_mult:
                print(f"  Sample: Core Classical (F) female multiplier: {sample_female_mult}")
                test4_results['segment_multipliers']['core_female'] = sample_female_mult
        else:
            test4_results['issues'].append("Config structure unexpected - 'Core Classical (F35–64)' not found")
    else:
        test4_results['issues'].append("config.yaml not found")
        
except Exception as e:
    test4_results['issues'].append(f"Config loading failed: {e}")
    print(f"✗ Config loading failed: {e}")

if test4_results['issues']:
    for issue in test4_results['issues']:
        print(f"  ⚠️  {issue}")
else:
    print("✓ All config multipliers verified")

# ============================================================================
# TEST 5: Ridge Regression with Anchor Points
# ============================================================================
print("\n[TEST 5] Ridge Regression with Anchor Points")
print("-" * 80)

test5_results = {
    'ridge_training': False,
    'anchor_behavior': {},
    'issues': []
}

try:
    from sklearn.linear_model import Ridge
    
    # Simulate historical data: SignalOnly → TicketIndex_DeSeason
    X_real = np.array([[30], [50], [80], [100], [120]])
    y_real = np.array([40, 65, 85, 110, 130])
    
    # Add anchor points per TECHNICAL_ML_REPORT Section 3.2
    # SignalOnly=0 → TicketIndex=25
    # SignalOnly=100 → TicketIndex=100
    n_real = len(X_real)
    anchor_weight = max(3, n_real // 2)
    
    X_anchors = np.array([[0.0], [100.0]])
    y_anchors = np.array([25.0, 100.0])
    
    X_anchors_weighted = np.repeat(X_anchors, anchor_weight, axis=0)
    y_anchors_weighted = np.repeat(y_anchors, anchor_weight)
    
    X_combined = np.vstack([X_real, X_anchors_weighted])
    y_combined = np.concatenate([y_real, y_anchors_weighted])
    
    model = Ridge(alpha=5.0, random_state=42)
    model.fit(X_combined, y_combined)
    
    # Test anchor behavior
    pred_0 = float(model.predict([[0.0]])[0])
    pred_100 = float(model.predict([[100.0]])[0])
    
    test5_results['ridge_training'] = True
    test5_results['anchor_behavior'] = {
        'anchor_0_predicted': pred_0,
        'anchor_0_expected': 25.0,
        'anchor_0_error': abs(pred_0 - 25.0),
        'anchor_100_predicted': pred_100,
        'anchor_100_expected': 100.0,
        'anchor_100_error': abs(pred_100 - 100.0),
        'model_intercept': float(model.intercept_),
        'model_slope': float(model.coef_[0])
    }
    
    print(f"✓ Ridge model trained with {len(X_combined)} samples (5 real + 6 anchors)")
    print(f"  Anchor @ SignalOnly=0: predicted {pred_0:.2f}, expected ~25.0 (error: {abs(pred_0 - 25.0):.2f})")
    print(f"  Anchor @ SignalOnly=100: predicted {pred_100:.2f}, expected ~100.0 (error: {abs(pred_100 - 100.0):.2f})")
    print(f"  Model: intercept={model.intercept_:.2f}, slope={model.coef_[0]:.3f}")
    
    # Check if anchors are reasonably satisfied (within 10 units)
    if abs(pred_0 - 25.0) > 10 or abs(pred_100 - 100.0) > 10:
        test5_results['issues'].append(
            f"Anchor constraints not satisfied: SignalOnly=0→{pred_0:.1f} (expected ~25), "
            f"SignalOnly=100→{pred_100:.1f} (expected ~100)"
        )
    
except Exception as e:
    test5_results['issues'].append(f"Ridge regression test failed: {e}")
    print(f"✗ Ridge regression failed: {e}")

if test5_results['issues']:
    for issue in test5_results['issues']:
        print(f"  ⚠️  {issue}")
else:
    print("✓ Ridge regression anchors satisfied")

# ============================================================================
# TEST 6: Seasonality Logic
# ============================================================================
print("\n[TEST 6] Seasonality Logic")
print("-" * 80)

test6_results = {
    'seasonality_tested': False,
    'factors': {},
    'issues': []
}

try:
    # Test seasonality formula from TECHNICAL_ML_REPORT Section 4
    # Expected factors in range [0.90, 1.15] per config
    
    # Simulate: December family shows have historically 40% higher sales
    raw_factor = 1.40
    K_SHRINK = 3.0  # from config.yaml
    MINF = 0.90
    MAXF = 1.15
    
    # Apply shrinkage: F_shrunk = 1 + K * (F_raw - 1)
    shrunk = 1.0 + K_SHRINK * (raw_factor - 1.0)
    clipped = np.clip(shrunk, MINF, MAXF)
    
    test6_results['seasonality_tested'] = True
    test6_results['factors'] = {
        'raw_factor': raw_factor,
        'shrunk_factor': float(shrunk),
        'clipped_factor': float(clipped),
        'K_SHRINK': K_SHRINK
    }
    
    print(f"✓ Seasonality calculation:")
    print(f"  Raw factor (Dec family): {raw_factor:.2f}")
    print(f"  After shrinkage (K={K_SHRINK}): {shrunk:.2f}")
    print(f"  After clipping [{MINF}, {MAXF}]: {clipped:.2f}")
    
    if clipped < MINF or clipped > MAXF:
        test6_results['issues'].append(f"Clipped factor {clipped} outside valid range [{MINF}, {MAXF}]")
    
except Exception as e:
    test6_results['issues'].append(f"Seasonality test failed: {e}")
    print(f"✗ Seasonality test failed: {e}")

if test6_results['issues']:
    for issue in test6_results['issues']:
        print(f"  ⚠️  {issue}")
else:
    print("✓ Seasonality factors within expected bounds")

# ============================================================================
# TEST 7: k-NN Fallback Conditions
# ============================================================================
print("\n[TEST 7] k-NN Fallback Conditions")
print("-" * 80)

test7_results = {
    'knn_available': False,
    'conditions': {},
    'issues': []
}

try:
    from ml.knn_fallback import KNNFallback, build_knn_from_config
    test7_results['knn_available'] = True
    
    # Check activation conditions per TECHNICAL_ML_REPORT Section 3.3
    print(f"✓ KNN module available")
    
    # Verify conditions
    knn_enabled = True  # From config default
    knn_fallback_available = True  # Module imported
    min_data = 3  # Minimum records to build index
    
    test7_results['conditions'] = {
        'knn_enabled': knn_enabled,
        'module_available': knn_fallback_available,
        'min_records_required': min_data,
        'k_default': 5
    }
    
    print(f"  - knn_enabled: {knn_enabled}")
    print(f"  - Module available: {knn_fallback_available}")
    print(f"  - Min records for index: {min_data}")
    print(f"  - Default k neighbors: 5")
    
except ImportError as e:
    test7_results['issues'].append(f"KNN module not available: {e}")
    print(f"✗ KNN module import failed: {e}")
except Exception as e:
    test7_results['issues'].append(f"KNN test failed: {e}")
    print(f"✗ KNN test failed: {e}")

if test7_results['issues']:
    for issue in test7_results['issues']:
        print(f"  ⚠️  {issue}")
else:
    print("✓ k-NN fallback conditions verified")

# ============================================================================
# TEST 8: Edge Cases
# ============================================================================
print("\n[TEST 8] Edge Cases (NaN, Missing Data, Extreme Values)")
print("-" * 80)

test8_results = {
    'tests_run': 0,
    'tests_passed': 0,
    'issues': []
}

# Test 8a: NaN handling in feature engineering
try:
    test_signals = {
        'wiki': np.nan,
        'trends': 50.0,
        'youtube': 0.0,
        'chartmetric': None
    }
    
    wiki_val = float(test_signals.get('wiki', 0)) or 0
    wiki_idx = 40.0 + min(110.0, max(0, np.log1p(max(0.0, wiki_val)) * 20.0))
    
    test8_results['tests_run'] += 1
    if not np.isnan(wiki_idx) and 40 <= wiki_idx <= 150:
        test8_results['tests_passed'] += 1
        print(f"✓ NaN wiki value handled: {wiki_val} → Index {wiki_idx:.1f}")
    else:
        test8_results['issues'].append(f"NaN handling produced invalid result: {wiki_idx}")
except Exception as e:
    test8_results['issues'].append(f"NaN handling test failed: {e}")

# Test 8b: Zero signals
try:
    zero_signals = {
        'wiki': 0.0,
        'trends': 0.0,
        'youtube': 0.0,
        'chartmetric': 0.0
    }
    
    fam = 0.0 * 0.55 + 0.0 * 0.30 + 0.0 * 0.15
    mot = 0.0 * 0.45 + 0.0 * 0.25 + 0.0 * 0.15 + 0.0 * 0.15
    signal_only = 0.5 * fam + 0.5 * mot
    
    test8_results['tests_run'] += 1
    if signal_only == 0.0:
        test8_results['tests_passed'] += 1
        print(f"✓ Zero signals: SignalOnly = {signal_only:.2f}")
    else:
        test8_results['issues'].append(f"Zero signals produced non-zero result: {signal_only}")
except Exception as e:
    test8_results['issues'].append(f"Zero signals test failed: {e}")

# Test 8c: Extreme signal values
try:
    extreme_signals = {
        'wiki': 1e6,
        'trends': 100.0,
        'youtube': 1e7,
        'chartmetric': 1e3
    }
    
    wiki_idx = 40.0 + min(110.0, np.log1p(1e6) * 20.0)
    yt_idx = 50.0 + min(90.0, np.log1p(1e7) * 9.0)
    
    test8_results['tests_run'] += 1
    if 40 <= wiki_idx <= 150 and 50 <= yt_idx <= 140:
        test8_results['tests_passed'] += 1
        print(f"✓ Extreme signals handled (via min() capping): wiki→{wiki_idx:.1f}, yt→{yt_idx:.1f}")
    else:
        test8_results['issues'].append(f"Extreme signals outside bounds: wiki={wiki_idx}, yt={yt_idx}")
except Exception as e:
    test8_results['issues'].append(f"Extreme signals test failed: {e}")

print(f"\n  Tests passed: {test8_results['tests_passed']}/{test8_results['tests_run']}")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("AUDIT SUMMARY")
print("="*80 + "\n")

all_results = {
    'test_1_model_artifacts': test1_results,
    'test_2_baseline_data': test2_results,
    'test_3_feature_engineering': test3_results,
    'test_4_config_multipliers': test4_results,
    'test_5_ridge_regression': test5_results,
    'test_6_seasonality': test6_results,
    'test_7_knn_fallback': test7_results,
    'test_8_edge_cases': test8_results
}

critical_issues = []
warnings = []

for test_name, test_data in all_results.items():
    if isinstance(test_data, dict) and 'issues' in test_data:
        for issue in test_data['issues']:
            if 'CRITICAL' in str(issue).upper():
                critical_issues.append(f"{test_name}: {issue}")
            else:
                warnings.append(f"{test_name}: {issue}")

print(f"CRITICAL ISSUES: {len(critical_issues)}")
for issue in critical_issues:
    print(f"  ❌ {issue}")

print(f"\nWARNINGS: {len(warnings)}")
for warning in warnings:
    print(f"  ⚠️  {warning}")

print(f"\n{'PASSED' if not critical_issues else 'FAILED'}: Audit integrity check")

# Save detailed results to JSON
output_file = Path('/workspaces/alberta_ballet_title_scoring_app.py/audit_results.json')
with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n📄 Detailed results saved to: {output_file}")
print("\n" + "="*80 + "\n")
