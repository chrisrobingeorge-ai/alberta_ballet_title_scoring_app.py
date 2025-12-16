#!/usr/bin/env python3
"""
Validation Script for Intent Disambiguation Correction Module

This script validates the complete implementation by running:
1. Module import tests
2. Basic functionality tests
3. Integration tests
4. Demo execution

Run this to verify the implementation is working correctly.

Author: Alberta Ballet Data Science Team
Date: December 2025
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("Intent Disambiguation Correction - Validation Script")
print("=" * 70)

# Test 1: Module Imports
print("\n1. Testing Module Imports...")
print("-" * 70)

try:
    from ml.intent_disambiguation import (
        apply_intent_disambiguation,
        batch_apply_corrections,
        get_penalty_for_category,
        get_all_penalty_categories,
        CATEGORY_PENALTIES,
        MOTIVATION_INDEX_WEIGHT
    )
    print("✓ Core module imports successful")
except ImportError as e:
    print(f"✗ Failed to import core module: {e}")
    sys.exit(1)

try:
    from ml.title_explanation_engine import build_title_explanation
    print("✓ Title explanation engine import successful")
except ImportError as e:
    print(f"✗ Failed to import title explanation engine: {e}")
    sys.exit(1)

try:
    from examples.intent_disambiguation_integration import (
        generate_corrected_explanation,
        process_season_with_corrections,
        compare_original_vs_corrected
    )
    print("✓ Integration examples import successful")
except ImportError as e:
    print(f"✗ Failed to import integration examples: {e}")
    sys.exit(1)

# Test 2: Basic Functionality
print("\n2. Testing Basic Functionality...")
print("-" * 70)

# Test penalty retrieval
print("\nTesting penalty retrieval:")
penalties_tested = {
    "family_classic": 0.20,
    "romantic_tragedy": 0.10,
    "contemporary": 0.0
}

for category, expected in penalties_tested.items():
    actual = get_penalty_for_category(category)
    if actual == expected:
        print(f"  ✓ {category}: {actual*100:.0f}%")
    else:
        print(f"  ✗ {category}: expected {expected}, got {actual}")
        sys.exit(1)

# Test basic correction
print("\nTesting basic correction (Cinderella example):")
cinderella = {
    "Title": "Cinderella",
    "Category": "family_classic",
    "Motivation": 100.0,
    "TicketIndex used": 100.0,
    "EstimatedTickets_Final": 11976
}

corrected = apply_intent_disambiguation(cinderella, verbose=False)

# Validate results
expected_motivation = 80.0
expected_penalty = True
expected_pct = 0.20

if corrected["Motivation_corrected"] == expected_motivation:
    print(f"  ✓ Motivation corrected: {corrected['Motivation_corrected']}")
else:
    print(f"  ✗ Motivation: expected {expected_motivation}, got {corrected['Motivation_corrected']}")
    sys.exit(1)

if corrected["Motivation_penalty_applied"] == expected_penalty:
    print(f"  ✓ Penalty applied: {corrected['Motivation_penalty_applied']}")
else:
    print(f"  ✗ Penalty applied: expected {expected_penalty}, got {corrected['Motivation_penalty_applied']}")
    sys.exit(1)

if corrected["Motivation_penalty_pct"] == expected_pct:
    print(f"  ✓ Penalty percentage: {corrected['Motivation_penalty_pct']*100:.0f}%")
else:
    print(f"  ✗ Penalty %: expected {expected_pct}, got {corrected['Motivation_penalty_pct']}")
    sys.exit(1)

# Validate ticket index adjustment
expected_index_range = (96.5, 97.0)  # Allow small floating point variance
if expected_index_range[0] <= corrected["TicketIndex_corrected"] <= expected_index_range[1]:
    print(f"  ✓ Ticket index corrected: {corrected['TicketIndex_corrected']:.2f}")
else:
    print(f"  ✗ Ticket index: {corrected['TicketIndex_corrected']} outside expected range {expected_index_range}")
    sys.exit(1)

# Validate ticket estimate
expected_tickets_range = (11500, 11600)
if expected_tickets_range[0] <= corrected["EstimatedTickets_corrected"] <= expected_tickets_range[1]:
    print(f"  ✓ Estimated tickets corrected: {corrected['EstimatedTickets_corrected']:,.0f}")
else:
    print(f"  ✗ Tickets: {corrected['EstimatedTickets_corrected']} outside expected range {expected_tickets_range}")
    sys.exit(1)

# Test 3: Batch Processing
print("\n3. Testing Batch Processing...")
print("-" * 70)

batch_titles = [
    {
        "Title": "Cinderella",
        "Category": "family_classic",
        "Motivation": 100.0,
        "TicketIndex used": 100.0,
        "EstimatedTickets_Final": 11976
    },
    {
        "Title": "Romeo & Juliet",
        "Category": "romantic_tragedy",
        "Motivation": 95.0,
        "TicketIndex used": 98.0,
        "EstimatedTickets_Final": 10500
    },
    {
        "Title": "Contemporary Work",
        "Category": "contemporary",
        "Motivation": 75.0,
        "TicketIndex used": 80.0,
        "EstimatedTickets_Final": 8000
    }
]

batch_results = batch_apply_corrections(batch_titles, verbose=False)

if len(batch_results) == len(batch_titles):
    print(f"✓ Batch processed {len(batch_results)} titles")
else:
    print(f"✗ Batch processing: expected {len(batch_titles)} results, got {len(batch_results)}")
    sys.exit(1)

# Validate each result
expected_corrections = [True, True, False]
for idx, (result, expected_corrected) in enumerate(zip(batch_results, expected_corrections)):
    title = result["Title"]
    corrected_flag = result["IntentCorrectionApplied"]
    
    if corrected_flag == expected_corrected:
        status = "corrected" if corrected_flag else "unchanged"
        print(f"  ✓ {title}: {status}")
    else:
        print(f"  ✗ {title}: expected correction={expected_corrected}, got {corrected_flag}")
        sys.exit(1)

# Test 4: Integration with Explanation Engine
print("\n4. Testing Integration with Explanation Engine...")
print("-" * 70)

integration_title = {
    "Title": "Cinderella",
    "Category": "family_classic",
    "Month": "March 2026",
    "Motivation": 100.0,
    "Familiarity": 110.0,
    "SignalOnly": 105.0,
    "TicketIndex used": 100.0,
    "EstimatedTickets_Final": 11976,
    "FutureSeasonalityFactor": 1.05,
    "PrimarySegment": "Family",
    "SecondarySegment": "Tourist",
    "YYC_Singles": 7186,
    "YEG_Singles": 4790,
    "IsRemount": False
}

try:
    narrative, corrected_meta = generate_corrected_explanation(
        integration_title,
        apply_correction=True,
        verbose=False
    )
    
    if isinstance(narrative, str) and len(narrative) > 0:
        print(f"✓ Narrative generated ({len(narrative)} characters)")
    else:
        print("✗ Narrative generation failed")
        sys.exit(1)
    
    if corrected_meta["IntentCorrectionApplied"]:
        print("✓ Correction applied in integration")
    else:
        print("✗ Correction not applied in integration")
        sys.exit(1)
    
except Exception as e:
    print(f"✗ Integration test failed: {e}")
    sys.exit(1)

# Test 5: Comparison Function
print("\n5. Testing Comparison Function...")
print("-" * 70)

try:
    comparison = compare_original_vs_corrected(cinderella, verbose=False)
    
    if "original" in comparison and "corrected" in comparison:
        print("✓ Comparison structure correct")
    else:
        print("✗ Comparison missing required keys")
        sys.exit(1)
    
    orig = comparison["original"]
    corr = comparison["corrected"]
    
    if orig["IntentCorrectionApplied"] == False and corr["IntentCorrectionApplied"] == True:
        print("✓ Original vs corrected flags correct")
    else:
        print("✗ Comparison flags incorrect")
        sys.exit(1)
    
    if orig["Motivation_corrected"] != corr["Motivation_corrected"]:
        print("✓ Motivation values differ as expected")
    else:
        print("✗ Motivation values should differ")
        sys.exit(1)
    
except Exception as e:
    print(f"✗ Comparison test failed: {e}")
    sys.exit(1)

# Test 6: Edge Cases
print("\n6. Testing Edge Cases...")
print("-" * 70)

# Test with zero motivation
zero_test = {
    "Title": "Test",
    "Category": "family_classic",
    "Motivation": 0.0,
    "TicketIndex used": 50.0,
    "EstimatedTickets_Final": 5000
}

try:
    zero_result = apply_intent_disambiguation(zero_test, verbose=False)
    if zero_result["Motivation_corrected"] == 0.0:
        print("✓ Zero motivation handled correctly")
    else:
        print(f"✗ Zero motivation: expected 0.0, got {zero_result['Motivation_corrected']}")
        sys.exit(1)
except Exception as e:
    print(f"✗ Zero motivation test failed: {e}")
    sys.exit(1)

# Test with correction disabled
disabled_test = apply_intent_disambiguation(cinderella, apply_correction=False, verbose=False)
if disabled_test["IntentCorrectionApplied"] == False:
    print("✓ Correction disable flag works")
else:
    print("✗ Correction disable flag failed")
    sys.exit(1)

# Test 7: Constants Validation
print("\n7. Validating Constants...")
print("-" * 70)

if MOTIVATION_INDEX_WEIGHT == 1.0 / 6.0:
    print(f"✓ MOTIVATION_INDEX_WEIGHT = {MOTIVATION_INDEX_WEIGHT:.6f}")
else:
    print(f"✗ MOTIVATION_INDEX_WEIGHT incorrect: {MOTIVATION_INDEX_WEIGHT}")
    sys.exit(1)

if len(CATEGORY_PENALTIES) == 6:
    print(f"✓ CATEGORY_PENALTIES has {len(CATEGORY_PENALTIES)} entries")
else:
    print(f"✗ CATEGORY_PENALTIES should have 6 entries, has {len(CATEGORY_PENALTIES)}")
    sys.exit(1)

# Final Summary
print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)
print("\n✓ All tests passed successfully!")
print("\nImplementation Status:")
print("  ✓ Module imports")
print("  ✓ Penalty retrieval")
print("  ✓ Basic corrections")
print("  ✓ Batch processing")
print("  ✓ Integration with explanation engine")
print("  ✓ Comparison functions")
print("  ✓ Edge case handling")
print("  ✓ Constants validation")

print("\nNext Steps:")
print("  1. Run full test suite: pytest tests/test_intent_disambiguation.py -v")
print("  2. Review documentation: docs/INTENT_DISAMBIGUATION.md")
print("  3. Try integration examples: python -m examples.intent_disambiguation_integration")
print("  4. Integrate into production pipeline")

print("\n" + "=" * 70)
print("Ready for Production Deployment! 🎉")
print("=" * 70)
