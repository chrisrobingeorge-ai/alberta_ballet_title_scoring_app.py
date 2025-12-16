# Title Scoring Mismatch - Debug Report and Fix

## Executive Summary

The mismatch between `title_scoring_helper.py` and `baselines.csv` was caused by **batch-relative normalization** instead of **global normalization**. The issue has been identified and fixed.

## Problem Identified

### Root Cause: Batch-Relative Normalization

The original `title_scoring_helper.py` normalized scores **relative to the current batch** of titles being scored:

1. User enters N titles
2. Fetch raw API values for those N titles
3. Apply `normalize_0_100()` to just those N values
4. Min value in batch → 0, Max value in batch → 100

This meant:
- Scoring `["Cinderella", "Swan Lake"]` together would give one a score of 0 and the other 100
- But in `baselines.csv`, both have scores of 80 (because they're compared against ALL titles)

### Example of the Problem

When scoring a batch of 3 titles together:

| Title | Wiki (Baseline) | Wiki (Batch-Normalized) | Difference |
|-------|-----------------|------------------------|------------|
| Cinderella | 80.00 | 100.00 | +20.00 |
| Swan Lake | 80.00 | 100.00 | +20.00 |
| Giselle | 74.00 | 0.00 | -74.00 |

The batch normalization made Giselle the "minimum" (→ 0) and the other two the "maximum" (→ 100), even though globally all three are high-scoring titles.

## Solution Implemented

### 1. New `normalize_with_reference()` Function

Added a new normalization function that uses the **global distribution** from `baselines.csv`:

```python
def normalize_with_reference(
    values: List[float],
    signal_name: str,
    reference_df: Optional[pd.DataFrame] = None
) -> List[float]:
    """
    Normalize values to 0-100 using a reference distribution from baselines.csv.
    This ensures scores are consistent with the global baseline scores.
    """
```

**Key features:**
- Loads min/max bounds from all titles in `baselines.csv`
- Applies same normalization bounds to new titles
- Clips values to reference range to keep scores within 0-100
- Falls back to batch normalization if reference data unavailable

### 2. UI Option to Choose Normalization Method

Added a dropdown in the helper app to select normalization method:

- **Reference-based (matches baselines.csv)** - Recommended, default
- **Batch-relative (legacy)** - Original behavior, retained for compatibility

### 3. Automatic Reference Distribution Loading

The helper now automatically loads `data/productions/baselines.csv` to get the reference distribution for each signal (wiki, trends, youtube, chartmetric).

## Verification Results

Created `scripts/verify_scoring_fix.py` to test the fix:

✅ **TEST 1: Extreme values map correctly**
- Min values → 0.00
- Max values → 100.00

✅ **TEST 2: Re-normalizing baseline values**
- Maximum error: 2.50 points
- Small differences due to the fact that baselines.csv scores are integers/low-precision decimals
- Within acceptable tolerance (< 3 points)

✅ **TEST 3: Batch independence**
- Same title gets same score regardless of other titles in batch
- Maximum difference: 0.00 (perfect)

## Files Modified

### 1. `title_scoring_helper.py`

**Added:**
- `load_reference_distribution()` - Loads baselines.csv for normalization bounds
- `normalize_with_reference()` - New normalization using reference distribution
- UI dropdown to select normalization method

**Changed:**
- Normalization now uses reference distribution by default
- Added warning if reference data cannot be loaded

### 2. Diagnostic Scripts Created

- `scripts/debug_scoring_mismatch.py` - Comprehensive diagnostic analysis
- `scripts/verify_scoring_fix.py` - Verification tests for the fix

## How to Use the Fix

### In the Streamlit App

1. Run the helper: `streamlit run title_scoring_helper.py`
2. Enter your titles
3. Select **"Reference-based (matches baselines.csv)"** normalization (default)
4. Click "Fetch & Normalize Scores"
5. Scores will now align with `baselines.csv` expectations

### Running Diagnostics

```bash
# See detailed analysis of the problem
python scripts/debug_scoring_mismatch.py

# Verify the fix works correctly
python scripts/verify_scoring_fix.py
```

## Expected Score Ranges (Reference Distribution)

Based on analysis of all 288 titles in `baselines.csv`:

| Signal | Min | Max | Mean | Median | Std Dev |
|--------|-----|-----|------|--------|---------|
| wiki | 0.87 | 100.00 | 57.05 | 57.00 | 17.87 |
| trends | 2.00 | 82.00 | 25.48 | 24.00 | 16.62 |
| youtube | 0.00 | 100.00 | 65.79 | 62.00 | 14.51 |
| chartmetric | 0.00 | 100.00 | 53.67 | 52.50 | 19.19 |

**Note:** The `trends` signal only ranges from 2-82, not 0-100. This is normal - Google Trends data rarely hits the full 0-100 range. The normalization maps this range to 0-100 for consistency.

## Other Potential Issues Investigated

✓ **Precision/Rounding** - baselines.csv has mix of integers and decimals, but this causes < 3 point differences (acceptable)

✓ **Data Sources** - Both historical and reference titles use same methodology

✓ **Random Seed** - No randomness in scoring functions, so this is not an issue

✓ **File Paths** - Verified correct paths to data files

✓ **Column Names** - All column names match between helper and CSV

## Recommendations

### For Production Use

1. **Use reference-based normalization** (now the default) for all new title scoring
2. **Keep batch normalization option** for backwards compatibility, but document that it's deprecated
3. **Update existing workflows** to use the new normalization method

### For Future Improvements

1. **Consider percentile-based normalization** instead of min-max:
   - Less sensitive to outliers
   - More stable as new titles are added
   - Example: Map 5th percentile → 0, 95th percentile → 100

2. **Add data validation** to warn if raw API values are far outside reference range:
   - Could indicate API changes or data quality issues

3. **Cache reference distribution** to avoid re-loading on every scoring operation

## Testing Recommendations

### When Adding New Titles to baselines.csv

1. Score the new titles using the helper with reference normalization
2. Compare helper scores to manually researched scores
3. If discrepancy > 5 points, investigate:
   - Is the raw API value unusual?
   - Has the API changed?
   - Is the title name matching correctly?

### Regression Testing

Run the verification script after any changes to normalization logic:

```bash
python scripts/verify_scoring_fix.py
```

All tests should pass (batch independence is critical).

## Conclusion

The scoring mismatch has been resolved by implementing reference-based normalization that aligns with the global distribution in `baselines.csv`. The fix maintains backwards compatibility while providing more consistent and predictable scores.

**Key takeaway:** When normalizing scores for use in a shared dataset, always normalize against the **global distribution**, not just the current batch of items being scored.
