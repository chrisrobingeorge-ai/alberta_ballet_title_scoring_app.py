# Quick Fix Guide: Title Scoring Mismatch

## Problem
Scores from `title_scoring_helper.py` don't match `baselines.csv`.

## Cause
**Batch-relative normalization** - scores were normalized only within the current batch instead of against the full baseline distribution.

## Solution Applied

### ✅ Code Changes
1. Added `normalize_with_reference()` function in `title_scoring_helper.py`
2. Added UI dropdown to select normalization method
3. Reference-based normalization is now the **default**

### ✅ How to Use
1. Run: `streamlit run title_scoring_helper.py`
2. Enter titles
3. Ensure **"Reference-based (matches baselines.csv)"** is selected (default)
4. Click "Fetch & Normalize Scores"

### ✅ Verification
Run these scripts to verify the fix:

```bash
# Detailed diagnostic analysis
python scripts/debug_scoring_mismatch.py

# Verify fix works correctly
python scripts/verify_scoring_fix.py
```

## Key Differences

| Method | Behavior | Use Case |
|--------|----------|----------|
| **Reference-based** (NEW) | Normalizes against all titles in baselines.csv | **Production use - ensures consistency** |
| Batch-relative (OLD) | Normalizes only within current batch | Legacy compatibility only |

## Expected Results

With reference-based normalization:
- Scores are consistent with `baselines.csv`
- Same title gets same score regardless of batch
- Scores remain stable as you add more titles

**Example:**
- Scoring "Cinderella" alone → wiki: 80
- Scoring "Cinderella" with others → wiki: 80 (same!)

## Files Changed
- ✏️ `title_scoring_helper.py` - Added reference normalization
- ➕ `scripts/debug_scoring_mismatch.py` - Diagnostic tool
- ➕ `scripts/verify_scoring_fix.py` - Verification tests
- ➕ `SCORING_MISMATCH_DEBUG_REPORT.md` - Detailed report

## Need Help?
See `SCORING_MISMATCH_DEBUG_REPORT.md` for full technical details.
