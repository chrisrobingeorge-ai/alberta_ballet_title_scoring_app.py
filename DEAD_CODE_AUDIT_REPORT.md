# Dead Code & Unused Files Audit Report

**Date**: December 4, 2025  
**Repository**: Alberta Ballet Title Scoring App  
**Status**: ✅ CLEANUP COMPLETED

---

## Executive Summary

This audit identified and **removed 19 files/directories** that were dead code, unused, or redundant:

### Files Removed
| Category | Count | Files |
|----------|-------|-------|
| Archive Python files | 1 | `streamlit_app_old.py` |
| Archive CSV files | 5 | `history_city_sales.csv`, `history_city_sales_OLD.csv`, `past_runs_OLD.csv`, `GUS_Community_Development_Insights.csv`, `comeback_kids.csv`, `inclusion_new_citizens.csv` |
| Legacy deprecated | 2 | `legacy/build_city_priors.py`, `legacy/README.md` |
| Empty directories | 2 | `data/archive/`, `legacy/` |
| Redundant documentation | 6 | Integration summaries, validation reports |
| Diagnostic scripts | 3 | One-time validation/test scripts |

### Files Relocated
| File | From | To |
|------|------|-----|
| `title_scoring_helper.py` | `data/archive/` | root (restored to proper location) |
| `test_arts_sentiment_integration.py` | root | `tests/` |
| `test_live_analytics_integration.py` | root | `tests/` |

### Files Renamed
| Old Name | New Name |
|----------|----------|
| `scripts/scripts_build_season_forecast.py` | `scripts/build_season_forecast.py` |

---

## Detailed Changes

### 1. Archive Directory (Removed Entirely)

The following files were removed from `data/archive/`:

| File | Reason |
|------|--------|
| `streamlit_app_old.py` | Old version of main app |
| `history_city_sales.csv` | Duplicate of `data/productions/history_city_sales.csv` |
| `history_city_sales_OLD.csv` | Old version |
| `past_runs_OLD.csv` | Old version |
| `GUS_Community_Development_Insights.csv` | No code references |
| `comeback_kids.csv` | No code references |
| `inclusion_new_citizens.csv` | No code references |

**Note:** `title_scoring_helper.py` was moved from `data/archive/` to the repository root (its proper location).

The empty `data/archive/` directory was also removed.

### 2. Legacy Directory (Removed Entirely)

| File | Reason |
|------|--------|
| `legacy/build_city_priors.py` | Explicitly deprecated; city priors now computed dynamically |
| `legacy/README.md` | Documented deprecated status |

The empty `legacy/` directory was also removed.

### 3. Redundant Documentation (Removed)

| File | Reason |
|------|--------|
| `ARTS_SENTIMENT_INTEGRATION.md` | Content covered in ML_MODEL_DOCUMENTATION.md |
| `LIVE_ANALYTICS_INTEGRATION.md` | Content covered in ML_MODEL_DOCUMENTATION.md |
| `ECONOMIC_FEATURES_WIRING_SUMMARY.md` | Content covered in ML_MODEL_DOCUMENTATION.md |
| `IMPLEMENTATION_SUMMARY.md` | Content covered in ML_MODEL_DOCUMENTATION.md |
| `FEATURE_VALIDATION_REPORT.md` | One-time validation report |
| `NEW_FEATURES_EVALUATION_REPORT.md` | One-time evaluation report |

### 4. Diagnostic Scripts (Removed)

| File | Reason |
|------|--------|
| `scripts/test_economic_integration.py` | One-time diagnostic test |
| `scripts/validate_new_features.py` | One-time validation script |
| `scripts/backtest.py` | Superseded by `scripts/backtest_timeaware.py` |

### 5. Test Scripts (Relocated)

| File | Action |
|------|--------|
| `test_arts_sentiment_integration.py` | Moved to `tests/` |
| `test_live_analytics_integration.py` | Moved to `tests/` |

### 6. Script Renamed

| Old | New | Reason |
|-----|-----|--------|
| `scripts/scripts_build_season_forecast.py` | `scripts/build_season_forecast.py` | Fixed naming convention |

---

## Documentation Updates

- **README.md**: Removed references to non-existent `title_scoring_helper.py` and `legacy/` directory
- **ADDING_BASELINE_TITLES.md**: Updated to provide manual data collection instructions

---

## Files Retained (Not Removed)

The following items were identified as potentially removable but **kept** for their utility:

### Instruction Files
Located in `data/*/instruction_files/` directories. These may serve as documentation for data formats.

### NOTES.md
Contains statistical refactoring notes that may still be useful for development context.

---

## Verification

After cleanup, the repository structure is cleaner and all removed code had:
- No active imports or references
- Been superseded by newer implementations
- Been explicitly marked as deprecated
- Been one-time diagnostic artifacts

All core functionality remains intact.
