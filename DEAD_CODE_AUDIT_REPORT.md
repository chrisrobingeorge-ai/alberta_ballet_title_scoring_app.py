# Dead Code & Unused Files Audit Report

**Date**: December 4, 2025  
**Repository**: Alberta Ballet Title Scoring App  
**Purpose**: Identify candidates for removal before the next cleanup phase

---

## Executive Summary

This audit identified **36 potential candidates** for removal across the following categories:
- **Archive Files**: 8 files (old versions, duplicate code)
- **Legacy Scripts**: 2 files (explicitly marked deprecated)
- **Standalone Test Scripts**: 2 files (at root level, could be moved to tests/)
- **Markdown Documentation**: 14 files (some redundant, some could be consolidated)
- **Instruction Files**: 6 CSV files (unclear purpose, possibly documentation artifacts)
- **Scripts with Limited Use**: 4 files (diagnostic/validation scripts)

---

## 1. Archive Directory Files (High Confidence - Unused)

These files are located in `data/archive/` and appear to be old versions or backups.

| File | Type | Reason Considered Unused | Uncertainty |
|------|------|-------------------------|-------------|
| `data/archive/streamlit_app_old.py` | Python | Old version of main app; current app is `streamlit_app.py` | Low |
| `data/archive/title_scoring_helper.py` | Python | Archive copy; no root-level version exists (documentation has been corrected) | Low |
| `data/archive/history_city_sales.csv` | CSV | Duplicate of `data/productions/history_city_sales.csv` | Low |
| `data/archive/history_city_sales_OLD.csv` | CSV | Old version with "_OLD" suffix | Low |
| `data/archive/past_runs_OLD.csv` | CSV | Old version with "_OLD" suffix | Low |
| `data/archive/GUS_Community_Development_Insights.csv` | CSV | No references found in codebase | Medium |
| `data/archive/comeback_kids.csv` | CSV | No references found in codebase | Medium |
| `data/archive/inclusion_new_citizens.csv` | CSV | No references found in codebase | Medium |

**Notes**:
- `streamlit_app_old.py` is nearly identical to current `streamlit_app.py` but lacks recent updates
- The three CSV files (GUS_Community, comeback_kids, inclusion_new_citizens) may contain historical research data but have no code references

---

## 2. Legacy Directory (High Confidence - Deprecated)

These files are explicitly marked as deprecated in `legacy/README.md`.

| File | Type | Reason Considered Unused | Uncertainty |
|------|------|-------------------------|-------------|
| `legacy/build_city_priors.py` | Python | Explicitly deprecated; city priors now computed dynamically in `streamlit_app.py` | Low |
| `legacy/README.md` | Markdown | Documents deprecated status; would be removed with directory | Low |

**Notes**:
- The script header explicitly states "DEPRECATED - DO NOT USE FOR PRODUCTION"
- Current system learns priors dynamically from `history_city_sales.csv`
- May keep for historical reference but could also document in version control history

---

## 3. Standalone Test Scripts at Root Level (Medium Confidence)

These test scripts are at the repository root instead of in `tests/`.

| File | Type | Reason Considered for Move/Removal | Uncertainty |
|------|------|-----------------------------------|-------------|
| `test_arts_sentiment_integration.py` | Python | Integration test; should be in `tests/` | Low |
| `test_live_analytics_integration.py` | Python | Integration test; should be in `tests/` | Low |

**Notes**:
- These are valid tests that verify feature integration
- Recommendation: Move to `tests/` directory for consistency, not delete
- Both are runnable and appear to be functional

---

## 4. Markdown Documentation Files (Variable Confidence)

### 4a. Potentially Redundant/Consolidatable Documentation

| File | Content Summary | Recommendation | Uncertainty |
|------|-----------------|----------------|-------------|
| `ARTS_SENTIMENT_INTEGRATION.md` | Documents arts sentiment feature integration | Consolidate into ML_MODEL_DOCUMENTATION.md | Medium |
| `ECONOMIC_FEATURES_WIRING_SUMMARY.md` | Documents economic feature fix | Consolidate into ML_MODEL_DOCUMENTATION.md | Medium |
| `FEATURE_VALIDATION_REPORT.md` | One-time validation report | Consider archiving or removing | Medium |
| `IMPLEMENTATION_SUMMARY.md` | Implementation details for ML models | Consolidate into ML_MODEL_DOCUMENTATION.md | Low |
| `LIVE_ANALYTICS_INTEGRATION.md` | Documents live analytics integration | Consolidate into ML_MODEL_DOCUMENTATION.md | Medium |
| `NEW_FEATURES_EVALUATION_REPORT.md` | One-time evaluation report | Consider archiving or removing | Medium |
| `NOTES.md` | Statistical refactoring notes | Could be in docs/ or consolidated | Medium |

### 4b. Documentation Files That Appear Current and Useful

| File | Status | Notes |
|------|--------|-------|
| `README.md` | **Keep** | Main documentation, well-maintained |
| `ADDING_BASELINE_TITLES.md` | **Keep** | User guide for adding titles |
| `ML_MODEL_DOCUMENTATION.md` | **Keep** | Comprehensive technical documentation |
| `MODEL_README.md` | **Keep** | Model overview |
| `TICKET_ESTIMATOR_FORMULAS.md` | **Keep** | Complete formula reference |
| `VARIABLE_REFERENCE.md` | **Keep** | Variable explanations |

### 4c. Documentation in subdirectories

| File | Location | Status | Notes |
|------|----------|--------|-------|
| `data/schema.md` | data/ | **Keep** | Simple schema reference |
| `docs/WEIGHTINGS_ASSESSMENT.md` | docs/ | **Keep** | Referenced by tests |
| `docs/WEIGHTINGS_QUICK_REFERENCE.md` | docs/ | **Keep** | Quick reference card |
| `docs/weightings_diagnostics.md` | docs/ | **Keep** | Referenced by tests |
| `docs/weightings_map.md` | docs/ | **Keep** | Referenced by scripts |
| `docs/boc_integration.md` | docs/ | Review | May be superseded by IMPLEMENTATION_SUMMARY |

### 4d. Auto-generated Results Files (Low Priority)

| File | Location | Notes |
|------|----------|-------|
| `results/model_recipe_summary.md` | results/ | Auto-generated by `scripts/analyze_safe_model.py` |
| `results/external_factors_yearly_story.md` | results/ | Auto-generated by `scripts/analyze_external_factors.py` |

---

## 5. Instruction Files Directories (Medium Confidence)

These appear to be documentation artifacts or schema definitions, not actively used by code.

| Directory | Files | Status | Uncertainty |
|-----------|-------|--------|-------------|
| `data/audiences/instruction_files/` | `Audience_Donor_Features.csv` | No code references found | Medium |
| `data/economics/instruction_files/` | `External_Factors.csv` | No code references found | Medium |
| `data/productions/instruction_files/` | 4 CSV files | No code references found | Medium |

**Notes**:
- These may serve as documentation for expected data formats
- Could be useful for data entry guidance
- No `import` or `read_csv` references found in code

---

## 6. Scripts Directory Analysis (Variable Confidence)

### 6a. Potentially Redundant Scripts

| File | Description | Concern | Uncertainty |
|------|-------------|---------|-------------|
| `scripts/backtest.py` | Lightweight backtest scaffold | May be superseded by `backtest_timeaware.py` | Medium |
| `scripts/scripts_build_season_forecast.py` | Build season forecast | Naming convention suggests it was renamed | Low |
| `scripts/test_economic_integration.py` | Quick test script | Should be in `tests/` | Low |
| `scripts/validate_new_features.py` | Validation script | One-time use diagnostic | Medium |

### 6b. Scripts That Appear Active

| File | Used By | Status |
|------|---------|--------|
| `scripts/run_full_pipeline.py` | Makefile, README | **Keep** |
| `scripts/build_modelling_dataset.py` | Makefile, README | **Keep** |
| `scripts/train_safe_model.py` | Makefile, README | **Keep** |
| `scripts/backtest_timeaware.py` | Makefile | **Keep** |
| `scripts/calibrate_predictions.py` | README | **Keep** |
| `scripts/pull_show_data.py` | README | **Keep** |
| `scripts/analyze_safe_model.py` | ML_MODEL_DOCUMENTATION | **Keep** |
| `scripts/analyze_external_factors.py` | Referenced in docs | **Keep** |
| `scripts/evaluate_models.py` | Used for evaluation | **Keep** |
| `scripts/diagnose_weightings.py` | Referenced by tests | **Keep** |
| `scripts/audit_repo.py` | Audit utility | **Keep** |

---

## 7. Code Analysis: Unused Functions/Modules

Based on import analysis, the following may have limited use:

| Module/Function | Location | Concern | Uncertainty |
|-----------------|----------|---------|-------------|
| `utils/priors.py` | utils/ | May be superseded by dynamic priors in streamlit_app.py | High |
| `ml/dataset.py` | ml/ | README mentions it's deprecated for training | Medium |

**Notes**:
- `utils/priors.py` - Need to verify if any active code imports from it
- `ml/dataset.py` - README explicitly warns against using for production training, but may still be imported

---

## 8. Summary Table: Removal Candidates

### High Confidence (Safe to Remove)
| Category | Count | Files |
|----------|-------|-------|
| Archive Python files | 2 | `streamlit_app_old.py`, `title_scoring_helper.py` |
| Archive OLD CSVs | 2 | `history_city_sales_OLD.csv`, `past_runs_OLD.csv` |
| Archive duplicate CSV | 1 | `history_city_sales.csv` (in archive) |
| Legacy deprecated | 1 | `legacy/build_city_priors.py` |

### Medium Confidence (Review Before Removing)
| Category | Count | Files |
|----------|-------|-------|
| Archive research CSVs | 3 | `GUS_Community...`, `comeback_kids...`, `inclusion_new_citizens...` |
| Redundant docs | 7 | Various integration summary docs |
| Instruction files | 6 | CSV files in instruction_files/ directories |
| Diagnostic scripts | 2 | `scripts/test_economic_integration.py`, `scripts/validate_new_features.py` |
| Potentially superseded | 1 | `scripts/backtest.py` |

### Should Move (Not Delete)
| Category | Count | Destination |
|----------|-------|-------------|
| Root test scripts | 2 | Move to `tests/` |
| Naming issue | 1 | Rename `scripts/scripts_build_season_forecast.py` |

---

## 9. Recommendations

### Immediate Actions (Low Risk)
1. **Delete archive OLD files**: `*_OLD.csv` files are clearly outdated
2. **Delete archive Python files**: `streamlit_app_old.py` is superseded
3. **Move test scripts**: `test_*_integration.py` to `tests/`

### Second Phase (Medium Risk)
1. **Consolidate documentation**: Merge integration summaries into ML_MODEL_DOCUMENTATION.md
2. **Review legacy/**: Consider fully removing or archiving to separate branch
3. **Rename script**: `scripts_build_season_forecast.py` → `build_season_forecast.py`

### Hold for Discussion
1. **Instruction files**: May serve documentation purpose - confirm with team
2. **Research CSVs in archive**: May contain valuable historical data
3. **docs/boc_integration.md**: Check if superseded by IMPLEMENTATION_SUMMARY

---

## 10. Documentation Inconsistencies Fixed

During this audit, the following documentation inconsistencies were identified and corrected:

### Issue: Non-existent `title_scoring_helper.py` references
- **Problem**: README.md and ADDING_BASELINE_TITLES.md referenced `title_scoring_helper.py` at the repository root, but this file only exists in `data/archive/`
- **Resolution**: 
  - Corrected README.md to remove the file from the project structure listing
  - Corrected README.md to remove the "Score new titles manually" command
  - Fixed README.md to reference ADDING_BASELINE_TITLES.md for baselines guidance
  - Fixed ADDING_BASELINE_TITLES.md to provide manual data collection instructions instead of referencing the non-existent helper script

---

## 11. Files NOT Recommended for Removal

The following were reviewed but should be **kept**:

- All files in `tests/` directory (47 test files)
- All files in `features/` directory (actively used)
- All files in `ml/` directory (core ML pipeline)
- All files in `integrations/` directory (API clients)
- All files in `service/` directory (forecast API)
- All files in `config/` directory (configuration)
- All files in `pages/` directory (Streamlit pages)
- All data files in `data/productions/` (except duplicates in archive)
- All economic data files in `data/economics/`
- Core documentation: README, ADDING_BASELINE_TITLES, ML_MODEL_DOCUMENTATION, etc.

---

*This report was generated as part of repository maintenance. No files were deleted during this audit.*
