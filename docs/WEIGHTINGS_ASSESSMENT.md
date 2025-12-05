# Weightings Assessment Summary

This document summarizes the three key weighting systems (Live Analytics, Economics, Stone Olafson) in the Alberta Ballet Title Scoring App.

---

## Overview

All three weighting systems are correctly wired and actively influence the app's ticket demand predictions.

### Related Documentation

1. **Documentation**
   - `docs/weightings_map.md` - Comprehensive mapping of each weighting system
   - `docs/weightings_diagnostics.md` - Quantitative findings and recommendations

2. **Diagnostic Tools**
   - `scripts/diagnose_weightings.py` - Automated diagnostic script
   - `results/weightings_impact_summary.csv` - Per-show impact analysis (286 shows)

3. **Tests**
   - `tests/test_weightings.py` - 15 unit/integration tests (13 passing, 2 known issues)

---

## Key Findings

### ✅ All Three Systems Are Active

| System | Wired Correctly? | Mean Impact | Correlation Gain | Status |
|--------|------------------|-------------|------------------|---------|
| **Live Analytics** | ✅ Yes | +3.2 points | +0.008 | Limited variance |
| **Economics** | ✅ Yes | +1.2 points | +0.104 | Strong predictor |
| **Stone Olafson** | ✅ Yes | +2.6 points | 0.000 | No predictive gain |

### Critical Issues Found

#### 🔴 HIGH PRIORITY: Consumer Confidence Data

- **Issue:** Only 1-2 unique values across entire dataset
- **Impact:** Effectively flat over time, providing minimal signal
- **Location:** `consumer_confidence_prairies` feature from `nanos_consumer_confidence.csv`
- **Action Required:** Investigate data source and temporal joins

#### 🟡 MEDIUM PRIORITY: Live Analytics Granularity

- **Issue:** Only 4 unique engagement factor values
- **Impact:** Limited discriminative power between categories
- **Location:** `aud__engagement_factor` from `live_analytics.csv`
- **Action Required:** Enhance category-level granularity or add subcategory factors

#### 🟡 MEDIUM PRIORITY: Stone Olafson Multipliers

- **Issue:** Hard-coded multipliers add zero predictive value
- **Impact:** Computational complexity without benefit
- **Location:** `SEGMENT_MULT` and `REGION_MULT` dictionaries in `streamlit_app.py`
- **Action Required:** Consider removing or making data-driven

---

## How Weightings Work

### 1. Live Analytics
**Data Source:** `data/audiences/live_analytics.csv`  
**Application:** Pre-model feature (`aud__engagement_factor`)  
**Method:** XGBoost learns weight during training

```python
# Applied in scripts/build_modelling_dataset.py
df['aud__engagement_factor'] = df['category'].apply(
    lambda cat: get_category_engagement_factor(cat)
)
```

**Current Values:** 1.051, 1.073, 1.077, 1.080 (only 4 distinct values)

### 2. Economics
**Data Sources:**
- `data/economics/nanos_consumer_confidence.csv` (Prairies confidence)
- `data/economics/commodity_price_index.csv` (Energy index)
- `data/economics/boc_cpi_monthly.csv` (Inflation)

**Application:** Pre-model features  
**Method:** XGBoost learns weights during training

```python
# Applied in data/features.py via temporal joins
features = join_consumer_confidence(features, nanos_df, 'opening_date')
features = join_energy_index(features, commodity_df, 'opening_date')
features = compute_inflation_adjustment_factor(features, cpi_df, 'opening_date')
```

**Current Values:**
- Consumer confidence: 50.0-50.41 (⚠️ flat)
- Energy index: 704-1867 (✅ good variance)
- Inflation: 0.94-1.21 (✅ excellent variance)

### 3. Stone Olafson
**Data Source:** Hard-coded dictionaries in `streamlit_app.py`  
**Application:** Post-signal multipliers  
**Method:** Deterministic scaling (not learned)

```python
# Applied in streamlit_app.py::calc_scores()
seg = SEGMENT_MULT[seg_key]
fam *= seg.get(gender, 1.0) * seg.get(category, 1.0)
mot *= seg.get(gender, 1.0) * seg.get(category, 1.0)

fam *= REGION_MULT[reg_key]
mot *= REGION_MULT[reg_key]
```

**Impact:** Adds ~2.6 points but no correlation improvement

---

## Test Results

**Test Suite:** `tests/test_weightings.py`

```
13 passed, 2 xfailed (expected failures)
```

**Expected Failures:**
1. `test_engagement_factor_variance` - Documents Live Analytics limited variance issue
2. `test_consumer_confidence_variance` - Documents Consumer Confidence flat data issue

**All Core Functionality Tests Passing:**
- ✅ Live Analytics loading and application
- ✅ Economics features (confidence, energy, inflation) loading and joining
- ✅ Stone Olafson multipliers existence and application
- ✅ Integration tests showing all weightings have non-zero impact

---

## Recommendations (Priority Order)

### 1. Fix Consumer Confidence Data (HIGH)

**Problem:** Only 1-2 values across 286 shows

**Actions:**
```bash
# Investigate source data
python -c "
import pandas as pd
df = pd.read_csv('data/economics/nanos_consumer_confidence.csv')
prairies = df[(df['category']=='Demographics') & (df['metric']=='Prairies')]
print(prairies['value'].value_counts())
"

# Check temporal join logic
python scripts/diagnose_weightings.py
```

**Options:**
- Switch to national/headline confidence index (more variance)
- Use weekly data instead of monthly aggregates
- Remove feature if truly flat (minimal value)

### 2. Enhance Live Analytics Granularity (MEDIUM)

**Problem:** Only 4 engagement factors

**Actions:**
- Add subcategory-level engagement (e.g., pop_ip_touring vs pop_ip_resident)
- Include secondary dimensions (age, spending patterns)
- Document if source data truly has limited variation

### 3. Evaluate Stone Olafson Multipliers (MEDIUM)

**Problem:** No correlation improvement despite +2.6 point impact

**Actions:**
- **Option A:** Remove hard-coded multipliers (simplify code)
- **Option B:** Make segment/region features for model to learn
- **Option C:** Base multipliers on historical performance data

### 4. Add Economic Caps (LOW)

**Problem:** Economics can swing scores ±26 points

**Actions:**
```python
# In diagnostic script, test capped version
energy_mult = np.clip(energy / 1000.0, 0.85, 1.15)
```

---

## How to Use These Tools

### Run Full Diagnostic

```bash
# Generate impact analysis
python scripts/diagnose_weightings.py

# Review results
head -20 results/weightings_impact_summary.csv
```

### Run Tests

```bash
# Run all weighting tests
python -m pytest tests/test_weightings.py -v

# Run specific test class
python -m pytest tests/test_weightings.py::TestEconomicsWeightings -v
```

### Check Feature Distributions

```bash
# Check actual feature values in modelling dataset
python -c "
import pandas as pd
df = pd.read_csv('data/modelling_dataset.csv')
for feat in ['aud__engagement_factor', 'consumer_confidence_prairies', 
             'energy_index', 'inflation_adjustment_factor']:
    print(f'\n{feat}:')
    print(f'  Unique: {df[feat].nunique()}')
    print(f'  Range: {df[feat].min():.2f} - {df[feat].max():.2f}')
    print(f'  Mean: {df[feat].mean():.3f}')
"
```

---

## Related Files

### Documentation
- `docs/weightings_map.md` - System architecture documentation
- `docs/weightings_diagnostics.md` - Quantitative findings report

### Diagnostic Tools
- `scripts/diagnose_weightings.py` - Diagnostic automation
- `tests/test_weightings.py` - Unit/integration tests
- `results/weightings_impact_summary.csv` - Per-show analysis

---

## Conclusion

✅ **All three weighting systems are correctly implemented and actively influencing scores.**

**Effectiveness Ranking:**
1. **Economics** (+10.4% correlation) - Most valuable, driven by energy_index and inflation
2. **Live Analytics** (+0.8% correlation) - Working but limited by 4-value granularity
3. **Stone Olafson** (0% correlation) - Technically working but adds no predictive value

**Known Data Quality Issues:**
1. Consumer confidence data source has limited variance (high priority to address)
2. Live Analytics granularity limited to 4 values
3. Stone Olafson multipliers are hard-coded rather than data-driven
