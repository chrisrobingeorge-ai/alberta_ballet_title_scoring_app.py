# Weightings Diagnostics Report

**Date:** December 2, 2025  
**Analysis:** Quantitative validation of three weighting systems  
**Dataset:** 286 shows from `data/modelling_dataset.csv`

---

## Executive Summary

All three weighting systems (**Live Analytics**, **Economics**, **Stone Olafson**) are actively influencing scores across the dataset. None are disconnected or inert. However, there are important concerns about data variance and impact magnitudes:

### Key Findings

| Weighting System | Status | Mean Impact | Data Variance | Concerns |
|------------------|--------|-------------|---------------|----------|
| **Live Analytics** | ✅ Active | +3.2 points | ⚠️ Limited (4 values) | Low variance limits predictive power |
| **Economics** | ✅ Active | +1.2 points | ⚠️ Mixed variance | Consumer confidence nearly flat |
| **Stone Olafson** | ✅ Active | +2.6 points | ✅ Good | Deterministic (not learned by model) |

---

## 1. Live Analytics Weightings

### Impact Assessment

**Are they changing scores?** ✅ **YES**

- Mean delta: **+3.249 points**
- Median delta: **+2.997 points**
- Range: +1.092 to +6.935 points
- **Zero deltas: 0%** (all shows affected)

**Magnitude:** Moderate. Live Analytics adds ~3 points on average to the base score (~60), representing a ~5% uplift.

### Data Variance Analysis

**Feature:** `aud__engagement_factor`

- **Unique values: 4** ⚠️
- Mean: 1.063
- Std deviation: 0.013 (very low)
- Range: 1.051 to 1.080

**Values distribution:**
- 1.051 (low engagement categories)
- 1.073
- 1.077
- 1.080 (high engagement categories)

### Red Flags

1. **⚠️ Extremely limited variance** - Only 4 unique values across 286 shows
2. **⚠️ Narrow range** - All values clustered between 1.05 and 1.08 (5-8% adjustment)
3. **⚠️ Low discriminative power** - Model has minimal signal to learn category differences

### Interpretation

The Live Analytics engagement factor is **technically working but severely constrained**:

- It successfully differentiates between high/low engagement categories (e.g., pop IP vs classical)
- However, the 4-value granularity means many different categories share the same multiplier
- The narrow 1.05-1.08 range suggests conservative application (avoiding extreme adjustments)

**Is this a bug?** No - the function `get_category_engagement_factor()` is correctly loading data and applying category lookups. The issue is **source data granularity**: the `live_analytics.csv` file appears to have limited category-level variation in engagement indices.

### Correlation with Target

- With Live Analytics: r = +0.366
- Without Live Analytics: r = +0.358
- **Delta: +0.008** (minimal difference)

**Interpretation:** Removing Live Analytics has almost no effect on correlation with actual ticket sales, suggesting its current implementation adds limited predictive value.

---

## 2. Economic Weightings

### Impact Assessment

**Are they changing scores?** ✅ **YES**

- Mean delta: **+1.227 points**
- Median delta: **+0.756 points**
- Range: -8.809 to +25.980 points
- **Zero deltas: 0%** (all shows affected)

**Magnitude:** Small to moderate, but highly variable. Economics can add up to +26 points in extreme cases (2021 high energy prices).

### Data Variance Analysis

#### Consumer Confidence (`consumer_confidence_prairies`)

- **Unique values: 2** ⚠️⚠️ CRITICAL
- Mean: 50.050
- Std deviation: 0.135 (virtually flat)
- Range: 50.0 to 50.41

**RED FLAG:** Essentially flat. Only 2 values across 286 shows means no meaningful temporal variation.

#### Energy Index (`energy_index`)

- **Unique values: 9** ✅
- Mean: 1049.38
- Std deviation: 113.37
- Range: 704 to 1867 (165% span)

**GOOD:** Substantial variation representing oil price cycles (2015-2025).

#### Inflation Factor (`inflation_adjustment_factor`)

- **Unique values: 31** ✅✅ EXCELLENT
- Mean: 1.007
- Std deviation: 0.039
- Range: 0.944 to 1.208 (28% span)

**EXCELLENT:** Rich temporal variation capturing inflation trends.

### Red Flags

1. **⚠️⚠️ Consumer confidence is nearly constant** - Only 2 values (50.0 and 50.41)
   - This suggests data is not varying over time as expected
   - May be using a regional average that's mostly flat
   - Limited predictive signal

2. **⚠️ High variance in deltas** - Std dev of 3.477 with mean of 1.227
   - Some shows get +26 points, others get -9 points
   - Could indicate oversensitivity to oil price swings

3. **⚠️ Economic impact can be negative** - Min delta = -8.8
   - During low oil price periods, economics reduces scores
   - Questionable if this reflects actual demand behavior

### Correlation with Target

- With Economics: r = +0.366
- Without Economics: r = +0.262
- **Delta: +0.104** (moderate improvement)

**Interpretation:** Economics provides meaningful predictive signal. Removing it drops correlation by ~10%, suggesting the energy index and inflation features do capture relevant demand drivers.

**However:** This is likely driven by energy_index (oil prices) rather than consumer confidence (which is flat).

---

## 3. Stone Olafson Weightings

### Impact Assessment

**Are they changing scores?** ✅ **YES**

- Mean delta: **+2.572 points**
- Median delta: **+2.572 points**
- Range: +1.072 to +4.494 points
- **Zero deltas: 0%** (all shows affected)

**Magnitude:** Moderate and consistent. Stone Olafson adds ~2.6 points on average (~4% uplift).

### Data Characteristics

**Note:** Stone Olafson weightings are **hard-coded multipliers**, not learned features. They are applied deterministically in `calc_scores()`:

```python
seg = SEGMENT_MULT[seg_key]
fam *= seg.get(gender, 1.0) * seg.get(category, 1.0)
mot *= seg.get(gender, 1.0) * seg.get(category, 1.0)

fam *= REGION_MULT[reg_key]
mot *= REGION_MULT[reg_key]
```

### Observations

1. **✅ Consistent impact** - Very low std dev (0.655) indicates stable multiplier application
2. **✅ Positive influence** - All deltas are positive (multipliers generally > 1.0)
3. **⚠️ Not learned by model** - These are business rules, not data-driven weights

### Correlation with Target

- With Stone Olafson: r = +0.366
- Without Stone Olafson: r = +0.366
- **Delta: 0.000** (no change)

**CRITICAL FINDING:** Removing Stone Olafson multipliers has **zero effect** on correlation with actual ticket sales.

**Interpretation:** The segment/region multipliers are having **no measurable impact on predictive accuracy**. They shift all scores by a consistent amount but don't improve the model's ability to rank shows by demand.

---

## Overall System Behavior

### Correlation Hierarchy

From the target correlation analysis (35 shows with historical sales):

1. **With all weights:** r = +0.366
2. **Without Live Analytics:** r = +0.358 (-0.008)
3. **Without Stone Olafson:** r = +0.366 (0.000)
4. **Without Economics:** r = +0.262 (-0.104)

**Ranking by importance:**
1. **Economics** contributes +0.104 to correlation (most important)
2. **Live Analytics** contributes +0.008 to correlation (minimal)
3. **Stone Olafson** contributes 0.000 to correlation (no effect)

### Effective Weights

Based on the diagnostics, the **effective weighting hierarchy** is:

1. **Base signal** (wiki/trends/chartmetric/youtube) - Foundation
2. **Economics** - Meaningful predictor (+10% correlation improvement)
3. **Live Analytics** - Minor predictor (+0.8% correlation improvement)
4. **Stone Olafson** - Cosmetic adjustment (0% correlation improvement)

---

## Recommendations

### Immediate Actions

#### 1. Consumer Confidence Data Investigation 🔴 HIGH PRIORITY

**Issue:** Only 2 unique values (50.0, 50.41) across 286 shows.

**Actions:**
- Verify `data/economics/nanos_consumer_confidence.csv` has temporal variation
- Check if `join_consumer_confidence()` is using the correct date column
- Consider using weekly or monthly breakdowns instead of regional averages
- If data is genuinely flat, consider removing this feature (adds minimal value with high complexity)

**Quick diagnostic:**
```bash
# Check source data variance
python -c "import pandas as pd; df = pd.read_csv('data/economics/nanos_consumer_confidence.csv'); print(df['value'].describe())"
```

#### 2. Live Analytics Granularity Enhancement 🟡 MEDIUM PRIORITY

**Issue:** Only 4 unique values limits predictive power.

**Actions:**
- Review `live_analytics.csv` to understand why only 4 categories emerge
- Consider subcategory-level engagement factors (e.g., pop_ip vs contemporary_mixed_bill)
- Add secondary engagement dimensions (age demographics, spending patterns)
- If source data truly has limited variation, document this as a known limitation

#### 3. Stone Olafson Multiplier Review 🟡 MEDIUM PRIORITY

**Issue:** Multipliers add computational complexity but provide no correlation improvement.

**Actions:**
- **Option A:** Remove Stone Olafson multipliers entirely (simplify codebase)
- **Option B:** Make multipliers segment/region-specific based on historical performance
- **Option C:** Feed segment/region as categorical features to the model and let it learn weights

**Rationale:** Hard-coded multipliers that don't improve predictions are technical debt.

#### 4. Economic Feature Weighting Caps 🟢 LOW PRIORITY

**Issue:** Economics can swing scores by ±26 points (large variance).

**Actions:**
- Consider capping economic multipliers to a narrower range (e.g., 0.85-1.15)
- Test whether log-transforming energy_index improves stability
- Evaluate whether the full oil price volatility should translate to demand volatility

**Current behavior:**
- Energy index of 1867 (2021 peak) → +1.87x multiplier
- Energy index of 704 (2015/2019 trough) → +0.70x multiplier

**Proposed:**
```python
energy_mult = np.clip(energy / 1000.0, 0.85, 1.15)
```

### Documentation Improvements

1. **Add data dictionary** for `live_analytics.csv` explaining engagement index methodology
2. **Document Stone Olafson multiplier** origins (are they research-based or business judgment?)
3. **Create time-series plots** of economic features to visualize temporal coverage
4. **Add variance report** to `scripts/validate_new_features.py`

### Testing Requirements

1. **Unit test:** `test_engagement_factor_loading()` - verify 4+ categories are loaded
2. **Unit test:** `test_consumer_confidence_variance()` - assert >2 unique values
3. **Integration test:** `test_weightings_non_zero_impact()` - assert mean delta > threshold
4. **Regression test:** Compare model performance with/without each weighting

---

## How to Run This Diagnostic

```bash
# Full diagnostic with default paths
python scripts/diagnose_weightings.py

# Custom dataset and output paths
python scripts/diagnose_weightings.py \
    --dataset data/modelling_dataset.csv \
    --output results/weightings_impact_summary.csv

# Review per-show impacts
head -20 results/weightings_impact_summary.csv

# Check variance of features
python -c "
import pandas as pd
df = pd.read_csv('results/weightings_impact_summary.csv')
print(df[['aud__engagement_factor', 'consumer_confidence_prairies', 
          'energy_index', 'inflation_adjustment_factor']].describe())
"
```

---

## Conclusion

### Summary Table

| Weighting | Active? | Impact | Variance | Correlation Gain | Priority Fix |
|-----------|---------|--------|----------|------------------|--------------|
| **Live Analytics** | ✅ Yes | +3.2 pts | ⚠️ Low (4 values) | +0.008 | Medium |
| **Economics** | ✅ Yes | +1.2 pts | ⚠️ Mixed | +0.104 | High |
| **Stone Olafson** | ✅ Yes | +2.6 pts | ✅ Good | 0.000 | Medium |

### Final Assessment

**All three weightings are wired correctly and actively influencing scores.** However:

1. **Economics is the only system providing meaningful predictive lift** (+10% correlation)
2. **Consumer confidence is effectively flat** and should be investigated
3. **Live Analytics has limited granularity** (4 values) reducing its utility
4. **Stone Olafson multipliers are cosmetic** - they shift scores but don't improve ranking

**Recommended next actions:**
1. Fix consumer confidence data variance (high priority)
2. Consider removing or revising Stone Olafson multipliers
3. Enhance Live Analytics granularity if source data permits
4. Add caps to economic multipliers to prevent extreme swings

This is **not a wiring bug** - the code is correctly applying all three systems. The issues are:
- **Data quality** (consumer confidence flat)
- **Feature engineering** (need more granular engagement factors)
- **Architecture** (hard-coded multipliers vs learned weights)
