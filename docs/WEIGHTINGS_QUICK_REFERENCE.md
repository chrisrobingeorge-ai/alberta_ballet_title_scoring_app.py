# Weightings Quick Reference

**Last Updated:** December 2, 2025

---

## Quick Status Check

Run this command to verify all weightings are active:

```bash
python scripts/diagnose_weightings.py
```

Expected output: All three systems should show **0% zero deltas** (meaning they're influencing all shows).

---

## The Weighting System

| System | File | Variable | Applied As | Impact |
|--------|------|----------|------------|--------|
| **Stone Olafson** | `streamlit_app.py` | `SEGMENT_MULT`<br/>`REGION_MULT` | Hard-coded multipliers | Segment/region adjustments |

---

## Known Issues (Dec 2025)

### 🔴 Consumer Confidence (HIGH PRIORITY)
- Only 1-2 unique values across dataset
- Effectively flat over time
- **Fix:** Investigate `nanos_consumer_confidence.csv` data quality

### 🟡 Live Analytics Granularity (MEDIUM)
- Only 4 unique engagement factors
- Limited discriminative power
- **Fix:** Add subcategory-level factors or secondary dimensions

### 🟡 Stone Olafson Multipliers (MEDIUM)
- Zero correlation improvement despite +2.6 point impact
- **Consider:** Removing or making data-driven

---

## Testing

```bash
# Run all weighting tests (13 should pass, 2 expected failures)
python -m pytest tests/test_weightings.py -v

# Expected failures document known issues above
```

---

## Correlation Impact

Based on 35 shows with historical ticket sales:

| Configuration | Correlation | Gain |
|---------------|-------------|------|
| All weights active | r = +0.366 | baseline |
| Without Live Analytics | r = +0.358 | -0.008 |
| Without Stone Olafson | r = +0.366 | 0.000 |
| Without Economics | r = +0.262 | **-0.104** |

**Conclusion:** Economics provides the most predictive value (+10.4% correlation).

---

## Documentation

- **Architecture:** `docs/weightings_map.md`
- **Diagnostics:** `docs/weightings_diagnostics.md`
- **Assessment:** `docs/WEIGHTINGS_ASSESSMENT.md`
- **Tests:** `tests/test_weightings.py`

---

## Common Commands

```bash
# Full diagnostic
python scripts/diagnose_weightings.py

# Check feature distributions
python -c "
import pandas as pd
df = pd.read_csv('data/modelling_dataset.csv')
print(df[['aud__engagement_factor', 'consumer_confidence_prairies', 
          'energy_index', 'inflation_adjustment_factor']].describe())
"

# Review per-show impacts
head -20 results/weightings_impact_summary.csv

# Run tests
python -m pytest tests/test_weightings.py -v
```

---

## Summary

✅ All three systems are **correctly wired and active**  
⚠️ Economics most valuable but consumer confidence needs fixing  
⚠️ Live Analytics working but limited by 4-value granularity  
⚠️ Stone Olafson adds no predictive value (consider removing)

**No implementation bugs found** - issues are data quality and feature engineering opportunities.
