# PDF Report Accuracy Reference Guide

This document maps hardcoded documentation numbers to their source definitions in the code, enabling future accuracy audits.

## Critical Parameters

### 1. **Ridge Regression Alpha (α=5.0)**

**Where it appears in documentation:**
- Line 357: Methodology text "We use a constrained Ridge regression model (α=5.0)"
- Line 361: "Ridge regression from online signals" with α=5.0 notation

**Where it's defined in code:**
- **Line 2792**: `model = Ridge(alpha=5.0, random_state=42)`
- **Line 2831**: `shap_model = Ridge(alpha=5.0, random_state=42)` (SHAP explainer)
- Multiple test files: `tests/test_shap.py`, `tests/test_integration_shap.py`

**How to verify:**
```bash
grep -n "Ridge(alpha=5.0" streamlit_app.py
```

---

### 2. **TicketIndex Floor Value (SignalOnly=0 → TicketIndex≈25)**

**Where it appears in documentation:**
- Line 225: "floor (TicketIndex ≈ 25)"
- Line 361: "(SignalOnly=0 → TicketIndex≈25)"
- Line 381: "TicketIndex ≈ 25"

**Where it's defined in code:**
- **Line 2780**: `y_anchors = np.array([25.0, 100.0])`
  - This anchor constrains the model so:
    - 0 signal input → 25 TicketIndex output (floor)
    - 100 signal input → 100 TicketIndex output (benchmark)

**How to verify:**
```bash
grep -n "y_anchors = np.array(\[25.0" streamlit_app.py
```

**Note:** The floor value is "anchored" in the Ridge regression training via weighted anchor points, not a hard-coded constraint. This allows the model to learn from data while respecting these bounds.

---

### 3. **Cinderella Benchmark Ticket Count (10,978)**

**Where it appears in documentation:**
- Line 357: "typically Cinderella, ≈10,978 tickets"

**Where it's defined in code:**
- This is a **reference value** (not used directly by the model)
- Calculated from historical data: `data/productions/history_city_sales.csv`
- Cinderella runs in the data:
  - Calgary 2018-03: 7,290 tickets
  - Calgary 2022-04: 6,666 tickets
  - Edmonton 2018-03: 3,900 tickets
  - Edmonton 2022-05: 4,101 tickets

**How to verify:**
```bash
# Check historical data
grep "Cinderella" data/productions/history_city_sales.csv

# Check documentation reference
grep -n "10,978" streamlit_app.py
```

**When to update:** 
- When Alberta Ballet computes a new historical average for Cinderella
- Update both the code comment AND this document

---

### 4. **Benchmark Normalization (Benchmark = 100)**

**Where it appears in documentation:**
- Line 357: "benchmark remains at 100"
- Line 381: "benchmark (Cinderella)"
- Line 428: "100+: Benchmark/Strong (★★★★★)"

**Where it's defined in code:**
- **Line 2780 anchor**: `y_anchors = np.array([25.0, 100.0])` 
  - Second value (100) is the benchmark target
- **Line 2169** (signal weighting): Model trained to map benchmark signals to TicketIndex=100

**How to verify:**
```bash
grep -n "100\.0\|Benchmark.*100" streamlit_app.py | head -20
```

---

### 5. **Season Year Display (End Year in PDFs/Reports)**

**Where it appears in documentation:**
- Line 3669: "Season year (end of season, e.g., 2027 for Sept 2026 - May 2027)"
- PDF title generation: "Season Report (2027)"
- Month columns: "September 2027", "October 2027", etc.

**Where it's defined in code:**
- **Line 3674**: `season_year = season_end_year - 1` (internal calendar year)
- **Line 3726**: `display_year = season_end_year` (display consistency)
- **Line 3765**: `"Month": f"{m_name} {display_year}"` (uses end year)
- **Line 4020**: `season_year=int(season_end_year)` (PDF receives end year)
- **Line 4027**: `file_name=f"alberta_ballet_season_report_{season_end_year}.pdf"`

**How to verify:**
```bash
grep -n "display_year = season_end_year\|Month.*display_year" streamlit_app.py
```

---

## Accuracy Audit Checklist

When you make changes to model parameters or benchmark data, use this checklist:

- [ ] Update the code parameter (e.g., `Ridge(alpha=X)`)
- [ ] Update the methodology text in `streamlit_app.py` (around lines 350-370)
- [ ] Update the TECHNICAL_ML_REPORT.md if referenced
- [ ] Update this reference guide with new location/value
- [ ] Run tests to verify: `pytest tests/test_shap.py -v`
- [ ] Generate a test PDF to visually inspect the numbers
- [ ] Commit changes with a clear message including the old/new values

---

## What NOT to Hardcode

**These should pull from data/config, not be hardcoded:**
- ❌ Specific Cinderella ticket numbers (use historical average)
- ❌ Benchmark title name (should be configurable)
- ❌ Historical data file paths (use config.yaml)
- ❌ User-specific thresholds (use parameter inputs)

**These are OK to hardcode (model constants):**
- ✅ Ridge alpha (model parameter, doesn't change per season)
- ✅ Anchor points like 25/100 (mathematical constraint)
- ✅ Signal weights (learned from model, rarely change)

---

## Recent Updates

- **2025-12-20**: Updated Cinderella benchmark from 11,976 → 10,978
- **2025-12-20**: Updated month display to use season end year (2027) not start year (2026)

---

## Questions to Answer When Updating Numbers

1. **Where is the source data?** (file path, date last updated)
2. **Is this a fixed model parameter or data-derived value?**
3. **How often does this need to be updated?** (per season, per year, per model retraining?)
4. **Who should be notified of changes?** (analytics team, business users, etc.)
5. **Does this affect any other parts of the system?** (other reports, forecasts, etc.)

