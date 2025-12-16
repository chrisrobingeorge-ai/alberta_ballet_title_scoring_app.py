# Quick Reference: Live SignalOnly Mapping

**Diagnostic Date:** December 16, 2025  
**Status:** ✓ All tasks complete, no code changes made

---

## Live Formula (Recovered)

```python
# Ridge Regression (alpha=5.0) with synthetic anchors
TicketIndex = 0.739 × SignalOnly + 26.065

# Ticket scaling
Tickets = (TicketIndex / 100) × 11,976
```

---

## Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Intercept (b) | 26.065 | Trained from 28 titles + 28 anchors |
| Coefficient (a) | 0.739 | Ridge(alpha=5.0, random_state=42) |
| Anchor weight | 14× | max(3, n_real // 2) |
| BenchmarkTickets | 11,976 | Documented (Cinderella median) |
| Anchor(0) target | 25.0 TI | Synthetic constraint |
| Anchor(100) target | 100.0 TI | Synthetic constraint |

---

## Anchor Compliance

| Anchor | Target | Actual | Error | Status |
|--------|--------|--------|-------|--------|
| SignalOnly=0 | 25.0 TI | 26.07 TI | 1.07 | ✓ PASS (< 3) |
| SignalOnly=100 | 100.0 TI | 99.98 TI | 0.02 | ✓ PASS (< 3) |

---

## Model Performance

**Dataset:** 28 real titles (excluding synthetic anchors)

| Metric | Value |
|--------|-------|
| MAE | 14.31 TI points |
| RMSE | 17.04 TI points |
| R² | 0.320 |

**Interpretation:** Model explains 32% of variance. SignalOnly is a directional indicator, not a precise predictor.

---

## Distribution Statistics

### SignalOnly (28 titles)
- **Mean:** 15.82
- **Std Dev:** 16.89
- **P5:** 0.00
- **P50:** 12.63
- **P95:** 46.00

### TicketIndex (28 titles)
- **Mean:** 38.28
- **Std Dev:** 21.04
- **P5:** 26.07
- **P50:** 35.40
- **P95:** 60.06

### Tickets (at benchmark 11,976)
- **P5:** 3,122
- **P50:** 4,240
- **P95:** 7,193
- **Range (P5-P95):** 4,071 (about 2.3× spread)

---

## Example Calculations

| Title | SignalOnly | TicketIndex | Tickets |
|-------|------------|-------------|---------|
| After the Rain | 5.41 | 30.06 | 3,600 |
| Afternoon of a Faun | 6.63 | 30.97 | 3,708 |
| Dracula | 81.82 | 86.54 | 10,364 |
| Zero buzz floor | 0.00 | 26.07 | 3,122 |
| Perfect benchmark | 100.00 | 99.98 | 11,975 |

**Step-by-step for Dracula:**
```
TI = 0.739 × 81.82 + 26.065 = 86.54
Tickets = (86.54 / 100) × 11,976 = 10,364
```

---

## Benchmark Sensitivity

**Actual Historical Medians:**
- P50: 2,225 tickets
- P60: 2,754 tickets  
- P70: 3,041 tickets
- **Documented:** 11,976 tickets ⚠️

### Impact on Low Signal Title (SignalOnly=10, TI=33.5):
| Benchmark | Tickets |
|-----------|---------|
| P50 (2,225) | 744 |
| P60 (2,754) | 921 |
| P70 (3,041) | 1,017 |
| Documented (11,976) | 4,011 |

**Critical Finding:** Documented benchmark is 4–5× higher than actual median performance. Verify if 11,976 includes subscriptions, represents pre-COVID, or is Calgary-only.

---

## Nonlinearity Test Results

| Model | RMSE | AIC | BIC | Anchor(0) Δ | Anchor(100) Δ | Compliance |
|-------|------|-----|-----|-------------|---------------|------------|
| **Linear** | 17.04 | 162.79 | 165.45 | 1.07 | 0.02 | ✓ PASS |
| Log-Linear | 18.74 | 168.13 | 170.79 | 1.16 | 46.65 | ✗ FAIL |
| Sigmoid | 16.56 | 163.19 | 167.19 | 2.65 | 57.34 | ✗ FAIL |

**Verdict:** Linear model wins. Log-linear and sigmoid fail to respect Anchor(100) constraint.

---

## Runtime Paths

### SignalOnly Mapping (LIVE)
- **Location:** `streamlit_app.py`, function `_train_ml_models()` (line ~2645)
- **Storage:** NOT persisted (retrained on-the-fly from baselines.csv)
- **Data Sources:**
  - `data/productions/baselines.csv` (281 titles, signal columns)
  - `data/productions/history_city_sales.csv` (84 runs, ticket sales)

### Full ML Pipeline (SEPARATE)
- **Training:** `scripts/train_safe_model.py` (XGBoost with 35 features)
- **Model:** `models/model_xgb_remount_postcovid.joblib`
- **Purpose:** Predict tickets from full feature set (NOT SignalOnly)

**Note:** The persisted XGBoost model and the ephemeral Ridge model are SEPARATE systems.

---

## Diagnostic Outputs

All files in `/workspaces/alberta_ballet_title_scoring_app.py/diagnostics/`:

1. **diagnose_signalonly_mapping.py** (875 lines)  
   - Comprehensive diagnostic script
   - Replicates production training exactly

2. **mapping_summary.md** (217 lines)  
   - Technical report with all findings
   - Step-by-step calculations

3. **DIAGNOSTIC_EXECUTIVE_SUMMARY.md** (this file's companion)  
   - Business-level summary with recommendations

4. **diagnostic_output.log**  
   - Full console output for reproducibility

---

## Critical Recommendations

1. **⚠️ HIGH PRIORITY: Investigate Benchmark Discrepancy**
   - Documented: 11,976 tickets
   - Actual P60: 2,754 tickets (4.3× lower)
   - Risk: All predictions systematically inflated by ~4×

2. **Persist Ridge Model**
   - Currently retrained on-the-fly (no version control)
   - Add coefficients to model_metadata.json

3. **Implement Time-Aware CV**
   - Task requested TimeSeriesSplit validation
   - Not currently used for Ridge training

4. **Communicate Uncertainty**
   - RMSE ±17 TI → ±2,000 tickets (at benchmark 11,976)
   - R²=0.32 means SignalOnly explains 32% of variance
   - Present predictions as "directional estimates" not precise forecasts

5. **Retain Linear Model**
   - Best anchor compliance
   - Most interpretable for stakeholders
   - Nonlinear alternatives fail constraints

---

## Status: Diagnostics Complete ✓

- ✓ Runtime paths identified
- ✓ Live mapping recovered (a=0.739, b=26.065)
- ✓ Baseline floor computed (26 TI → 3,122 tickets)
- ✓ Spread analyzed (P5-P95: 34 TI, 4,071 tickets)
- ✓ Benchmark sensitivity tested (P50/P60/P70)
- ✓ Nonlinearity compared (linear wins on constraints)
- ✓ No code changes made (per instructions)

**All outputs reproducible via:**
```bash
python diagnostics/diagnose_signalonly_mapping.py
```
