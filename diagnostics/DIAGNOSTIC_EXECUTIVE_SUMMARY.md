# SignalOnly → TicketIndex → Tickets Mapping Diagnostic Summary

**Diagnostic Run Completed:** December 16, 2025  
**Repository:** alberta_ballet_title_scoring_app.py  
**Outputs:** `/workspaces/alberta_ballet_title_scoring_app.py/diagnostics/`

---

## Executive Summary

Successfully diagnosed the live SignalOnly → TicketIndex → Tickets mapping using actual persisted data and the production Ridge regression implementation. All outputs are reproducible and saved in the diagnostics directory.

### Critical Findings

1. **Live Formula Recovered:**
   ```
   TicketIndex = 0.739 × SignalOnly + 26.065
   Tickets = (TicketIndex / 100) × 11,976
   ```

2. **Anchor Compliance:** ✓ PASS  
   - SignalOnly=0 → TI=26.07 (target: 25.0, error: 1.07)
   - SignalOnly=100 → TI=99.98 (target: 100.0, error: 0.02)

3. **Model Type:** Ridge Regression (α=5.0)
   - Training: 28 real titles + 28 synthetic anchor points
   - Performance: MAE=14.31 TI, RMSE=17.04 TI, R²=0.320

4. **Nonlinearity Test:** Linear model wins on anchor compliance
   - Log-linear: Fails anchor(100) by 46.65 TI points
   - Sigmoid: Fails anchor(100) by 57.34 TI points
   - Linear: Best anchor compliance (max error: 1.07 TI)

---

## Section 1: Runtime Paths Identified

### Scoring Entrypoint
- **Primary:** `service/forecast.py` (minimal implementation, uses TODO placeholders)
- **Actual:** `streamlit_app.py` contains the live Ridge regression mapping
  - Function: `_train_ml_models()` at line ~2645
  - Implementation: Ridge(alpha=5.0) with synthetic anchors

### Training Scripts
- **Safe Model Pipeline:** `scripts/train_safe_model.py`
  - XGBoost/LightGBM training with time-aware CV
  - NOT used for SignalOnly mapping (that's in streamlit_app)
- **Legacy:** `ml/training.py` (deprecated, has leakage risks)

### Persisted Model Artefacts
- **Model:** `models/model_xgb_remount_postcovid.joblib`
  - Type: XGBoost model for full-feature prediction
  - NOT used for SignalOnly mapping
- **Metadata:** `models/model_metadata.json`
  - 35 features including wiki, trends, youtube, chartmetric
  - Does NOT contain SignalOnly Ridge model

### Key Insight
The **SignalOnly → TicketIndex mapping is NOT persisted** to disk. It is retrained on-the-fly in `streamlit_app.py` whenever the app loads title baselines. The Ridge model parameters (a=0.739, b=26.065) are ephemeral and derived from:
- Real historical data (`data/productions/baselines.csv` + `history_city_sales.csv`)
- Synthetic anchors (weighted 14x for 28 real titles)

---

## Section 2: Live Mapping Recovery

### Ridge Model Parameters
```python
# From streamlit_app._train_ml_models():
model = Ridge(alpha=5.0, random_state=42)

# Training data composition:
X_real:    28 titles × 1 feature (SignalOnly)
X_anchors: 28 synthetic points (14 × [0, 100])
y_anchors: 28 target values (14 × [25, 100])

# Learned parameters:
intercept (b) = 26.065
coefficient (a) = 0.739
```

### Synthetic Anchors
Two anchor points prevent unrealistic predictions:

| Anchor | Input | Target TI | Actual TI | Error |
|--------|-------|-----------|-----------|-------|
| Low Floor | SignalOnly=0 | 25.0 | 26.07 | 1.07 |
| Benchmark | SignalOnly=100 | 100.0 | 99.98 | 0.02 |

**Status:** ✓ Both anchors satisfied within ±3 TI tolerance

### Ticket Scaling Formula
```
Tickets = (TicketIndex / 100) × BenchmarkTickets
        = (TicketIndex / 100) × 11,976
```

**Benchmark Source:** Documented value (Cinderella median)  
**Note:** Actual Cinderella median in data is 6,281 tickets (Calgary+Edmonton combined). The 11,976 value appears to be historical or Calgary-only.

### Model Performance
Evaluated on 28 real titles (excluding synthetic anchors):
- **MAE:** 14.31 TI points
- **RMSE:** 17.04 TI points  
- **R²:** 0.320

**Interpretation:** Modest R² indicates high variance in real data relative to the simple linear model. The Ridge regularization (α=5.0) and anchor constraints prioritize interpretability over perfect fit.

---

## Section 3: Baseline and Spread Diagnostics

### Baseline (Zero Buzz)
For titles with no online presence (SignalOnly=0):
```
TicketIndex(0) = 26.07
Tickets(0) = (26.07 / 100) × 11,976 = 3,122 tickets
```

### SignalOnly Distribution
Across 28 historical titles with ticket data:
- **P5:** 0.00 (minimal buzz)
- **P50:** 12.63 (median title)
- **P95:** 46.00 (high buzz)
- **Range (P5–P95):** 45.99

### TicketIndex Spread
- **TI(P5):** 26.07
- **TI(P50):** 35.40
- **TI(P95):** 60.06
- **ΔTI (P5→P95):** 33.99 (about 34 points of separation)

### Tickets Spread
Using BenchmarkTickets = 11,976:
- **Tickets(P5):** 3,122
- **Tickets(P50):** 4,240
- **Tickets(P95):** 7,193
- **ΔTickets (P5→P95):** 4,071 (about 2.3× from P5 to P95)

### Example Title Calculations

#### 1. After the Rain (SignalOnly = 5.41)
```
TicketIndex = 0.739 × 5.41 + 26.065 = 30.06
Tickets = (30.06 / 100) × 11,976 = 3,600
```

#### 2. Afternoon of a Faun (SignalOnly = 6.63)
```
TicketIndex = 0.739 × 6.63 + 26.065 = 30.97
Tickets = (30.97 / 100) × 11,976 = 3,708
```

#### 3. Dracula (SignalOnly = 81.82)
```
TicketIndex = 0.739 × 81.82 + 26.065 = 86.54
Tickets = (86.54 / 100) × 11,976 = 10,364
```

**Observation:** Low-signal titles (After the Rain, Afternoon of a Faun) cluster near the baseline floor (~3,600 tickets), while high-signal titles (Dracula) reach ~10,400 tickets — about 2.9× the low-signal baseline.

---

## Section 4: Benchmark Sensitivity Analysis

### Percentile Benchmarks
From actual historical ticket medians:
- **P50:** 2,225 tickets
- **P60:** 2,754 tickets
- **P70:** 3,041 tickets

**Note:** These are MUCH lower than the documented benchmark of 11,976. This discrepancy suggests:
1. The 11,976 value is from Cinderella's total (Calgary + Edmonton aggregated)
2. Or it represents pre-COVID attendance
3. Or it includes subscription tickets (not just single tickets)

### Impact on Predictions

| Title Type | SignalOnly | TicketIndex | P50 (2,225) | P60 (2,754) | P70 (3,041) | Documented (11,976) |
|------------|------------|-------------|-------------|-------------|-------------|---------------------|
| Low Signal | 10.0 | 33.5 | 744 | 921 | 1,017 | 4,011 |
| Medium Signal | 50.0 | 63.0 | 1,402 | 1,735 | 1,917 | 7,545 |
| High Signal | 90.0 | 92.6 | 2,060 | 2,549 | 2,816 | 11,089 |

**Key Observations:**
- TicketIndex remains constant regardless of benchmark (by design)
- Benchmark choice scales ALL predictions proportionally
- Low-signal titles see smaller absolute changes than high-signal titles
- **Critical:** The documented 11,976 benchmark produces predictions 4–5× higher than P50/P60/P70

### Recommendation
**Investigate the benchmark discrepancy.** If actual median performance is ~2,200–3,000 tickets, using 11,976 will systematically overestimate all titles by 4–5×.

---

## Section 5: Nonlinearity Comparison

### Models Tested

1. **Linear (Current):**
   ```
   TicketIndex = 0.739 × SignalOnly + 26.065
   ```
   - RMSE: 17.04, AIC: 162.79, BIC: 165.45
   - Anchor(0) error: 1.07, Anchor(100) error: 0.02

2. **Log-Linear:**
   ```
   TicketIndex = 5.893 × log(SignalOnly + 1) + 26.155
   ```
   - RMSE: 18.74, AIC: 168.13, BIC: 170.79
   - Anchor(0) error: 1.16, **Anchor(100) error: 46.65** ❌

3. **Sigmoid:**
   ```
   TicketIndex = 1,436,215 / (1 + exp(-0.0174 × (SignalOnly - 624.44)))
   ```
   - RMSE: 16.56, AIC: 163.19, BIC: 167.19
   - Anchor(0) error: 2.65, **Anchor(100) error: 57.34** ❌

### Anchor Compliance Test (±3 TI tolerance)
- **Linear:** ✓ PASS (max error: 1.07)
- **Log-Linear:** ✗ FAIL (Anchor(100) off by 46.65 TI)
- **Sigmoid:** ✗ FAIL (Anchor(100) off by 57.34 TI)

### Verdict
**Linear model wins on interpretability and anchor compliance.**  
While sigmoid has slightly lower RMSE (16.56 vs 17.04), it:
- Fails to respect the critical Anchor(100) constraint
- Produces unrealistic parameters (L=1.4M, x0=624)
- Is less interpretable for stakeholders

The log-linear model diminishes returns at high SignalOnly (saturating effect), which contradicts the business assumption that high-buzz titles (Nutcracker, Dracula) should track proportionally with their signal.

**Recommendation:** Retain the linear mapping unless there is strong empirical evidence of nonlinearity in actual ticket outcomes.

---

## Detailed Outputs

### Files Created

1. **`diagnostics/diagnose_signalonly_mapping.py`**
   - Comprehensive diagnostic script (875 lines)
   - Replicates exact production training procedure
   - Loads baselines.csv, history_city_sales.csv
   - Computes all metrics, tests alternatives

2. **`diagnostics/mapping_summary.md`**
   - Markdown report with all findings (217 lines)
   - Includes step-by-step calculations for example titles
   - Model comparison tables
   - Anchor compliance tests

3. **`diagnostics/diagnostic_output.log`**
   - Full console output from script execution
   - Reproducible audit trail

### Reproducibility
To re-run diagnostics:
```bash
cd /workspaces/alberta_ballet_title_scoring_app.py
python diagnostics/diagnose_signalonly_mapping.py
```

All outputs deterministic (random_state=42).

---

## Critical Observations & Recommendations

### 1. Benchmark Discrepancy (HIGH PRIORITY)
**Issue:** Documented benchmark (11,976) is 4–5× higher than actual median performance (P50=2,225, P60=2,754).

**Impact:** If using 11,976, all predictions will be systematically inflated by 4–5×.

**Recommendation:**
- Verify whether 11,976 represents:
  - Total tickets (subscriptions + single)?
  - Pre-COVID attendance?
  - Calgary-only vs combined markets?
- Update to a realistic benchmark (suggest P60=2,754 or P70=3,041)
- OR document that predictions are "potential capacity" vs actual median

### 2. Model Not Persisted
**Issue:** SignalOnly Ridge model is retrained on-the-fly, not saved to disk.

**Impact:** 
- Parameters (a=0.739, b=26.065) can drift if baselines.csv or history changes
- No version control for model coefficients
- Harder to audit "what model was used for forecast X?"

**Recommendation:**
- Persist Ridge model parameters to JSON metadata
- Include in model_metadata.json:
  ```json
  {
    "signalonly_ridge": {
      "intercept": 26.065,
      "coefficient": 0.739,
      "anchor_0_pred": 26.07,
      "anchor_100_pred": 99.98,
      "training_date": "2025-12-16",
      "n_titles": 28,
      "benchmark_tickets": 11976
    }
  }
  ```

### 3. Low R² (0.320)
**Issue:** Model explains only 32% of variance in TicketIndex.

**Interpretation:**
- SignalOnly is a weak predictor of actual tickets (expected for online buzz)
- Real performance driven by marketing, timing, demographics, etc.
- Ridge regularization + anchor constraints prioritize stability over fit

**Recommendation:**
- Acceptable for a "directional estimate" tool
- NOT suitable for precise capacity planning
- Communicate prediction intervals (±17 TI RMSE → ±2,000 tickets at benchmark 11,976)

### 4. Time-Aware CV Not Used for SignalOnly Mapping
**Issue:** Task requested time-aware CV splitter, but streamlit_app doesn't use it for Ridge training.

**Current:** Single train on all 28 titles + anchors  
**Expected:** TimeSeriesSplit cross-validation

**Impact:** Cannot assess out-of-sample performance over time.

**Recommendation:**
- Implement time-aware CV in a separate validation script
- Report CV-RMSE to assess temporal stability
- Check if coefficients drift across time folds

### 5. Nonlinearity Not Critical
**Finding:** Log-linear and sigmoid fail anchor compliance tests.

**Recommendation:** Retain linear mapping. The simplicity and anchor compliance outweigh marginal RMSE gains.

---

## Next Steps (DO NOT IMPLEMENT — DIAGNOSTICS ONLY)

Per your instructions, NO CODE CHANGES were made. If you wish to proceed, consider:

1. **Investigate Benchmark:**
   - Query: What is the true Cinderella median? (6,281 in data vs 11,976 documented)
   - Decision: Use realistic benchmark or clearly label predictions as "capacity potential"

2. **Persist Model:**
   - Save Ridge coefficients to model_metadata.json
   - Version control for auditability

3. **Time-Aware Validation:**
   - Implement TimeSeriesSplit CV for SignalOnly Ridge
   - Report temporal stability of coefficients

4. **Floor Adjustment (Optional):**
   - If Anchor(0)=26.07 is too high, adjust target to 20 or 22
   - Reweight anchors (currently 14× for 28 titles)

5. **Switch Benchmark (Optional):**
   - Test P60=2,754 or P70=3,041 as benchmark
   - Compare predictions to actual outcomes

---

## Conclusion

✓ **Diagnostics Complete**  
✓ **Live mapping recovered:** TI = 0.739 × SignalOnly + 26.065  
✓ **Anchors verified:** Within ±3 TI tolerance  
✓ **Baseline floor:** 26 TI → 3,122 tickets (at benchmark 11,976)  
✓ **Spread:** P5–P95 range of 34 TI (4,071 tickets)  
✓ **Nonlinearity:** Linear model wins on anchor compliance  
✓ **Benchmark sensitivity:** Critical choice — 11,976 may be 4–5× too high  

All outputs reproducible in `/workspaces/alberta_ballet_title_scoring_app.py/diagnostics/`.

**No code changes made** per your instructions.
