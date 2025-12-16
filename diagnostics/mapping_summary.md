# SignalOnly → TicketIndex → Tickets Mapping Diagnostics

**Generated:** 2025-12-16 18:58:24

---

## 1. Live Mapping Recovery

### Model Type

- **Type:** Ridge Regression (sklearn.linear_model.Ridge)
- **Regularization:** α = 5.0
- **Training Strategy:** Real data + synthetic anchor points

### Model Parameters

- **Intercept (b):** 26.065
- **Coefficient (a):** 0.739

### Live Formula

```
TicketIndex = 0.739 × SignalOnly + 26.065
```

### Synthetic Anchors

The model enforces two anchor points to prevent unrealistic predictions:

- **SignalOnly = 0 → TicketIndex ≈ 25**
  - Target: 25.0
  - Actual: 26.07
  - Error: 1.07 TI points

- **SignalOnly = 100 → TicketIndex ≈ 100**
  - Target: 100.0
  - Actual: 99.98
  - Error: 0.02 TI points

### Ticket Scaling Formula

```
Tickets = (TicketIndex / 100) × BenchmarkTickets
        = (TicketIndex / 100) × 2754
```

### Model Performance

Evaluated on 28 real titles (excluding anchor points):

- **MAE:** 14.31 TI points
- **RMSE:** 17.04 TI points
- **R²:** 0.320

---

## 2. Baseline and Spread Diagnostics

### Baseline (SignalOnly = 0)

- **TicketIndex(0):** 26.07
- **Tickets(0):** 718
  - Calculation: (26.07 / 100) × 2754

### SignalOnly Distribution

- **P5:** 0.00
- **P50:** 12.63
- **P95:** 46.00
- **Range (P5–P95):** 45.99

### TicketIndex Spread

- **TI(P5):** 26.07
- **TI(P50):** 35.40
- **TI(P95):** 60.06
- **ΔTI (P5→P95):** 33.99

### Tickets Spread

- **Tickets(P5):** 718
- **Tickets(P50):** 975
- **Tickets(P95):** 1654
- **ΔTickets (P5→P95):** 936

### Example Calculations

#### After the Rain (SignalOnly = 5.41)

```
TicketIndex = 0.739 × 5.41 + 26.065
            = 3.999 + 26.065
            = 30.06

Tickets = (30.06 / 100) × 2754
        = 0.3006 × 2754
        = 828
```

#### Afternoon of a Faun (SignalOnly = 6.63)

```
TicketIndex = 0.739 × 6.63 + 26.065
            = 4.900 + 26.065
            = 30.97

Tickets = (30.97 / 100) × 2754
        = 0.3097 × 2754
        = 853
```

#### Dracula (SignalOnly = 81.82)

```
TicketIndex = 0.739 × 81.82 + 26.065
            = 60.476 + 26.065
            = 86.54

Tickets = (86.54 / 100) × 2754
        = 0.8654 × 2754
        = 2383
```

---

## 3. Benchmark Sensitivity Analysis

This analysis tests how different benchmark values affect ticket predictions while keeping TicketIndex constant.

### Benchmark Percentiles

- **P50:** 2225 tickets
- **P60:** 2754 tickets
- **P70:** 3041 tickets

### Impact on Predictions

| Title Type | SignalOnly | TicketIndex | P50 Benchmark | P60 Benchmark | P70 Benchmark |
|------------|------------|-------------|---------------|---------------|---------------|
| Low Signal | 10.0 | 33.5 | 744 | 921 | 1017 |
| Medium Signal | 50.0 | 63.0 | 1402 | 1735 | 1917 |
| High Signal | 90.0 | 92.6 | 2060 | 2549 | 2816 |

### Key Observations

- TicketIndex remains constant across all benchmarks (by design)
- Higher benchmark values proportionally increase all ticket predictions
- Low-signal titles see smaller absolute changes than high-signal titles
- Benchmark choice is critical for calibrating the overall ticket scale

---

## 4. Nonlinearity Comparison

This section compares the current linear mapping against alternative nonlinear models.

### Model Formulas

1. **Linear (Current):** TI = 0.739 × SignalOnly + 26.065
2. **Log-Linear:** TI = 5.893 × log(SignalOnly + 1) + 26.155
3. **Sigmoid:** TI = 1436215.30 / (1 + exp(-0.0174 × (SignalOnly - 624.44)))

### Model Comparison

| Model | RMSE | AIC | BIC | Anchor(0) Error | Anchor(100) Error |
|-------|------|-----|-----|-----------------|-------------------|
| Linear | 17.04 | 162.79 | 165.45 | 1.07 | 0.02 |
| Log-Linear | 18.74 | 168.13 | 170.79 | 1.16 | 46.65 |
| Sigmoid | 16.56 | 163.19 | 167.19 | 2.65 | 57.34 |

### Interpretation

- **Lower AIC/BIC:** Better model fit with parsimony penalty
- **Anchor Error:** Deviation from target constraints (SignalOnly=0→TI≈25, SignalOnly=100→TI≈100)
- **RMSE:** Out-of-sample prediction accuracy

### Anchor Compliance Test

Can alternative models respect anchor constraints within ±3 TI points?

- **Linear:** ✓ PASS
- **Log-Linear:** ✗ FAIL
- **Sigmoid:** ✗ FAIL

---

## Summary and Recommendations

### Key Findings

1. **Live Mapping:** TicketIndex = 0.739 × SignalOnly + 26.065
2. **Anchor Compliance:** Both anchors satisfied within 1.07 TI points
3. **Baseline Floor:** Titles with zero buzz → 26 TI → 718 tickets
4. **Distribution Spread:** P5–P95 range covers 34 TI points (936 tickets)
5. **Model Simplicity:** Linear model is simplest and most interpretable

### Diagnostic Status

- ✓ Runtime paths identified
- ✓ Live mapping recovered and verified
- ✓ Baseline and spread analyzed
- ✓ Benchmark sensitivity tested
- ✓ Nonlinear alternatives evaluated

### Next Steps

If modifications are needed:

1. **Adjust Anchors:** Modify synthetic anchor values in streamlit_app._train_ml_models()
2. **Change Regularization:** Adjust Ridge alpha parameter (currently 5.0)
3. **Test Alternatives:** Consider log-linear or sigmoid if nonlinearity is critical
4. **Calibrate Benchmark:** Update BenchmarkTickets to adjust overall ticket scale

---

**Report saved to:** /workspaces/alberta_ballet_title_scoring_app.py/diagnostics/mapping_summary.md
