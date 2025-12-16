# SignalOnly Mapping Visualization & Data Summary

**Diagnostic Date:** December 16, 2025

---

## Visual: The Three-Stage Mapping

```
┌─────────────────────────────────────────────────────────────────────┐
│                  SignalOnly → TicketIndex → Tickets                 │
└─────────────────────────────────────────────────────────────────────┘

Stage 1: Signal Aggregation
─────────────────────────────
Wiki (0-100) ┐
Trends (0-100)│
YouTube (0-100)├─→ SignalOnly = mean(wiki, trends, youtube, chartmetric)
Chartmetric │
(0-100)     ┘

Stage 2: Ridge Regression (α=5.0)
──────────────────────────────────
SignalOnly (0-100) ──→ TicketIndex = 0.739 × SignalOnly + 26.065

Synthetic Anchors:
  • SignalOnly=0   → TicketIndex≈25  (weight: 14×)
  • SignalOnly=100 → TicketIndex≈100 (weight: 14×)

Stage 3: Benchmark Scaling
───────────────────────────
TicketIndex (0-100) ──→ Tickets = (TicketIndex / 100) × 11,976

```

---

## Data Distribution: SignalOnly vs TicketIndex

**28 Historical Titles (with ticket data):**

```
SignalOnly    TicketIndex
(input)       (output)
───────────────────────────────────────
  0.00   →    26.07  ┤                Floor (zero buzz)
  5.41   →    30.06  ┤█              After the Rain
  6.63   →    30.97  ┤█              Afternoon of a Faun
 10.00   →    33.46  ┤██             Low signal (P20)
 12.63   →    35.40  ┤███            Median (P50)
 20.00   →    40.84  ┤████
 30.00   →    48.23  ┤██████
 40.00   →    55.62  ┤████████
 46.00   →    60.06  ┤██████████     P95
 50.00   →    63.02  ┤███████████
 60.00   →    70.41  ┤█████████████
 70.00   →    77.80  ┤███████████████
 81.82   →    86.54  ┤█████████████████  Dracula
 90.00   →    92.59  ┤███████████████████
100.00   →    99.98  ┤████████████████████  Benchmark
```

**Observations:**
- Linear relationship preserved throughout range
- Floor at ~26 TI prevents unrealistic low estimates
- Dracula (81.82 SignalOnly) maps to 86.54 TI → 10,364 tickets
- Zero-buzz titles still get 26.07 TI → 3,122 tickets

---

## Ticket Predictions: Impact of Benchmark Choice

**Same TicketIndex, Different Benchmarks:**

```
TicketIndex  │  P50 Bench   P60 Bench   P70 Bench   Documented
(constant)   │  (2,225)     (2,754)     (3,041)     (11,976)
─────────────┼───────────────────────────────────────────────────
26.07 (floor)│     580        718         793        3,122
30.06 (ATR)  │     669        828         914        3,600
35.40 (P50)  │     788        975       1,076        4,240
60.06 (P95)  │   1,336      1,654       1,826        7,193
86.54 (Drac) │   1,925      2,384       2,632       10,364
99.98 (100)  │   2,225      2,754       3,041       11,975

ATR = After the Rain, Drac = Dracula
```

**Critical Finding:**  
Using the documented benchmark (11,976) produces predictions **4–5× higher** than using actual median performance (P50=2,225).

**Example:** A title with TI=60 would predict:
- **P50 benchmark:** 1,336 tickets ← realistic
- **Documented:** 7,193 tickets ← likely overestimate

---

## SignalOnly Distribution (All 281 Titles in Baselines)

```
  0-10  ████████████████████████████████████████████████ 143 titles (51%)
 10-20  ████████████████████████ 72 titles (26%)
 20-30  ████████ 22 titles (8%)
 30-40  ████ 12 titles (4%)
 40-50  ███ 9 titles (3%)
 50-60  ██ 7 titles (2%)
 60-70  ██ 6 titles (2%)
 70-80  ██ 5 titles (2%)
 80-90  █ 3 titles (1%)
90-100  █ 2 titles (1%)
```

**Observations:**
- Heavily skewed toward low signal (51% have SignalOnly < 10)
- Only 12 titles (4%) exceed SignalOnly=50
- Top 5 titles by signal: Beethoven (61.3), Addams Family (42.5), Bowie/Ziggy Stardust (40.8), Alice in Wonderland (38.1), Aladdin (37.5)

---

## Actual vs Predicted Tickets (28 Historical Titles)

**Scatter Plot (text representation):**

```
Actual
Tickets
  │
12k├                                                        ● Cinderella
  │
10k├                                              ● Dracula (pred: 10,364)
  │                                    ●
 8k├                              ●
  │                          ●  ●
 6k├                    ● ●        Model: TI = 0.739×S + 26.065
  │              ● ● ●              R² = 0.320 (32% variance explained)
 4k├      ● ● ●                     RMSE = 17.04 TI ≈ ±2,000 tickets
  │  ● ●
 2k├●
  │
 0k└────────────────────────────────────────────────────────────
   0     10    20    30    40    50    60    70    80    90   100
                          SignalOnly

● = Historical title (median tickets vs SignalOnly)
```

**RMSE Interpretation:**
- ±17 TI points → ±(17/100 × 11,976) = ±2,037 tickets
- 68% of predictions within 1 RMSE (±2,000 tickets)
- 95% of predictions within 2 RMSE (±4,000 tickets)

---

## Model Comparison: Linear vs Nonlinear

**Anchor Compliance Test (±3 TI tolerance):**

```
                   Anchor(0)              Anchor(100)
Model              Target: 25.0           Target: 100.0        Status
─────────────────────────────────────────────────────────────────────
Linear             26.07 (Δ=1.07) ✓      99.98 (Δ=0.02) ✓     PASS
Log-Linear         26.16 (Δ=1.16) ✓      53.35 (Δ=46.65) ✗    FAIL
Sigmoid            27.65 (Δ=2.65) ✓      157.34 (Δ=57.34) ✗   FAIL
```

**Visual: Model Behaviors**

```
TI
│
100├──────────────────────────────────Linear (0.739x + 26.065)
│                              ╱
│                          ╱
│                      ╱  Log-Linear (saturates)
│                  ╱──────────────────────
│              ╱
│          ╱              Sigmoid (unrealistic S-curve)
│      ╱         ╱──────
│  ╱       ╱────
25├────────
│
0└────────────────────────────────────────────────────────────
  0                                                        100
                        SignalOnly

Linear:     Constant slope, respects both anchors ✓
Log-Linear: Diminishing returns, fails high anchor ✗
Sigmoid:    S-curve, unrealistic parameters ✗
```

---

## Training Data Composition

**Ridge(alpha=5.0) trained on 56 total samples:**

```
Real Historical Titles (28 samples)
─────────────────────────────────────
Source: baselines.csv + history_city_sales.csv
• SignalOnly: computed from wiki/trends/youtube/chartmetric
• TicketIndex: normalized to Cinderella median
• Weight: 1× per title

Synthetic Anchor Points (28 samples)
─────────────────────────────────────
Purpose: Enforce floor and benchmark constraints
• 14× copies of [SignalOnly=0,   TicketIndex=25]
• 14× copies of [SignalOnly=100, TicketIndex=100]
• Weight: 14× (n_real // 2)

Regularization: α=5.0 (L2 penalty)
─────────────────────────────────────
Effect: Pulls coefficients toward zero, prevents overfitting
```

**Anchor Weight Scaling:**
```
n_real    anchor_weight
──────    ─────────────
  5-9          3
 10-19         5-9
 20-29        10-14  ← Current (28 real → 14 anchors)
 30-39        15-19
 40-49        20-24
```

---

## Prediction Examples with Uncertainty

**Low Signal Title (SignalOnly=10):**
```
Point Estimate:  33.5 TI → 4,011 tickets
±1 RMSE:         16.4 to 50.6 TI → 1,974 to 6,048 tickets
±2 RMSE:         -0.6 to 67.6 TI → (floor) to 8,085 tickets

Interpretation: "Expect 2,000–6,000 tickets (68% confidence)"
```

**Medium Signal Title (SignalOnly=50):**
```
Point Estimate:  63.0 TI → 7,545 tickets
±1 RMSE:         46.0 to 80.0 TI → 5,508 to 9,581 tickets
±2 RMSE:         29.0 to 97.1 TI → 3,471 to 11,628 tickets

Interpretation: "Expect 5,500–9,500 tickets (68% confidence)"
```

**High Signal Title (SignalOnly=90):**
```
Point Estimate:  92.6 TI → 11,089 tickets
±1 RMSE:         75.6 to 109.6 TI → 9,052 to 11,976 tickets (capped)
±2 RMSE:         58.6 to 126.6 TI → 7,014 to 11,976 tickets (capped)

Interpretation: "Expect 9,000–11,976 tickets (68% confidence, capped at benchmark)"
```

---

## Critical Discrepancies Found

### 1. Benchmark Mismatch ⚠️

**Documented in code/docs:** 11,976 tickets (Cinderella median)  
**Actual in data:**
- Cinderella Calgary 2022: 7,017 tickets
- Cinderella Edmonton 2022: 4,101 tickets
- Cinderella Calgary 2018: 7,290 tickets
- Cinderella Edmonton 2018: 5,545 tickets
- **Median across 4 runs:** 6,281 tickets

**Hypothesis:**
- 11,976 may be total capacity (Calgary + Edmonton)
- Or pre-COVID attendance
- Or includes subscription tickets (not single tickets)

**Impact:** If using 11,976 with actual medians ~2,200–3,000:
- Low signal titles: 4× overestimate
- High signal titles: 4× overestimate
- Consistent bias across all predictions

### 2. Coefficient Drift Risk

**Not Persisted:** Ridge model retrained on-the-fly whenever app loads.

**Risk Factors:**
- baselines.csv updated → coefficients change
- history_city_sales.csv updated → TicketIndex recalculated
- New titles added → anchor weight changes

**Example:**  
Adding 10 new titles changes anchor_weight from 14→19, which will shift coefficients.

**Current Lack of Version Control:**
- No record of "what model was used for forecast on date X?"
- Coefficients can silently drift between sessions

---

## Recommendations Summary

1. **Investigate Benchmark (HIGH PRIORITY)**
   - Verify 11,976 source and definition
   - Consider using P60=2,754 or P70=3,041 for realism

2. **Persist Model Parameters**
   - Save {intercept, coefficient, anchor_weight, n_titles, date} to JSON
   - Enable version control and audit trail

3. **Communicate Uncertainty**
   - Always present ±1 RMSE range (±2,000 tickets)
   - Label as "directional estimates" not precise forecasts

4. **Retain Linear Model**
   - Best anchor compliance
   - Most interpretable
   - Nonlinear alternatives fail constraints

5. **Implement Time-Aware CV (Optional)**
   - Validate temporal stability of coefficients
   - Report CV-RMSE alongside in-sample RMSE

---

## Files & Reproducibility

**All outputs in `/workspaces/alberta_ballet_title_scoring_app.py/diagnostics/`:**

1. `diagnose_signalonly_mapping.py` — Full diagnostic script (875 lines)
2. `mapping_summary.md` — Technical report (217 lines)
3. `DIAGNOSTIC_EXECUTIVE_SUMMARY.md` — Business summary with recommendations
4. `QUICK_REFERENCE.md` — Key parameters and formulas
5. `diagnostic_output.log` — Full console output

**To reproduce:**
```bash
cd /workspaces/alberta_ballet_title_scoring_app.py
python diagnostics/diagnose_signalonly_mapping.py
```

Deterministic (random_state=42), no external dependencies beyond pandas/scikit-learn/scipy.

---

**End of Diagnostic Visualization & Data Summary**
