# Ticket Estimator: Complete Variable & Formula Reference

This document provides a comprehensive listing of all variables, weightings, and formulas used in the Alberta Ballet ticket estimator calculations.

---

## Table of Contents
1. [Input Signal Variables](#1-input-signal-variables)
2. [Familiarity & Motivation Formulas](#2-familiarity--motivation-formulas)
3. [Segment & Region Multipliers](#3-segment--region-multipliers)
4. [Ticket Index Calculation](#4-ticket-index-calculation)
5. [Seasonality Factors](#5-seasonality-factors)
6. [Remount Decay Formula](#6-remount-decay-formula)
7. [City Split Calculation](#7-city-split-calculation)
8. [Marketing Spend Calculation](#8-marketing-spend-calculation)
9. [Economic Sentiment Factors](#9-economic-sentiment-factors)
10. [Composite Score & Final Tickets](#10-composite-score--final-tickets)
11. [ML Model Variables](#11-ml-model-variables)
12. [Configuration Constants](#12-configuration-constants)

---

## 1. Input Signal Variables

### Raw Signal Indices
Each title has four online visibility signal scores that measure public awareness and engagement:

| Signal | Source | Index Formula | Description |
|--------|--------|--------------|-------------|
| **WikiIdx** | Wikipedia API | `40 + min(110, 20 × ln(1 + views/day))` | Average daily pageviews over the past year |
| **TrendsIdx** | Google Trends | Proxy heuristic (scaled 0-100) | Search interest relative to peak |
| **YouTubeIdx** | YouTube Data API | `50 + min(90, 9 × ln(1 + median_views))` | Median view counts across relevant videos |
| **SpotifyIdx** | Spotify API | 80th percentile track popularity (0-100) | Track popularity near the query |

### YouTube Winsorization
YouTube scores are **winsorized by category** to prevent outliers from dominating:
- Clip to 3rd-97th percentile within each show category

---

## 2. Familiarity & Motivation Formulas

### Raw Score Calculation (Pre-Normalization)

**Familiarity** (how well-known the title is):
```
Familiarity_Raw = 0.55 × WikiIdx + 0.30 × TrendsIdx + 0.15 × SpotifyIdx
```

| Component | Weight |
|-----------|--------|
| Wikipedia Index | **55%** |
| Google Trends Index | **30%** |
| Spotify Index | **15%** |

**Motivation** (how engaged people are with it):
```
Motivation_Raw = 0.45 × YouTubeIdx + 0.25 × TrendsIdx + 0.15 × SpotifyIdx + 0.15 × WikiIdx
```

| Component | Weight |
|-----------|--------|
| YouTube Index | **45%** |
| Google Trends Index | **25%** |
| Spotify Index | **15%** |
| Wikipedia Index | **15%** |

### Benchmark Normalization
All scores are normalized to a benchmark title (benchmark = 100):
```
Familiarity = (Familiarity_Raw / Benchmark_Familiarity_Raw) × 100
Motivation = (Motivation_Raw / Benchmark_Motivation_Raw) × 100
```

### SignalOnly Score
Simple average of normalized scores:
```
SignalOnly = (Familiarity + Motivation) / 2
```

---

## 3. Segment & Region Multipliers

### Segment Multipliers by Gender
Multipliers applied to scores based on audience segment and show gender:

| Segment | Female | Male | Co-Lead | N/A |
|---------|--------|------|---------|-----|
| General Population | 1.00 | 1.00 | 1.00 | 1.00 |
| Core Classical (F35–64) | 1.12 | 0.95 | 1.05 | 1.00 |
| Family (Parents w/ kids) | 1.10 | 0.92 | 1.06 | 1.00 |
| Emerging Adults (18–34) | 1.02 | 1.02 | 1.00 | 1.00 |

### Segment Multipliers by Category
All categories from config.yaml:

| Segment | family_classic | classic_romance | romantic_tragedy | classic_comedy | contemporary | pop_ip | dramatic | adult_literary_drama | contemporary_mixed_bill | touring_contemporary_company |
|---------|----------------|-----------------|------------------|----------------|--------------|--------|----------|---------------------|------------------------|------------------------------|
| General Population | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Core Classical (F35–64) | 1.10 | 1.08 | 1.05 | 1.02 | 0.90 | 1.00 | 1.00 | 1.00 | 0.90 | 0.90 |
| Family (Parents w/ kids) | 1.18 | 0.95 | 0.85 | 1.05 | 0.82 | 1.20 | 0.90 | 0.90 | 0.82 | 0.82 |
| Emerging Adults (18–34) | 0.95 | 0.92 | 0.90 | 0.98 | 1.25 | 1.15 | 1.05 | 1.05 | 1.25 | 1.25 |

### Region Multipliers
```
Province: 1.00
Calgary: 1.05
Edmonton: 0.95
```

### Combined Segment Multiplier Formula
```
Segment_Mult = SEGMENT_MULT[segment][gender] × SEGMENT_MULT[segment][category] × REGION_MULT[region]
```

---

## 4. Ticket Index Calculation

### De-Seasonalized Ticket Index
Historical ticket medians are de-seasonalized to create a comparable index:
```
TicketIndex_DeSeason = (Title_Median_Tickets / Historical_Month_Factor) / Benchmark_Median_Deseason × 100
```

### Regression Model Selection (Based on Sample Size)

| Dataset Size | Overall Model | Category Models | Rationale |
|--------------|---------------|-----------------|-----------|
| ≥8 samples | XGBoost (n_estimators=100, max_depth=3) | Ridge Regression (α=1.0) | Best for non-linear patterns |
| 5-7 samples | GradientBoosting (n_estimators=50, max_depth=2) | Ridge Regression (α=1.0) | Balanced accuracy |
| 3-4 samples | Use category model | Linear Regression | Avoid overfitting |
| <3 samples | Simple linear fallback | N/A | Insufficient data |

### ML Model Prediction
For unknown titles, the model predicts TicketIndex from SignalOnly:
```
TicketIndex_DeSeason_Predicted = ML_Model.predict(SignalOnly)
```
- Predictions are **clipped to [20, 180]** range

### Linear Regression Formula (Fallback)
```
TicketIndex = a × SignalOnly + b
```
Where a, b are fit parameters from known titles

---

## 5. Seasonality Factors

### Category × Month Factor
Seasonality factors adjust ticket expectations based on when the show runs:

```
Raw_Factor = Month_Category_Median / Overall_Category_Median
```

### Shrinkage Formula
Low-sample months are shrunk toward 1.0:
```
Shrinkage_Weight = n / (n + K_SHRINK)
Shrunk_Factor = Shrinkage_Weight × Raw_Factor + (1 - Shrinkage_Weight) × 1.0
```

### Clipping
```
Final_Factor = clip(Shrunk_Factor, MIN_FACTOR, MAX_FACTOR)
```

### Seasonality Constants
| Constant | Value | Description |
|----------|-------|-------------|
| **K_SHRINK** | 3.0 | Shrinkage factor for low-sample months |
| **MIN_FACTOR** | 0.90 | Minimum seasonality multiplier |
| **MAX_FACTOR** | 1.15 | Maximum seasonality multiplier |
| **N_MIN** | 3 | Minimum samples to trust a month factor |

### Effective Ticket Index
```
EffectiveTicketIndex = TicketIndex_DeSeason × FutureSeasonalityFactor
```

---

## 6. Remount Decay Formula

Remount decay reduces estimates for titles that ran recently:

| Years Since Last Run | Decay Percentage | Decay Factor |
|---------------------|------------------|--------------|
| < 1 year | 25% | 0.75 |
| 1-2 years | 20% | 0.80 |
| 3-4 years | 12% | 0.88 |
| ≥ 5 years | 5% | 0.95 |
| Never run before | 0% | 1.00 |

### Decay Formula
```
ReturnDecayFactor = 1.0 - ReturnDecayPct
EstimatedTickets_After_Remount = EstimatedTickets × ReturnDecayFactor
```

---

## 7. City Split Calculation

### Priority Order for City Split
1. **Title-level split** (learned from historical data for this specific title)
2. **Category-level split** (weighted average for this category)
3. **Default split** (Calgary 60% / Edmonton 40%)

### Default City Split
```yaml
Calgary: 60%
Edmonton: 40%
```

### Clipping Range
City shares are clipped to prevent extreme allocations:
```
City Share = clip(Learned_Share, 0.15, 0.85)
```

### City Ticket Allocation
```
YYC_Total = EstimatedTickets_Final × CityShare_Calgary
YEG_Total = EstimatedTickets_Final × CityShare_Edmonton
YYC_Singles = YYC_Total  # All tickets are single tickets in this app
YEG_Singles = YEG_Total
```

---

## 8. Marketing Spend Calculation

### Marketing Spend Per Ticket (SPT)
Priority order for determining $/ticket:
1. **Title × City median** (e.g., "Cinderella in Calgary")
2. **Category × City median** (e.g., "classic_romance in Edmonton")
3. **City-wide default** (Calgary: $10.00, Edmonton: $8.00)

### Default Marketing SPT
```yaml
Calgary: $10.00 per single ticket
Edmonton: $8.00 per single ticket
```

### Marketing Budget Calculation
```
YYC_Mkt_Spend = YYC_Singles × YYC_Mkt_SPT
YEG_Mkt_Spend = YEG_Singles × YEG_Mkt_SPT
Total_Mkt_Spend = YYC_Mkt_Spend + YEG_Mkt_Spend
```

---

## 9. Economic Sentiment Factors

### Combined Sentiment Weights
When both sources are available:
```
Combined_Sentiment = 0.40 × BoC_Sentiment + 0.60 × Alberta_Sentiment
```

### Bank of Canada (BoC) Indicators

| Series | Weight | Direction | Baseline |
|--------|--------|-----------|----------|
| Policy Rate | 15% | Negative | 3.0% |
| CORRA | 5% | Negative | 3.0% |
| 2Y Bond Yield | 8% | Negative | 3.0% |
| 5Y Bond Yield | 8% | Negative | 3.2% |
| 10Y Bond Yield | 9% | Negative | 3.4% |
| BCPI Total | 10% | Positive | 130.0 |
| BCPI Energy | 25% | Positive | 130.0 |
| BCPI Ex-Energy | 10% | Positive | 110.0 |
| Core CPI | 10% | Negative | 2.0% |

### Alberta Economic Indicators

| Indicator | Weight | Direction | Baseline |
|-----------|--------|-----------|----------|
| Unemployment Rate | 12% | Negative | 7.0% |
| Employment Rate | 8% | Positive | 64.0% |
| Employment Level | 5% | Positive | 2,400K |
| Participation Rate | 5% | Positive | 70.0% |
| Avg Weekly Earnings | 10% | Positive | $1,200 |
| Consumer Price Index | 8% | Negative | 150.0 |
| WCS Oil Price | 15% | Positive | $60.00 USD |
| Retail Trade | 10% | Positive | $8B CAD |
| Restaurant Sales | 7% | Positive | $700M CAD |
| Air Passengers | 5% | Positive | 2M monthly |
| Net Migration | 8% | Positive | 15,000 quarterly |
| Population (Quarterly) | 7% | Positive | 4.5M |

### Sentiment Calculation Formula
```
1. Compute z-score for each indicator: z = (current - mean) / std
2. Apply direction: if negative indicator, flip sign
3. Weighted average: avg_z = Σ(weight × z-score)
4. Convert to factor: factor = 1.0 + (avg_z × sensitivity)
5. Clamp to bounds: sentiment = clip(factor, 0.85, 1.15)
```

### Sensitivity Parameters
```yaml
BoC Sensitivity: 0.10
Alberta Sensitivity: 0.08
```

---

## 10. Composite Score & Final Tickets

### Composite Index Formula
```
TICKET_BLEND_WEIGHT = 0.50

If TicketIndexSource == "Not enough data":
    tickets_component = SignalOnly
Else:
    tickets_component = EffectiveTicketIndex

Composite = (1 - TICKET_BLEND_WEIGHT) × SignalOnly + TICKET_BLEND_WEIGHT × tickets_component
```

### Estimated Tickets Formula
```
EstimatedTickets = (EffectiveTicketIndex / 100) × Benchmark_Tickets_DeSeasonalized
```

### Final Tickets (After All Adjustments)
```
EstimatedTickets_Final = EstimatedTickets × ReturnDecayFactor × POSTCOVID_FACTOR
```

### Post-COVID Factor
This value is loaded from `config.yaml` under `demand.postcovid_factor`:
```yaml
# From config.yaml
demand:
  postcovid_factor: 0.85  # 15% haircut vs pre-COVID baseline
```

---

## 11. ML Model Variables

### XGBoost Feature Set (Safe Model)
Features used in the trained XGBoost model:

| Feature | Description | Allowed |
|---------|-------------|---------|
| prior_total_tickets | Historical tickets from prior seasons | ✅ Yes |
| ticket_median_prior | Median tickets from all prior runs | ✅ Yes |
| trends | Google Trends signal | ✅ Yes |
| youtube | YouTube signal | ✅ Yes |
| wiki | Wikipedia signal | ✅ Yes |
| spotify | Spotify signal | ✅ Yes |
| familiarity | Computed familiarity score | ✅ Yes |
| motivation | Computed motivation score | ✅ Yes |
| is_remount_recent | Binary: ≤2 years since last run | ✅ Yes |
| is_remount_medium | Binary: 2-4 years since last run | ✅ Yes |
| years_since_last_run | Numeric years since last run | ✅ Yes |
| run_count_prior | Number of previous runs | ✅ Yes |
| month_of_opening | Opening month (1-12) | ✅ Yes |
| holiday_flag | Holiday overlap flag | ✅ Yes |
| postcovid_factor | Post-COVID adjustment | ✅ Yes |

### Forbidden Features (Data Leakage)
These columns are **NEVER** used as predictors:
- Single Tickets - Calgary/Edmonton
- Total Tickets / Total Single Tickets
- YourModel_* columns (current-run predictions)

### k-NN Cold-Start Fallback
For titles without history:
```yaml
k: 5  # Number of nearest neighbors
metric: "cosine"  # Distance metric
recency_weight: 0.5  # Weight for preferring recent shows
recency_decay: 0.1  # Decay rate per year
```

---

## 12. Configuration Constants

### Demand Settings
```yaml
postcovid_factor: 0.85  # Applied to all estimates
ticket_blend_weight: 0.50  # Balance signals vs ticket history
```

### Seasonality Settings
```yaml
k_shrink: 3.0  # Shrinkage for low-sample months
min_factor: 0.90  # Minimum seasonality multiplier
max_factor: 1.15  # Maximum seasonality multiplier
n_min: 3  # Minimum samples to trust month factor
```

### City Split Settings
```yaml
default_base_city_split:
  Calgary: 0.60
  Edmonton: 0.40
city_clip_range: [0.15, 0.85]
```

### Marketing Defaults
```yaml
default_marketing_spt_city:
  Calgary: 10.0
  Edmonton: 8.0
```

### Calibration Settings
```yaml
calibration:
  enabled: false
  mode: "global"  # Options: global, per_category, by_remount_bin
```

### Model Settings
```yaml
model:
  path: "models/model_xgb_remount_postcovid.joblib"
  use_for_cold_start: true
  confidence_threshold: 0.6  # R² threshold for KNN fallback
```

---

## Summary: End-to-End Calculation Flow

```
1. Collect Signals
   ├── Wikipedia pageviews → WikiIdx
   ├── Google Trends → TrendsIdx  
   ├── YouTube views → YouTubeIdx (winsorized by category)
   └── Spotify popularity → SpotifyIdx

2. Compute Raw Scores
   ├── Familiarity_Raw = 0.55×Wiki + 0.30×Trends + 0.15×Spotify
   └── Motivation_Raw = 0.45×YouTube + 0.25×Trends + 0.15×Spotify + 0.15×Wiki

3. Normalize to Benchmark (=100)
   ├── Familiarity = (Familiarity_Raw / Benchmark_Raw) × 100
   └── Motivation = (Motivation_Raw / Benchmark_Raw) × 100

4. Compute SignalOnly
   └── SignalOnly = (Familiarity + Motivation) / 2

5. Predict/Lookup TicketIndex
   ├── If historical data exists → use de-seasonalized median
   └── If unknown → ML model predicts from SignalOnly

6. Apply Future Seasonality
   └── EffectiveTicketIndex = TicketIndex_DeSeason × FutureSeasonalityFactor

7. Compute Composite
   └── Composite = 0.50×SignalOnly + 0.50×EffectiveTicketIndex

8. Estimate Raw Tickets
   └── EstimatedTickets = (EffectiveTicketIndex / 100) × Benchmark_Tickets

9. Apply Remount Decay
   └── EstimatedTickets × (1 - decay_pct)

10. Apply Post-COVID Factor
    └── EstimatedTickets_Final = EstimatedTickets_After_Remount × 0.85

11. Split by City
    ├── YYC = Final × CityShare_Calgary (default 60%)
    └── YEG = Final × CityShare_Edmonton (default 40%)

12. Calculate Marketing Spend
    ├── YYC_Mkt = YYC_Singles × SPT_Calgary
    └── YEG_Mkt = YEG_Singles × SPT_Edmonton
```

---

*Last updated: December 2024*
