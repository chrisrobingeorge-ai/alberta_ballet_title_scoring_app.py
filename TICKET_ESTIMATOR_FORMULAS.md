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
10. [Live Analytics Integration](#10-live-analytics-integration)
11. [Composite Score & Final Tickets](#11-composite-score--final-tickets)
12. [ML Model Variables](#12-ml-model-variables)
13. [Configuration Constants](#13-configuration-constants)
14. [Ticket Estimator Export: Diagnostic & Contextual Fields](#14-ticket-estimator-export-diagnostic--contextual-fields)

---

## 1. Input Signal Variables

### Raw Signal Indices
Each title has four online visibility signal scores that measure public awareness and engagement:

| Signal | Source | Index Formula | Description |
|--------|--------|--------------|-------------|
| **WikiIdx** | Wikipedia API | `40 + min(110, 20 × ln(1 + views/day))` | Average daily pageviews over the past year |
| **TrendsIdx** | Google Trends | Proxy heuristic (scaled 0-100) | Search interest relative to peak |
| **YouTubeIdx** | YouTube Data API | `50 + min(90, 9 × ln(1 + median_views))` | Median view counts across relevant videos |
| **chartmetricIdx** | chartmetric API | 80th percentile track popularity (0-100) | Track popularity near the query |

### YouTube Winsorization
YouTube scores are **winsorized by category** to prevent outliers from dominating:
- Clip to 3rd-97th percentile within each show category

---

## 2. Familiarity & Motivation Formulas

### Raw Score Calculation (Pre-Normalization)

**Familiarity** (how well-known the title is):
```
Familiarity_Raw = 0.55 × WikiIdx + 0.30 × TrendsIdx + 0.15 × chartmetricIdx
```

| Component | Weight |
|-----------|--------|
| Wikipedia Index | **55%** |
| Google Trends Index | **30%** |
| chartmetric Index | **15%** |

**Motivation** (how engaged people are with it):
```
Motivation_Raw = 0.45 × YouTubeIdx + 0.25 × TrendsIdx + 0.15 × chartmetricIdx + 0.15 × WikiIdx
```

| Component | Weight |
|-----------|--------|
| YouTube Index | **45%** |
| Google Trends Index | **25%** |
| chartmetric Index | **15%** |
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

**Current Model (December 2024 Update):** Constrained Ridge Regression

To prevent inflated predictions for low-buzz titles, the system now uses **constrained Ridge regression** for all dataset sizes. The model is anchored with synthetic data points to enforce realistic behavior:

| Dataset Size | Overall Model | Category Models | Rationale |
|--------------|---------------|-----------------|-----------|
| ≥5 samples | **Constrained Ridge (α=5.0)** | **Constrained Ridge (α=5.0)** | Regularized with anchor points |
| 3-4 samples | **Constrained Linear** | **Constrained Linear** | Linear fit with anchor constraints |
| <3 samples | Simple linear fallback | N/A | Insufficient data |

**Anchor Points:**
- `SignalOnly = 0` → `TicketIndex = 25` (realistic floor for minimal online presence)
- `SignalOnly = 100` → `TicketIndex = 100` (benchmark alignment)

These anchor points are weighted and added to the training data to guide the regression model, preventing the high intercept problem that previously caused low-signal titles to be overestimated by ~30%.

### Constrained Ridge Regression Model
For unknown titles, the model predicts TicketIndex from SignalOnly using a constrained Ridge regression:
```
TicketIndex_DeSeason_Predicted = Ridge_Model.predict(SignalOnly)
```

**Model Formula (typical):**
```
TicketIndex ≈ 0.75 × SignalOnly + 27.3
```

**Key Features:**
- Ridge regularization (α=5.0) prevents overfitting
- Anchor points enforce: low buzz → low index, benchmark buzz → index=100
- Prevents the "high intercept problem" that inflated estimates for obscure titles
- All predictions are **clipped to [20, 180]** range

### Linear Regression Formula (Fallback for <3 samples)
```
TicketIndex = a × SignalOnly + b
```
Where a, b are fit using anchor-constrained polyfit
- Same anchor points applied to prevent unrealistic intercepts

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

> **⚠️ REMOVED (December 2024)**: Per external audit finding "Structural Pessimism", the remount decay factor has been **eliminated**. The compounding of this factor with Post-COVID adjustments caused up to 33% reduction in valid predictions. The base ML model already accounts for remount behavior through the `is_remount_recent`, `is_remount_medium`, and `years_since_last_run` features.

~~Remount decay reduces estimates for titles that ran recently:~~

| Years Since Last Run | Decay Percentage | Decay Factor |
|---------------------|------------------|--------------|
| ~~< 1 year~~ | ~~25%~~ | ~~0.75~~ |
| ~~1-2 years~~ | ~~20%~~ | ~~0.80~~ |
| ~~3-4 years~~ | ~~12%~~ | ~~0.88~~ |
| ~~≥ 5 years~~ | ~~5%~~ | ~~0.95~~ |
| All titles | 0% | **1.00** |

### Current Formula
```
ReturnDecayFactor = 1.0  # Always 1.0 - no decay applied
EstimatedTickets_After_Remount = EstimatedTickets  # No reduction
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

### Arts Sentiment (Supplemental Economic Indicator)
Arts-specific sentiment derived from Nanos survey data on arts giving, integrated as a supplemental economic factor:

| Year | Arts Sentiment | Source |
|------|----------------|--------|
| 2023 | 11.0% | Nanos survey - % of charitable donation going to arts |
| 2024 | 12.0% | Nanos survey - % of charitable donation going to arts |
| 2025 | 12.0% | Nanos survey - % of charitable donation going to arts |

The `Econ_ArtsSentiment` feature is merged to show data by year using `merge_asof` with forward-fill logic for future dates.

### Temporal Join Strategy (December 2025 Update)
All economic feature joins now use **temporal matching** via `pd.merge_asof`:

1. Sort shows by `opening_date`
2. Sort economic indicator data by date
3. Use `pd.merge_asof` with `direction='backward'` to match each show to the most recent prior economic reading
4. Fallback to median value for shows without dates (cold-start titles)

This ensures economic features vary by when the show runs, providing time-varying signals for the model:

| Feature | Unique Values | Range | Notes |
|---------|---------------|-------|-------|
| `energy_index` | ~9 | [704, 1867] | BCPI Energy commodity prices |
| `inflation_adjustment_factor` | ~31 | [0.94, 1.21] | CPI-based adjustment relative to 2020 baseline |
| `consumer_confidence_prairies` | ~2 | [50.0, 50.4] | Nanos consumer confidence for Prairies (limited source data) |
| `city_median_household_income` | 1 | 98,000 | Static census value (not time-varying) |

---

## 10. Live Analytics Integration

### Category Engagement Factors
Live Analytics data from audience research provides category-specific engagement adjustments:

| Feature | Description | Default |
|---------|-------------|---------|
| **LA_EngagementFactor** | Category engagement multiplier | 1.0 |
| **LA_HighSpenderIdx** | High spender index (100 = average) | 100 |
| **LA_ActiveBuyerIdx** | Active buyer index (100 = average) | 100 |
| **LA_RepeatBuyerIdx** | Repeat buyer index (100 = average) | 100 |
| **LA_ArtsAttendIdx** | Arts attendance index (100 = average) | 100 |

### Engagement Factor Calculation
```
1. Load raw indices from live_analytics.csv per category
2. Compute raw_factor = mean(HighSpenderIdx, ActiveBuyerIdx, RepeatBuyerIdx, ArtsAttendIdx) / 100
3. Dampen: engagement = 1.0 + (raw_factor - 1.0) × 0.25
4. Clip to range: engagement = clip(engagement, 0.92, 1.08)
```

### Sample Category Engagement Factors

| Category | Engagement Factor | High Spender Idx | Active Buyer Idx |
|----------|-------------------|------------------|------------------|
| pop_ip | ~1.05 | 164 | 156 |
| classic_romance | ~1.02 | 145 | 140 |
| family_classic | ~1.03 | 148 | 150 |
| contemporary | ~0.98 | 102 | 108 |
| dramatic | ~0.97 | 95 | 98 |

### Addressable Market Features
Additional features from live analytics:

| Feature | Description |
|---------|-------------|
| **LA_AddressableMarket** | Raw customer count per category |
| **LA_AddressableMarket_Norm** | Normalized value (0-1 scale) |

---

## 11. Composite Score & Final Tickets

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

> **⚠️ UPDATED (December 2024)**: Per external audit finding "Structural Pessimism", both the Remount Decay Factor and Post-COVID Factor have been **removed** (set to 1.0). The compounding of these factors caused up to 33% reduction in valid predictions. The Region Factor is retained for geographical variance.

```
# Previous formula (DEPRECATED):
# EstimatedTickets_Final = EstimatedTickets × ReturnDecayFactor × POSTCOVID_FACTOR

# Current formula:
EstimatedTickets_Final = EstimatedTickets  # No penalty factors applied
```

### Post-COVID Factor

> **⚠️ REMOVED (December 2024)**: The Post-COVID factor has been eliminated to prevent "Structural Pessimism".

```yaml
# From config.yaml
demand:
  # postcovid_factor REMOVED per audit finding "Structural Pessimism"
  # Setting to 1.0 eliminates the compounding penalty that reduced predictions by up to 33%
  postcovid_factor: 1.0  # No haircut applied (was 0.85)
```

---

## 12. ML Model Variables

### XGBoost Feature Set (Safe Model)
Features used in the trained XGBoost model:

| Feature | Description | Allowed |
|---------|-------------|---------|
| prior_total_tickets | Historical tickets from prior seasons | ✅ Yes |
| ticket_median_prior | Median tickets from all prior runs | ✅ Yes |
| trends | Google Trends signal | ✅ Yes |
| youtube | YouTube signal | ✅ Yes |
| wiki | Wikipedia signal | ✅ Yes |
| chartmetric | chartmetric signal | ✅ Yes |
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
k: 5                    # Number of nearest neighbors
metric: "cosine"        # Distance metric: cosine, euclidean, manhattan
weights: "distance"     # Voting weights: distance or uniform
normalize: true         # Standardize features before distance computation
recency_weight: 0.5     # Weight for preferring recent shows (0-1)
recency_decay: 0.1      # Decay rate per year for older runs
use_pca: false          # Apply PCA preprocessing
pca_components: 3       # Number of PCA components (if use_pca=true)
```

### k-NN Features Used
The k-NN similarity matching uses these baseline signals:
- `wiki` - Wikipedia pageview index
- `trends` - Google Trends index
- `youtube` - YouTube view index
- `chartmetric` - chartmetric popularity index

---

## 13. Configuration Constants

### Demand Settings
```yaml
# postcovid_factor REMOVED per audit finding "Structural Pessimism"
postcovid_factor: 1.0  # No haircut applied (was 0.85)
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

## 14. Ticket Estimator Export: Diagnostic & Contextual Fields

The ticket estimator export CSV includes additional non-mathematical diagnostic fields that surface underlying model inputs and context. These fields help analysts interpret and validate predictions.

### Show & Audience Context

| Field | Type | Description |
|-------|------|-------------|
| **lead_gender** | string | Gender of lead character(s): "male", "female", "co-lead", or "n/a" |
| **dominant_audience_segment** | string | Primary audience segment (e.g., "Family (Parents w/ kids)", "Core Classical (F35–64)") |
| **segment_weights** | JSON | Segment weight distribution as JSON object (e.g., `{"Family": 0.6, "Emerging Adults": 0.2, ...}`) |

### Model & Historical Inputs

| Field | Type | Description |
|-------|------|-------------|
| **ticket_median_prior** | float | Median tickets from prior runs (from TICKET_PRIORS_RAW) |
| **prior_total_tickets** | int | Sum of all ticket values from prior runs |
| **run_count_prior** | int | Number of previous runs for this title |
| **TicketIndex_Predicted** | float | Raw ML model prediction before seasonality adjustment |
| **TicketIndexSource** | string | Source of ticket index: "History", "ML Category", "ML Overall", "Linear Fallback", "kNN", "Not enough data" |

### Temporal & Seasonality Info

| Field | Type | Description |
|-------|------|-------------|
| **month_of_opening** | int | Opening month (1-12) |
| **holiday_flag** | boolean | True if show opens during holiday season (Nov-Jan) |
| **category_seasonality_factor** | float | Raw seasonality factor for the category |
| **FutureSeasonalityFactor** | float | Applied seasonality factor (after shrinkage and clipping) |

### Live Analytics Signals

| Field | Type | Description |
|-------|------|-------------|
| **LA_EngagementFactor** | float | Category engagement multiplier |
| **LA_HighSpenderIdx** | float | High spender index (100 = average) |
| **LA_ActiveBuyerIdx** | float | Active buyer index (100 = average) |
| **LA_RepeatBuyerIdx** | float | Repeat buyer index (100 = average) |
| **LA_ArtsAttendIdx** | float | Arts attendance index (100 = average) |
| **LA_Category** | string | Category as mapped from live_analytics.csv |

### k-NN Metadata

| Field | Type | Description |
|-------|------|-------------|
| **kNN_used** | boolean | True if k-NN fallback was used for prediction |
| **kNN_neighbors** | JSON array | Array of neighbor titles used (top k, if kNN was used) |

### Purpose

These diagnostic fields enable:
- **Validation**: Compare model inputs to expected values
- **Debugging**: Identify why specific estimates differ from expectations  
- **Correlation Analysis**: Investigate patterns (e.g., "are female-led shows underestimated for Family audiences?")
- **Model Improvement**: Identify titles that consistently fallback to the same neighbors

### Non-Breaking Change
Adding these fields to the export is a non-breaking change. They are purely informational and do not affect:
- Ticket prediction math
- Model architecture
- UI presentation of core results

---

## Summary: End-to-End Calculation Flow

```
1. Collect Signals
   ├── Wikipedia pageviews → WikiIdx
   ├── Google Trends → TrendsIdx  
   ├── YouTube views → YouTubeIdx (winsorized by category)
   └── chartmetric popularity → chartmetricIdx

2. Compute Raw Scores
   ├── Familiarity_Raw = 0.55×Wiki + 0.30×Trends + 0.15×chartmetric
   └── Motivation_Raw = 0.45×YouTube + 0.25×Trends + 0.15×chartmetric + 0.15×Wiki

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

9. [REMOVED] Remount Decay
   └── No longer applied (was: EstimatedTickets × decay_factor)
   └── Remount behavior now captured in ML model features

10. [REMOVED] Post-COVID Factor
    └── No longer applied (was: × 0.85)
    └── Factor set to 1.0 per audit finding "Structural Pessimism"

11. Final Tickets = EstimatedTickets (no penalty factors)

12. Split by City (Region Factor RETAINED)
    ├── YYC = Final × CityShare_Calgary (default 60%)
    └── YEG = Final × CityShare_Edmonton (default 40%)
    └── Region multipliers: Calgary 1.05, Edmonton 0.95

13. Calculate Marketing Spend
    ├── YYC_Mkt = YYC_Singles × SPT_Calgary
    └── YEG_Mkt = YEG_Singles × SPT_Edmonton

14. Apply Live Analytics Overlays (per category)
    ├── LA_EngagementFactor → category engagement multiplier
    ├── LA_HighSpenderIdx → high spender index
    ├── LA_ActiveBuyerIdx → active buyer index
    ├── LA_RepeatBuyerIdx → repeat buyer index
    └── LA_ArtsAttendIdx → arts attendance index

15. Apply Economic Sentiment (supplemental context, time-varying)
    ├── Econ_BocFactor → Bank of Canada sentiment
    ├── Econ_AlbertaFactor → Alberta economic sentiment
    ├── Econ_ArtsSentiment → Arts giving sentiment
    ├── energy_index → BCPI Energy (temporal, range 704-1867)
    ├── inflation_adjustment_factor → CPI-based (temporal, range 0.94-1.21)
    └── Combined_Sentiment = 0.40×BoC + 0.60×Alberta
```

---

*Last updated: December 2025*
*Audit update: Removed Post-COVID Factor and Remount Decay to fix "Structural Pessimism"*
*December 2025 update: Economic features now use temporal joins (pd.merge_asof) for time-varying data*
