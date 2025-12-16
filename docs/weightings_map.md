# Weightings Systems Map

**Date:** December 2, 2025  
**Purpose:** Document the three key weighting systems in the Alberta Ballet Title Scoring App

---

## Overview

The app integrates three distinct weighting systems that adjust ticket demand predictions:

1. **Live Analytics** - Audience engagement patterns from historical live event data
2. **Economics** - Macroeconomic factors (consumer confidence, energy prices, inflation)
3. **Stone Olafson** - Market research segment and region multipliers

---

## 1. Live Analytics Weightings

### Data Source
- **File:** `data/audiences/live_analytics.csv`
- **Format:** CSV with segment-level audience behavior indices
- **Columns:** Various behavioral metrics (e.g., Male/Female%, Age groups, Event counts, Spend patterns)
- **Coverage:** ~75 rows of segment/category data with index values (base 100)

### Data Ingestion
- **Module:** `data/loader.py`
- **Function:** `get_category_engagement_factor(category: str) -> float`
  - Loads live analytics data
  - Parses engagement indices for each show category
  - Returns category-specific multiplier (1.0 = neutral, >1.0 = higher engagement)

### Weighting Variables
- **Variable:** `aud__engagement_factor` (numeric, range ~0.5-2.0)
- **Computation:** `get_category_engagement_factor()` → category aliases → engagement index
- **Default:** 1.0 (neutral) if category not found

### Application Point
- **Stage:** Feature construction (pre-model)
- **Location:** `scripts/build_modelling_dataset.py`, line ~660
- **Method:** Applied as a column in the modelling dataset
  ```python
  df['aud__engagement_factor'] = df['category'].apply(
      lambda cat: get_category_engagement_factor(cat) if pd.notna(cat) else 1.0
  )
  ```
- **Impact:** Becomes a numeric feature used by Ridge regression model during training/prediction

### Streamlit App Usage
- **Location:** `streamlit_app.py`, `_add_live_analytics_overlays()` function (line ~2155)
- **Method:** Fetches live analytics percentages by category for display/reporting
- **Impact:** Informational overlays showing audience behavior patterns (e.g., mobile %, early buyer %, premium %)

---

## 2. Economic Weightings

### Data Sources

#### Consumer Confidence
- **File:** `data/economics/nanos_consumer_confidence.csv`
- **Format:** Time-series CSV with weekly confidence readings
- **Columns:** `category`, `subcategory`, `metric`, `year_or_period`, `value`
- **Coverage:** Nanos BNCCI Prairies regional data (2015-2025)

#### Energy Index
- **File:** `data/economics/commodity_price_index.csv`
- **Format:** Monthly commodity price data
- **Columns:** `date`, `A.ENER` (Energy commodity index)
- **Coverage:** Bank of Canada commodity index (monthly, 2015-2025)

#### Inflation Factor
- **File:** `data/economics/boc_cpi_monthly.csv`
- **Format:** Monthly Consumer Price Index
- **Columns:** `date`, `value`
- **Coverage:** Bank of Canada CPI data (monthly)

### Data Ingestion
- **Module:** `data/features.py`
- **Functions:**
  - `join_consumer_confidence(df, nanos_df, date_column='show_date') -> pd.DataFrame`
  - `join_energy_index(df, commodity_df, date_column='show_date') -> pd.DataFrame`
  - `compute_inflation_adjustment_factor(df, cpi_df, date_column='show_date', base_date='2019-01-01') -> pd.DataFrame`

### Weighting Variables
- **`consumer_confidence_prairies`:** Numeric, range ~50-51 (current data has limited variance)
- **`energy_index`:** Numeric, range ~700-1900 (Alberta energy price index)
- **`inflation_adjustment_factor`:** Numeric, range ~0.94-1.21 (CPI-based factor relative to 2019 baseline)

### Application Point
- **Stage:** Feature construction (pre-model)
- **Location:** `data/features.py::build_feature_store()` function
- **Method:** Temporal join using `pd.merge_asof(direction='backward')`
  - Each show is matched to the most recent prior economic data point
  - Features added as columns to the modelling dataset
- **Integration Point:** `scripts/build_modelling_dataset.py`
  - Calls `build_feature_store()` which applies all economic joins
  - Economic features become predictors in the Ridge regression model

### Historical Context
- **Previous Issue:** Economic features were flat/constant (all rows had same values)
- **Resolution:** Rewrote join functions (Dec 2025) to use temporal matching with `pd.merge_asof`
- **Current State:** Features now vary over time (validated in `ECONOMIC_FEATURES_WIRING_SUMMARY.md`)

---

## 3. Stone Olafson Weightings

### Data Source
- **File:** `data/audiences/live_analytics.csv` (same as Live Analytics but used differently)
- **Format:** CSV containing segment indices from Stone Olafson market research
- **Columns:** Segment-level indices (INDEX REPORT columns with values like 107, 103, 125, etc.)
- **Coverage:** Demographic and behavioral segments (Millennials, Gen X, Professionals, etc.)

### Data Ingestion
- **Module:** `streamlit_app.py` (embedded configuration)
- **Variables:**
  - `SEGMENT_MULT`: Dict of segment-specific multipliers for different genders and categories
  - `REGION_MULT`: Dict of region-specific multipliers (Calgary vs Edmonton)

### Weighting Variables

#### Segment Multipliers (`SEGMENT_MULT`)
Structure:
```python
SEGMENT_MULT = {
    "segment_key": {
        "male": 1.05,      # Gender multiplier
        "female": 0.98,
        "pop_ip": 1.20,    # Category multiplier
        "classic_romance": 1.10,
        ...
    }
}
```

#### Region Multipliers (`REGION_MULT`)
Structure:
```python
REGION_MULT = {
    "Calgary": 1.10,  # Calgary multiplier
    "Edmonton": 0.95   # Edmonton multiplier
}
```

### Application Point
- **Stage:** Score calculation (post-signal, pre-ticketing)
- **Location:** `streamlit_app.py::calc_scores()` function (line ~1814)
- **Method:**
  ```python
  def calc_scores(entry, seg_key, reg_key):
      # Base scores from wiki/trends/chartmetric/youtube
      fam = entry["wiki"] * 0.55 + entry["trends"] * 0.30 + entry["chartmetric"] * 0.15
      mot = entry["youtube"] * 0.45 + entry["trends"] * 0.25 + ...
      
      # Apply segment multipliers (Stone Olafson)
      seg = SEGMENT_MULT[seg_key]
      fam *= seg.get(gender, 1.0) * seg.get(category, 1.0)
      mot *= seg.get(gender, 1.0) * seg.get(category, 1.0)
      
      # Apply region multipliers
      fam *= REGION_MULT[reg_key]
      mot *= REGION_MULT[reg_key]
      
      return fam, mot
  ```
- **Impact:** Multiplies base familiarity/motivation scores to adjust for segment affinity and regional preferences

### Usage Pattern
- Scores are computed for each segment × region combination
- Results shown in segment-by-segment breakdowns
- Used for:
  - Ticket split allocation (Calgary vs Edmonton)
  - Segment appeal visualization
  - Demand forecasting by demographic

---

## Summary Table

| Weighting System | Data Source | Key Variables | Application Stage | Method |
|------------------|-------------|---------------|-------------------|---------|
| **Live Analytics** | `audiences/live_analytics.csv` | `aud__engagement_factor` | Pre-model (feature) | Category lookup → Ridge feature |
| **Economics** | `economics/nanos_*.csv`<br/>`economics/commodity_price_index.csv`<br/>`economics/boc_cpi_monthly.csv` | `consumer_confidence_prairies`<br/>`energy_index`<br/>`inflation_adjustment_factor` | Pre-model (feature) | Temporal join → Ridge features |
| **Stone Olafson** | `audiences/live_analytics.csv` (indices) | `SEGMENT_MULT` (dict)<br/>`REGION_MULT` (dict) | Post-signal (scoring) | Multipliers in `calc_scores()` |

---

## Key Differences

1. **Live Analytics** operates as a **feature** in the ML model (Ridge regression learns its weight)
2. **Economics** operates as **features** in the ML model (Ridge regression learns their weights)
3. **Stone Olafson** operates as **explicit multipliers** applied after base scores (deterministic, not learned)

**Critical Insight:** Live Analytics and Economics are learned by the model, while Stone Olafson weightings are hard-coded business rules that directly scale the final scores.
