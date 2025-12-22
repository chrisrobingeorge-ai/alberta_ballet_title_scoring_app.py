# Alberta Ballet Title Scoring Application: Technical ML Pipeline Report

**Document Type:** Engineering Analysis & Technical Audit  
**Generated:** December 20, 2025  
**Repository:** chrisrobingeorge-ai/alberta_ballet_title_scoring_app.py  
**Primary Implementation File:** `streamlit_app.py` (4140 lines)

---

## Executive Summary

The system implements a hybrid predictive architecture combining:

1. **Dynamically-Trained Ridge Regression** (locally trained on user-supplied historical data)
2. **LinearRegression Fallback** (when insufficient historical data exists)
3. **k-Nearest Neighbors fallback** for cold-start predictions
4. **Multi-factor digital signal aggregation** from Wikipedia, Google Trends, YouTube, and Chartmetric

The application trains Ridge regression models dynamically at runtime using user-provided historical production data.

The application predicts ballet production ticket sales by synthesizing online visibility metrics with historical performance data, applying seasonality adjustments, and decomposing forecasts by city (Calgary/Edmonton) and audience segment.

---

## 1. System Architecture Overview

### 1.1 Data Flow Pipeline

```
Raw Input Signals → Feature Engineering → ML Prediction → Post-Processing → City/Segment Split
     ↓                    ↓                    ↓               ↓                  ↓
  [Baselines]   [Normalization]    [Ridge/Linear]       [Seasonality]     [Learned Priors]
                 [Multipliers]                           [Decay Factors]
```

**File:** `streamlit_app.py:2830-4140` (main prediction pipeline)

Models are trained dynamically at runtime using user-supplied historical data.

### 1.2 Model Selection Hierarchy

The system employs a four-tier fallback strategy based on data availability (streamlit_app.py:3185-3225):

1. **Tier 1 - Historical Data** (`TicketIndexSource = "History"`)
   - Direct lookup from `BASELINES` dictionary containing 282 reference productions
   - Median ticket sales from prior runs used as ground truth

2. **Tier 2 - ML Models** (Dynamically-Trained Regression)
   - **Category-specific Ridge Regression models** (trained if ≥5 samples per category)
   - **Overall Ridge Regression model** (trained if ≥3 total samples available)
   - **Category-specific LinearRegression** (trained if 3-4 samples per category)
   - **Overall LinearRegression** (fallback if ≥3 total samples available)
   - Models are trained on demand at runtime

3. **Tier 3 - Signal-Only Estimate**
   - Falls back to `SignalOnly` composite score if all other predictions unavailable

---

## 2. Feature Engineering & Signal Extraction

### 2.1 Digital Signal Acquisition

**Data Source:** Manually curated baseline signals from external platforms (not live API calls)

**Implementation:** `streamlit_app.py:1982-2060`

Four primary signals are stored and referenced from the `baselines.csv` file:

#### Wikipedia Pageview Index

**Data Collection Period:** January 1, 2020 to present  
**Metric:** Average daily pageviews over the collection period

```python
# streamlit_app.py:2015-2030
wiki_raw = baseline_signals['wiki']
wiki_idx = 40.0 + min(110.0, (math.log1p(max(0.0, wiki_raw)) * 20.0))
```

**Formula:**  
$$\text{WikiIdx} = 40 + \min(110, \ln(1 + \text{views}_{\text{daily}}) \times 20)$$

**Range:** [40, 150] (log-scaled to dampen outlier influence)

#### Google Trends Index

**Data Collection Period:** January 1, 2022 to present  
**Metric:** Relative search volume (0-100 scale, normalized via bridge calibration with Giselle as control)

```python
# streamlit_app.py:2032
trends_idx = baseline_signals['trends']
```

**Bridge Calibration Protocol:** To normalize disparate Google Trends batches, a control title (Giselle) with known average score of 14.17 is used to rescale all batches to a common master axis.

#### YouTube Engagement Index

**Data Collection Period:** January 10, 2023 to present  
**Metric:** View counts from top-ranked YouTube videos, indexed against Cinderella benchmark

```python
# streamlit_app.py:2035-2055, 1940-1950
yt_value = baseline_signals['youtube']
yt_idx = 50.0 + min(90.0, np.log1p(max(0.0, yt_value)) * 9.0)
```

**Formula:**  
$$\text{YouTubeIdx} = 50 + \min(90, \ln(1 + \text{indexed view count}) \times 9)$$

**Indexing Logic:** Raw YouTube view counts are indexed against Cinderella as the benchmark title:
```python
YouTube_indexed_score = (view_count_title / view_count_cinderella) * 100
```

**Winsorization** (`streamlit_app.py:1958-1970`): YouTube indices are clipped to category-specific ranges (3rd to 97th percentile) to prevent viral videos from distorting forecasts.

#### Chartmetric Streaming Index

**Data Collection Period:** Last 2 years of data  
**Metric:** Weighted artist rank scores from streaming platforms (Spotify, Apple Music, Amazon, Deezer, YouTube Music, Shazam) and social platforms (TikTok, Instagram, Twitter/X, Facebook)

```python
# streamlit_app.py:2057-2065
cm_value = baseline_signals['chartmetric']
cm_idx = float(cm_value) if cm_value else 50.0
```

**Normalization:** Raw artist rank scores are inverted and normalized to 0-100 scale:
```python
Chartmetric_normalized = 100 * (1 - (artist_rank / max_rank))
```

### 2.2 Signal Winsorization

YouTube indices are clipped to category-specific ranges to prevent anomalies:

```python
# streamlit_app.py:1958-1970
def _winsorize_youtube_to_baseline(category: str, yt_value: float) -> float:
    ref = baseline_youtube_values_for_category[category]
    lo = np.percentile(ref, 3)
    hi = np.percentile(ref, 97)
    return np.clip(yt_value, lo, hi)
```

### 2.3 Composite Signal Construction

**Familiarity Index** (lines 2101-2105):
```python
fam = wiki * 0.55 + trends * 0.30 + chartmetric * 0.15
```

**Motivation Index** (lines 2101-2105):
```python
mot = youtube * 0.45 + trends * 0.25 + chartmetric * 0.15 + wiki * 0.15
```

**Rationale:**
- **Familiarity** weights Wikipedia heavily (static knowledge, "I've heard of it")
- **Motivation** weights YouTube heavily (active engagement, "I want to see it")

### 2.4 Segment & Region Multipliers

**File:** `config.yaml:1-80` → loaded into `SEGMENT_MULT` and `REGION_MULT` dictionaries

```python
# streamlit_app.py:2424-2435
fam *= SEGMENT_MULT[segment][gender] * SEGMENT_MULT[segment][category]
mot *= SEGMENT_MULT[segment][gender] * SEGMENT_MULT[segment][category]
fam *= REGION_MULT[region]
mot *= REGION_MULT[region]
```

**Example Multipliers (from `config.yaml`):**
- `Core Classical (F35-64)` × `female` × `classic_romance` = 1.12 × 1.08 = **1.21x**
- `Family (Parents w/ kids)` × `family_classic` = 1.18x
- `Province` (Alberta-wide) = 1.0 (baseline)

### 2.5 Benchmark Normalization

All signals are indexed to a user-selected benchmark title (default: "Cinderella"):

```python
# streamlit_app.py:2431-2438
def _normalize_signals_by_benchmark(seg_to_raw, benchmark_entry, region_key):
    for seg_key, (fam_raw, mot_raw) in seg_to_raw.items():
        bench_fam_raw, bench_mot_raw = calc_scores(benchmark_entry, seg_key, region_key)
        fam_idx = (fam_raw / bench_fam_raw) * 100.0
        mot_idx = (mot_raw / bench_mot_raw) * 100.0
        combined_idx = (fam_idx + mot_idx) / 2.0
```

**Formula:**  
$$\text{IndexedSignal}_{\text{seg}} = \frac{1}{2} \left( \frac{\text{Fam}_{\text{title}}}{\text{Fam}_{\text{benchmark}}} \times 100 + \frac{\text{Mot}_{\text{title}}}{\text{Mot}_{\text{benchmark}}} \times 100 \right)$$

Benchmark title always scores exactly 100; other titles scale proportionally.

### 2.6 SignalOnly Composite

**Implementation:** `streamlit_app.py:2900-2910`

```python
SignalOnly = 0.50 * Familiarity + 0.50 * Motivation
```

Consolidated online visibility metric (range typically [20, 180])

---

## 3. Machine Learning Models

### 3.1 Dynamically-Trained Ridge Regression (Primary Model)

**Architecture:** Locally-trained regression models created at runtime based on user-supplied historical data

**Implementation:** `streamlit_app.py:2923-2980` (`_fit_overall_and_by_category` function)

#### Model Training Strategy

The system trains regression models dynamically when users provide historical production data:

```python
# streamlit_app.py:2923-2980
def _fit_overall_and_by_category(df_known_in: pd.DataFrame):
    if ML_AVAILABLE and len(df_known_in) >= 3:
        # Use constrained Ridge regression models
        overall_model, cat_models, ... = _train_ml_models(df_known_in)
        return ('ml', overall_model, cat_models, ...)
    else:
        # Fallback to constrained LinearRegression if data insufficient
        overall = None
        if len(df_known_in) >= 5:
            # Fit linear model with anchor point constraints
            a, b = np.polyfit(x_combined, y_combined, 1)
            overall = (float(a), float(b))
        return ('linear', overall, cat_coefs, ...)
```

### 3.2 Constrained Ridge Regression

**Implementation:** `streamlit_app.py:2655-2770` and `streamlit_app.py:2923-2980`

When sufficient historical data exists (≥3 samples), the system trains Ridge regression models with synthetic anchor points to ensure realistic predictions.

#### Training Logic

```python
# streamlit_app.py:2680-2710
def _train_ml_models(df_known_in: pd.DataFrame):
    X_original = df_known_in[['SignalOnly']].values
    y_original = df_known_in['TicketIndex_DeSeason'].values
    
    # Add synthetic anchor points
    n_real = len(df_known_in)
    anchor_weight = max(3, n_real // 2)
    X_anchors = np.array([[0.0], [100.0]])
    y_anchors = np.array([25.0, 100.0])
    
    # Repeat anchors to increase influence
    X_anchors_weighted = np.repeat(X_anchors, anchor_weight, axis=0)
    y_anchors_weighted = np.repeat(y_anchors, anchor_weight)
    
    # Combine with real data
    X = np.vstack([X_original, X_anchors_weighted])
    y = np.concatenate([y_original, y_anchors_weighted])
    
    model = Ridge(alpha=5.0, random_state=42)
    model.fit(X, y)
```

#### Constraint Mechanism

**Anchor Points:**
- `SignalOnly = 0` → `TicketIndex = 25` (realistic floor for minimal buzz)
- `SignalOnly = 100` → `TicketIndex = 100` (benchmark alignment)

**Mathematical Effect:**

Standard Ridge regression minimizes:
$$\min_{\beta} \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_i)^2 + \alpha \|\beta\|^2$$

With weighted anchors, this becomes:
$$\min_{\beta} \sum_{i=1}^{n_{\text{real}}} (y_i - \hat{y}_i)^2 + w_{\text{anchor}} [(25 - \hat{y}_0)^2 + (100 - \hat{y}_{100})^2] + \alpha \|\beta\|^2$$

where $w_{\text{anchor}} = \max(3, n_{\text{real}} / 2)$.

Model is "pulled" toward desired endpoints while still fitting historical data.

#### Category-Specific Models

```python
# streamlit_app.py:2746-2765
for cat, g in df_known_in.groupby("Category"):
    if len(g) >= 3:
        model = Ridge(alpha=1.0, random_state=42) if len(g) >= 5 else LinearRegression()
        model.fit(X_cat, y_cat)
        cat_models[cat] = model
```

**Hierarchy:**
1. If ≥5 samples in category → Ridge with α=1.0
2. If 3-4 samples → Linear regression (no regularization)
3. If <3 samples → Skip, fall back to overall model

### 3.3 LinearRegression Fallback Model

**Implementation:** `streamlit_app.py:2945-2980`

When scikit-learn is unavailable (`ML_AVAILABLE = False`) or for legacy compatibility, the system falls back to constrained LinearRegression using NumPy's `polyfit`.

#### Fallback Activation Conditions

```python
# streamlit_app.py:2945-2950
if ML_AVAILABLE and len(df_known_in) >= 3:
    # Use Ridge regression (primary path)
    ...
else:
    # Use LinearRegression fallback (legacy path)
    if len(df_known_in) >= 5:
        a, b = np.polyfit(x_combined, y_combined, 1)
        overall = (float(a), float(b))
```

#### Constraint Implementation

```python
# streamlit_app.py:2960-2975
# Add weighted anchor points
n_anchors = max(2, len(x_real) // 3)
x_anchors = np.array([0.0] * n_anchors + [100.0] * n_anchors)
y_anchors = np.array([25.0] * n_anchors + [100.0] * n_anchors)

x_combined = np.concatenate([x_real, x_anchors])
y_combined = np.concatenate([y_real, y_anchors])

a, b = np.polyfit(x_combined, y_combined, 1)
```

Linear fit minimizes $\sum_i (y_i - (ax_i + b))^2$ with weighted anchor points pulling the line toward:
- $(x=0, y=25)$ for minimal buzz
- $(x=100, y=100)$ for benchmark alignment

### 3.4 k-Nearest Neighbors Cold-Start Fallback

**Implementation:** `ml/knn_fallback.py:1-679`

When neither Ridge nor LinearRegression models can provide predictions, the system uses k-NN similarity matching.

#### Algorithm Configuration

```python
# ml/knn_fallback.py:95-130
class KNNFallback:
    def __init__(
        self,
        k: int = 5,
        metric: str = "cosine",
        normalize: bool = True,
        recency_weight: float = 0.5,
        recency_decay: float = 0.1,
        weights: str = "distance"
    ):
        self.k = k
        self.metric = metric
        self.normalize = normalize
        self.recency_weight = recency_weight
        self.recency_decay = recency_decay
        self.weights = weights
```

#### Distance Computation

**Feature Space:** `[wiki, trends, youtube, chartmetric]`

**Normalization (if enabled):**
```python
# ml/knn_fallback.py:215-220
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)
```

**Formula (Z-score standardization):**
$$x_{\text{scaled}} = \frac{x - \mu}{\sigma}$$

**Distance Metric (Cosine):**
```python
# ml/knn_fallback.py:218-220
nn_model = NearestNeighbors(n_neighbors=k, metric='cosine')
nn_model.fit(X_scaled)
distances, indices = nn_model.kneighbors(query_point)
```

**Cosine Distance:**
$$d_{\text{cosine}}(a, b) = 1 - \frac{a \cdot b}{\|a\| \|b\|}$$

#### Recency-Weighted Voting

```python
# ml/knn_fallback.py:320-350
similarity = 1.0 - distance
years_ago = (today - last_run_date).days / 365.25
recency_factor = exp(-recency_decay * years_ago)
weight = similarity * (recency_weight * recency_factor + (1 - recency_weight))
```

**Combined Weight Formula:**
$$w_i = \text{sim}_i \times \left[ r \cdot e^{-\lambda t_i} + (1 - r) \right]$$

where:
- $\text{sim}_i$ = similarity score (1 - cosine distance)
- $r$ = `recency_weight` (default 0.5)
- $\lambda$ = `recency_decay` (default 0.1 per year)
- $t_i$ = years since neighbor's last run

**Prediction:**
```python
# ml/knn_fallback.py:355-360
weights_normalized = weights / weights.sum()
prediction = sum(outcomes * weights_normalized)
```

$$\hat{y} = \sum_{i=1}^{k} w_i^{\text{norm}} \cdot y_i$$

#### Activation Conditions & Data Requirements

**Required Conditions:**

```python
# streamlit_app.py:2971-2972
knn_enabled = KNN_CONFIG.get("enabled", True)
if knn_enabled and KNN_FALLBACK_AVAILABLE and len(df_known) >= 3:
    # Build and use KNN index
```

| Condition | Source | Default | Impact if False |
|-----------|--------|---------|-----------------|
| `knn_enabled = true` | `config.yaml:113` | Enabled | KNN completely skipped |
| `KNN_FALLBACK_AVAILABLE = true` | Import check | True (if scikit-learn installed) | KNN unavailable; falls back to Ridge |
| `len(df_known) >= 3` | Data loader | Depends on your data | KNN not activated; uses ML model or defaults |

---

## 4. Seasonality Adjustment

**Implementation:** `streamlit_app.py:2360-2415`

### 4.1 Historical Seasonality Learning

```python
# streamlit_app.py:2360-2400
for cat in categories:
    for month in range(1, 13):
        hist_sales = historical_data[(category == cat) & (month_of_opening == month)]
        if len(hist_sales) >= N_MIN:
            median_sales = np.median(hist_sales)
            overall_median = np.median(all_sales_in_category)
            raw_factor = median_sales / overall_median
            
            # Shrinkage toward 1.0
            shrunk = 1.0 + K_SHRINK * (raw_factor - 1.0)
            
            # Clip to [MINF, MAXF]
            factor_final = np.clip(shrunk, MINF, MAXF)
```

**Parameters (from `config.yaml:85-95`):**
- `K_SHRINK`: 3.0 (shrinkage for months with limited historical data)
- `MINF`: 0.90 (floor at -10% penalty)
- `MAXF`: 1.15 (ceiling at +15% boost)
- `N_MIN`: 3 (minimum samples required per category-month)

**Shrinkage Formula:**
$$F_{\text{shrunk}} = 1 + K \cdot (F_{\text{raw}} - 1)$$

### 4.2 Application to Predictions

```python
# streamlit_app.py:3092-3095
FutureSeasonalityFactor = seasonality_factor(category, proposed_run_date)
EffectiveTicketIndex = TicketIndex_DeSeason_Used * FutureSeasonalityFactor
```

A show with `TicketIndex = 100` in neutral month becomes `TicketIndex = 120` if scheduled in December (family category).

---

## 5. City & Segment Decomposition

### 5.1 City Split Learning

**Implementation:** `streamlit_app.py:1435-1540`

```python
# streamlit_app.py:1435-1500
def learn_priors_from_history(hist_df: pd.DataFrame):
    # Per-title priors
    for title, group in hist_df.groupby("Show Title"):
        yyc_total = group["Single Tickets - Calgary"].sum()
        yeg_total = group["Single Tickets - Edmonton"].sum()
        total = yyc_total + yeg_total
        calgary_share = yyc_total / total if total > 0 else 0.6
        calgary_share = np.clip(calgary_share, *CITY_CLIP_RANGE)  # [0.40, 0.75]
        TITLE_CITY_PRIORS[title] = {
            "Calgary": calgary_share,
            "Edmonton": 1.0 - calgary_share
        }
    
    # Per-category priors (weighted aggregate)
    for category, cat_group in hist_df.groupby("Category"):
        yyc_total = cat_group["Single Tickets - Calgary"].sum()
        yeg_total = cat_group["Single Tickets - Edmonton"].sum()
        # ... same logic as above
```

**Fallback Hierarchy:**
1. **Title-specific prior** (if title seen in history)
2. **Category-level prior** (if category has historical data)
3. **Default split:** Calgary 60% / Edmonton 40%

**Constraints:** `CITY_CLIP_RANGE = (0.40, 0.75)` prevents unrealistic 95/5 or 10/90 splits.

### 5.2 Segment Propensity Model

**Implementation:** `streamlit_app.py:3120-3200`

#### Prior Weights (from `data/productions/segment_priors.csv`)

```python
# streamlit_app.py:1845-1920
SEGMENT_PRIORS[region][category][segment] = weight
```

#### Signal-Based Affinity

```python
# streamlit_app.py:3135-3145
for segment in SEGMENTS:
    fam_idx_seg, mot_idx_seg = calc_scores_for_segment(entry, segment, region)
    bench_fam_seg, bench_mot_seg = calc_scores_for_segment(benchmark, segment, region)
    
    indexed_signal_seg = (
        (fam_idx_seg / bench_fam_seg) * 100 +
        (mot_idx_seg / bench_mot_seg) * 100
    ) / 2.0
```

#### Combined Propensity

```python
# streamlit_app.py:3150-3160
combined_affinity = prior_weight * indexed_signal
shares = softmax_like(combined_affinity, temperature=1.0)
```

**Softmax-Like Normalization:**
$$P(\text{segment}) = \frac{w_{\text{prior}} \cdot s_{\text{signal}}}{\sum (w \cdot s)}$$

Segments with both high prior weight (historical attendance) AND high signal affinity (title characteristics) receive higher ticket allocation.

### 5.3 Ticket Allocation

```python
# streamlit_app.py:3175-3185
for segment, share in shares.items():
    segment_tickets[segment] = int(round(EstimatedTickets_Final * share))

# City split applied AFTER segment split
for segment in SEGMENTS:
    yyc_segment = segment_tickets[segment] * city_share_calgary
    yeg_segment = segment_tickets[segment] * city_share_edmonton
```

**Two-Stage Decomposition:**
1. **Total tickets → Segments** (via propensity model)
2. **Segment tickets → Cities** (via learned city priors)

Each show gets 8 ticket estimates (4 segments × 2 cities).

---

## 6. Pipeline Integration & Execution Flow

### 6.1 End-to-End Workflow

**Function:** `compute_scores_and_store()` (`streamlit_app.py:2830-3400`)

```
1. Input Validation
   ├─ Parse title list from text area
   ├─ Lookup titles in BASELINES dictionary
   └─ Fetch live data for unknown titles (if enabled)

2. Feature Engineering
   ├─ Extract digital signals (Wiki, Trends, YouTube, Chartmetric)
   ├─ Apply segment/region multipliers
   ├─ Normalize to benchmark (Familiarity/Motivation → IndexedSignal)
   └─ Compute SignalOnly composite

3. Historical Data Join
   ├─ Lookup prior ticket medians from history
   ├─ Compute TicketIndex_DeSeason (historical)
   └─ Train/load Ridge regression models

4. ML Prediction
   ├─ Train/load Ridge regression (if available, with anchors)
   ├─ Else use LinearRegression (legacy fallback)
   └─ Else use SignalOnly as proxy

5. Seasonality Application
   ├─ Lookup FutureSeasonalityFactor for (category, month)
   ├─ EffectiveTicketIndex = TicketIndex * SeasonalityFactor
   └─ EstimatedTickets = (EffectiveTicketIndex / 100) * BenchmarkMedian

6. City/Segment Decomposition
   ├─ Compute segment propensity (priors × signals)
   ├─ Allocate tickets to 4 segments
   ├─ Split each segment to Calgary/Edmonton
   └─ Generate Singles estimates (100% of tickets)

7. Output Assembly
   ├─ Store results in st.session_state["results"]
   ├─ Generate export DataFrame with 50+ columns
   └─ Render UI tables and charts
```

### 6.2 Key Intermediate Variables

| Variable | Formula | Range | Purpose |
|----------|---------|-------|---------|
| `WikiIdx` | $40 + \min(110, \ln(1 + v_{\text{daily}}) \times 20)$ | [40, 150] | Wikipedia pageview index |
| `YouTubeIdx` | $50 + \min(90, \ln(1 + \text{median}(views)) \times 9)$ | [50, 140] | YouTube engagement |
| `Familiarity` | $0.55 \cdot \text{Wiki} + 0.30 \cdot \text{Trends} + 0.15 \cdot \text{Chartmetric}$ | [~30, ~140] | Static awareness |
| `Motivation` | $0.45 \cdot \text{YouTube} + 0.25 \cdot \text{Trends} + 0.15 \cdot \text{Chartmetric} + 0.15 \cdot \text{Wiki}$ | [~30, ~140] | Active engagement |
| `SignalOnly` | $0.50 \cdot \text{Fam} + 0.50 \cdot \text{Mot}$ | [~20, ~150] | Composite online visibility |
| `TicketIndex_DeSeason` | Ridge($\text{SignalOnly}$) or Linear($\text{SignalOnly}$) | [20, 180] | Seasonality-neutral demand |
| `FutureSeasonalityFactor` | Learned from history with shrinkage | [0.70, 1.25] | Month × Category boost/penalty |
| `EffectiveTicketIndex` | $\text{TicketIndex} \times \text{SeasonalityFactor}$ | [14, 225] | Final demand index |
| `EstimatedTickets` | $(\text{EffectiveTicketIndex} / 100) \times \text{BenchmarkMedian}$ | [0, 15000] | Absolute ticket forecast |

---

## 7. Data Leakage Prevention

### 7.1 Training Data Safeguards

The application enforces data leakage prevention during dynamic model training:

**Allowed Historical Features:**
- `prior_total_tickets` (tickets from *prior* seasons)
- `ticket_median_prior` (median from *past* runs)
- `years_since_last_run` (temporal gap from *prior* run)

**Forbidden Features:**
- `single_tickets` (current-run actual sales)
- `total_tickets_calgary` (current-run city split)

Prevents model from "cheating" by seeing actual outcomes during training.

### 7.2 Time-Aware Cross-Validation

The Ridge and LinearRegression models are trained on historical data where training data chronologically precedes validation data, preventing "future peeking".

---

## 8. Error Handling & Robustness

### 8.1 Missing Data Imputation

```python
# streamlit_app.py:2950-2980
# Historical ticket index missing → use ML model
if pd.isna(TicketIndex_DeSeason):
    if category in cat_models:
        TicketIndex_DeSeason = cat_models[category].predict([[SignalOnly]])[0]
        source = "ML Category"
    elif overall_model is not None:
        TicketIndex_DeSeason = overall_model.predict([[SignalOnly]])[0]
        source = "ML Overall"
    elif knn_index is not None:
        TicketIndex_DeSeason, source = knn_index.predict(baseline_signals)
    else:
        TicketIndex_DeSeason = SignalOnly  # Last resort
        source = "SignalOnly Fallback"
```

**Fallback Chain:**
1. Historical data (if available)
2. Category-specific ML model (if ≥3 historical samples in category)
3. Overall ML model (if ≥5 total historical samples)
4. Raw SignalOnly score (always available)

### 8.2 Numerical Stability

**Clipping:** All indices and multipliers are clipped to prevent extreme values:

```python
# streamlit_app.py:2650
wiki_idx = float(np.clip(wiki_idx, 40.0, 150.0))
youtube_idx = float(np.clip(youtube_idx, 45.0, 140.0))
TicketIndex = float(np.clip(TicketIndex, 20.0, 180.0))
```

Prevents single outlier from breaking forecast.

**Zero-Division Guards:**

```python
# streamlit_app.py:3300-3310
total = yyc + yeg
if total <= 0:
    yyc_share, yeg_share = 0.6, 0.4  # Default split
else:
    yyc_share = yyc / total
    yeg_share = yeg / total
```

---

## 9. SHAP Explainability Layer

### 9.1 Overview

The SHAP (SHapley Additive exPlanations) integration provides per-title interpretation of ticket index predictions, decomposing each forecast into individual feature contributions.

**Module:** `ml/shap_explainer.py` (841 lines)  
**Integration:** Trains alongside Ridge regression models during ML pipeline execution  
**Public Artifact:** Explanations embedded in PDF report narratives

### 9.2 Architecture

#### SHAP Model Training

```python
# streamlit_app.py:2775-2830
# Main model: Ridge regression with SignalOnly feature
overall_model = Ridge(alpha=5.0, random_state=42)
overall_model.fit(X_signals, y)

# SHAP model: Ridge regression with 4 individual signals
if len(available_signals) > 0:
    shap_model = Ridge(alpha=5.0, random_state=42)
    shap_model.fit(X_4features, y)  # [wiki, trends, youtube, chartmetric]
    overall_explainer = SHAPExplainer(shap_model, X_4features)
```

Using individual signals (wiki, trends, youtube, chartmetric) allows SHAP to decompose each signal's contribution separately, providing granular explainability.

**Training Data:** Same historical samples as main model, optionally with anchor points:
```python
# Anchor points: signal values [0,0,0,0] → 25 tickets; [mean,mean,mean,mean] → 100 tickets
anchor_points = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [X_signals.mean(axis=0)]
])
anchor_values = np.array([25.0, 100.0])
```

#### SHAP Computation Engine

```python
# ml/shap_explainer.py:61-170
class SHAPExplainer:
    def __init__(self, model, X_train, feature_names=None, sample_size=100):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.base_value = model.predict(X_train.mean().values.reshape(1, -1))[0]
        
        # Create KernelExplainer for model-agnostic interpretability
        n_background = min(sample_size, len(X_train))
        background_data = shap.sample(X_train, n_background)
        self.explainer = shap.KernelExplainer(
            model.predict,
            background_data,
            feature_names=feature_names
        )
```

**Algorithm:** `shap.KernelExplainer` uses LIME-based approach with weighted regression to estimate Shapley values.

**Mathematical Guarantee:**
$$\hat{y} = \text{base\_value} + \sum_{i=1}^{n} \text{SHAP}_i$$

### 9.3 Per-Prediction Explanations

#### Explanation Structure

```python
# ml/shap_explainer.py:305-330
explanation = {
    'prediction': float(result['predictions'][0]),
    'base_value': float(self.base_value),
    'shap_values': result['shap_values'][0],
    'feature_names': result['feature_names'],
    'feature_values': dict(zip(names, values[0])),
    'feature_contributions': [
        {
            'name': 'youtube',
            'value': 85.3,
            'shap': +12.4,
            'direction': 'up',
            'abs_impact': 12.4
        },
        # ... sorted by abs_impact descending
    ]
}
```

### 9.4 Narrative Generation

#### Format Function

```python
# ml/shap_explainer.py:356-400
def format_shap_narrative(explanation, n_top=5, min_impact=1.0, include_base=True) -> str:
    prediction = explanation['prediction']
    base_value = explanation['base_value']
    drivers = get_top_shap_drivers(explanation, n_top=n_top, min_impact=min_impact)
    
    driver_parts = []
    for driver in drivers:
        impact_str = f"+{driver['shap']:.0f}" if driver['shap'] > 0 else f"{driver['shap']:.0f}"
        display_name = driver['name'].replace("_", " ").title()
        driver_parts.append(f"{display_name} {impact_str}")
    
    drivers_text = " ".join(driver_parts)
    
    if include_base:
        return f"{prediction:.0f} tickets (base {base_value:.0f} {drivers_text})"
    else:
        return f"{prediction:.0f} tickets ({drivers_text})"
```

**Output Examples:**
- `"119 tickets (base 100 + YouTube +12 + Wiki +9 - Trends -2)"`
- `"85 tickets (base 100 - Contemporary -12 - Opening Month -8 + Prior History +5)"`

#### Integration in Board Narratives

**Current Implementation in streamlit_app.py:**

```python
# streamlit_app.py:698-780
def _narrative_for_row(r: dict, shap_explainer=None) -> str:
    try:
        from ml.title_explanation_engine import build_title_explanation
        
        # Extract SHAP values if explainer available
        if shap_explainer and SHAP_AVAILABLE:
            signal_input = pd.Series({col: r.get(col, 0) for col in ['wiki', 'trends', 'youtube', 'chartmetric']})
            explanation = shap_explainer.explain_single(signal_input)
            shap_values = {name: val for name, val in zip(explanation['feature_names'], explanation['shap_values'])}
        
        # Build multi-paragraph narrative with SHAP drivers
        narrative = build_title_explanation(
            title_metadata=dict(r),
            prediction_outputs=None,
            shap_values=shap_values,
            style="board"
        )
        return narrative
    except Exception as e:
        # Fallback to simpler text-based narrative
        return _fallback_narrative(r)
```

### 9.5 Caching & Performance

#### Two-Tier Caching

```python
# ml/shap_explainer.py:175-210
def explain_single(self, X_single, use_cache=True, cache_key=None):
    # In-memory cache (dictionary)
    if cache_key in self._explanation_cache:
        return self._explanation_cache[cache_key]  # ~0.0001s
    
    # Disk cache (pickle)
    if self.cache_dir:
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached  # ~0.001s
    
    # Compute SHAP (KernelExplainer)
    explanation = self.explainer.shap_values(X_single.values.reshape(1, -1))  # ~0.02s
    
    # Cache for future use
    self._save_to_cache(cache_key, explanation)
    return explanation
```

**Performance Metrics:**
- Cold (no cache): 270 predictions/sec (~3.7ms each)
- Warm (disk cache): 11,164 predictions/sec (~89µs each)
- **Speedup:** 27.7x (small dataset), 25.3x (large dataset)

### 9.6 Error Handling & Fallbacks

#### Production Hardening

```python
# ml/shap_explainer.py:70-140
class SHAPExplainer:
    def __init__(self, model, X_train, feature_names=None, ...):
        # Validation #1: Model has predict method
        if not hasattr(model, 'predict'):
            raise TypeError("Model must have a 'predict' method")
        
        # Validation #2: X_train not empty
        if X_train is None or len(X_train) == 0:
            raise ValueError("X_train cannot be empty")
        
        # Validation #3: Handle NaN values
        if X_train.isnull().any().any():
            n_missing = X_train.isnull().sum().sum()
            logger.warning(f"X_train contains {n_missing} NaN values - filling with 0")
            X_train = X_train.fillna(0)
        
        # Validation #4: Handle Inf values
        if np.isinf(X_train.values).any():
            logger.warning("X_train contains infinite values - clipping")
            X_train = X_train.clip(-1e10, 1e10)
```

#### Graceful Degradation

```python
# streamlit_app.py:721-750
if shap_explainer and SHAP_AVAILABLE:
    try:
        explanation = shap_explainer.explain_single(signal_input)
        shap_values = {...}
    except Exception as e:
        logging.debug(f"SHAP computation failed: {e}")
        shap_values = None  # Continue without SHAP
else:
    shap_values = None  # SHAP not available
```

**Fallback Chain:**
1. Full SHAP explanation with 4 signals (if available)
2. Single-signal SHAP (if only SignalOnly available)
3. Title explanation engine without SHAP values (generic narrative)
4. Minimal fallback text (signal-only estimate)

---

## 10. Output Schema

### 10.1 Primary Export Columns

**Full DataFrame:** `streamlit_app.py:3400-3450` → 50+ columns

**Key Columns (grouped by purpose):**

#### Input Signals
- `WikiIdx`, `TrendsIdx`, `YouTubeIdx`, `ChartmetricIdx` (raw signals)

#### Derived Indices
- `Familiarity`, `Motivation`, `SignalOnly` (composite scores)
- `TicketIndex_DeSeason_Used` (seasonality-neutral index)
- `EffectiveTicketIndex` (seasonality-adjusted index)

#### Final Predictions
- `EstimatedTickets_Final` (absolute ticket forecast)
- `YYC_Singles`, `YEG_Singles` (city decomposition)
- `Seg_GP_Tickets`, `Seg_Core_Tickets`, `Seg_Family_Tickets`, `Seg_EA_Tickets` (segment decomposition)

#### Contextual Metadata
- `TicketIndexSource` ("History", "ML Category", "kNN Fallback", etc.)
- `FutureSeasonalityFactor` (month × category boost)
- `PredictedPrimarySegment`, `PredictedSecondarySegment`
- `kNN_used` (boolean), `kNN_neighbors` (JSON list of similar titles)

#### Historical Features
- `ticket_median_prior` (median from past runs)
- `prior_total_tickets` (sum of past ticket sales)
- `run_count_prior` (number of prior productions)
- `years_since_last_run`

---

**End of Report**
