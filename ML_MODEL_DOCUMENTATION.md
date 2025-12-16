# Machine Learning Model Integration Documentation

## Overview

This document describes the **constrained Ridge regression** implementation used in the Alberta Ballet title scoring application to provide accurate and realistic ticket demand projections.

## Problem Statement

The ticket demand forecasting model must produce realistic estimates across the full spectrum of title popularity. Key challenges include:

- **Low-Buzz Titles**: Must avoid inflated estimates for titles with minimal online presence
- **High-Buzz Titles**: Must properly differentiate strong performers from weak ones
- **Interpretability**: Stakeholders need transparent formulas they can understand and audit

## Solution

We implemented a **constrained Ridge regression** approach that enforces realistic behavior through synthetic anchor points:

### Model Selection Strategy

The system now uses Ridge regression with explicit constraints for all dataset sizes:

| Dataset Size | Overall Model | Category Models | Rationale |
|--------------|---------------|-----------------|-----------|
| ≥5 samples | **Constrained Ridge (α=5.0)** | **Constrained Ridge (α=5.0)** | Regularized with anchor points |
| 3-4 samples | **Constrained Linear** | **Constrained Linear** | Linear fit with anchor constraints |
| <3 samples | Fallback to simple linear regression | N/A | Not enough data for ML |

### Anchor Point Constraints

The model is trained with weighted synthetic data points that enforce:

1. **Low Floor**: `SignalOnly = 0` → `TicketIndex = 25`
   - Ensures titles with minimal online buzz get realistic baseline estimates
   - Prevents the high intercept problem
   
2. **Benchmark Alignment**: `SignalOnly = 100` → `TicketIndex = 100`
   - Maintains consistency with the benchmark title (e.g., Cinderella)
   - Ensures the index scale remains interpretable

**Typical Model Formula:**
```
TicketIndex ≈ 0.75 × SignalOnly + 27.3
```

This produces realistic estimates:
- Low-buzz titles (SignalOnly ≈ 5): ~3,800 tickets (down from ~5,500)
- Medium-buzz titles (SignalOnly ≈ 50): ~8,000 tickets  
- High-buzz titles (SignalOnly ≈ 80): ~10,600 tickets

### Key Features

1. **Constrained Training**: Synthetic anchor points guide model behavior
2. **Ridge Regularization**: α=5.0 prevents overfitting to historical outliers
3. **Performance Metrics**: Displays R², MAE, RMSE, and anchor verification in UI
4. **Backward Compatibility**: Falls back to constrained linear regression if ML unavailable
5. **Prediction Clipping**: All predictions clipped to [20, 180] range for validity

## Implementation Details

### Core Functions

#### `_train_ml_models(df_known_in: pd.DataFrame)`
Trains both overall and per-category constrained Ridge regression models.

**Process:**
1. Prepares training data from historical TicketIndex and SignalOnly values
2. Creates synthetic anchor points: `[(0, 25), (100, 100)]`
3. Weights anchor points based on dataset size: `anchor_weight = max(3, n_real // 2)`
4. Combines real data with weighted anchors
5. Fits Ridge regression model with α=5.0 regularization
6. Evaluates on real data only (not anchors) for metrics

**Returns:**
- `overall_model`: Ridge regression model with constraints
- `cat_models`: Dictionary of category-specific Ridge/Linear models
- `overall_metrics`: Dictionary with R², MAE, RMSE, n_samples, intercept, slope, anchor_0, anchor_100
- `cat_metrics`: Dictionary of metrics for each category

#### `_predict_with_ml_model(model, signal_only: float)`
Makes predictions with trained Ridge regression models.

**Returns:**
- Float prediction or np.nan if model is None or prediction fails

#### `_fit_overall_and_by_category(df_known_in: pd.DataFrame)`
Main entry point that applies constrained regression for all cases.

**Returns:**
- Tuple: (model_type, model, metrics, ...)
- model_type: 'ml' or 'linear'

### Data Flow

1. Historical ticket data is loaded and de-seasonalized
2. SignalOnly scores are computed from familiarity/motivation indices
3. `_fit_overall_and_by_category()` is called with known data
4. If ML_AVAILABLE and sufficient data (≥5 samples):
   - Create anchor points: `SignalOnly=0 → TicketIndex=25`, `SignalOnly=100 → TicketIndex=100`
   - Weight anchors proportional to dataset size
   - Train constrained Ridge regression overall model (α=5.0)
   - Train constrained Ridge/Linear category models
   - Compute performance metrics on real data
5. If <5 samples:
   - Apply constrained linear regression with same anchors
6. For unknown titles, predict using category model → overall model → fallback
7. All predictions are clipped to [20, 180] range

## Performance Results

### Anchor Point Verification

The model achieves excellent anchor alignment:
- `SignalOnly = 0`: Predicts TicketIndex ≈ 27.3 (target: 25, error: 2.3)
- `SignalOnly = 100`: Predicts TicketIndex ≈ 102.2 (target: 100, error: 2.2)

### Real-World Impact

Example predictions demonstrating the model's performance:

| Title | SignalOnly | Estimated Tickets | Notes |
|-------|------------|-------------------|--------|
| After the Rain | 5.41 | **3,755 tickets** | Realistic for low-buzz contemporary |
| Afternoon of a Faun | 6.63 | **3,865 tickets** | Appropriate for limited recognition |
| Dracula (high buzz) | 81.82 | **10,619 tickets** | Strong differentiation for popular title |

### Key Model Strengths

The constrained Ridge model provides:
- **Realistic estimates** for low-buzz titles (~3,800 tickets typical)
- **Strong differentiation** between high and low demand titles
- **Interpretable linear relationship** (slope ≈ 0.75)
- **Stable predictions** through regularization (α=5.0)
- **Benchmark alignment** maintained through anchor at SignalOnly=100

## Dependencies

```
scikit-learn>=1.5.0  # For Ridge regression and metrics
```

## Usage in the App

### UI Display

When the app trains models, it displays metrics like:

```
🤖 Ridge Model Performance — R²: 0.903 | MAE: 3.9 | RMSE: 4.7 | Samples: 66
📊 Category Models (Ridge/Linear) — classic_romance: R²=0.88 (n=12), contemporary: R²=0.77 (n=7)
```

### Prediction Sources

In the results table, the `TicketIndexSource` column shows:
- "ML Overall" - Predicted by constrained Ridge regression
- "ML Category" - Predicted by category-specific Ridge/Linear model
- "History" - From actual historical data
- "Category model" - Simple linear fallback
- "Overall model" - Simple linear fallback
- "Not enough data" - Insufficient data for prediction

## Configuration

Currently uses hardcoded hyperparameters optimized for small-medium datasets. Future enhancements could:

1. Add hyperparameter tuning with GridSearchCV
2. Implement model persistence for faster re-runs
3. Add feature importance display
4. Support additional features beyond SignalOnly

## Maintenance Notes

### Adding New Models

To add a new model type:

1. Import the model in the ML imports section
2. Update `_train_ml_models()` to include the new model
3. Add appropriate dataset size threshold
4. Update documentation

### Monitoring Model Performance

The UI automatically displays:
- R² score (model fit quality, higher is better)
- MAE (prediction error in index points, lower is better)
- CV-MAE (cross-validated error for robustness)
- RMSE (root mean squared error, lower is better)
- Sample counts (transparency about data size)

Monitor these metrics when:
- Adding new historical data
- Changing model hyperparameters
- Investigating prediction quality issues

## Security Considerations

All dependencies checked with GitHub Advisory Database:
- ✅ No known vulnerabilities in scikit-learn 1.7.2
- ✅ CodeQL analysis passed with 0 alerts

## Robust Forecasting Pipeline (New)

### Non-Leaky Training Process

The new training pipeline (`scripts/train_safe_model.py`) ensures no data leakage:

1. **Dataset Building** (`scripts/build_modelling_dataset.py`):
   - Uses only forecast-time-available features
   - Computes lagged historical features from PRIOR seasons only
   - Explicit assertions to block current-run ticket columns

2. **Forbidden Features** (will cause assertion error if detected):
   - Single Tickets - Calgary/Edmonton
   - Total Tickets columns
   - Any column matching current-run sales patterns

3. **Allowed Features**:
   - Baseline signals: wiki, trends, youtube, chartmetric
   - Categorical: category, gender
   - Prior-season lags: prior_total_tickets, ticket_median_prior
   - Remount features: years_since_last_run, is_remount_recent
   - Seasonality: month_of_opening, holiday_flag

### Remount & Post-COVID Features

The model learns:
- **years_since_last_run**: Numeric, derived from past_runs.csv
- **is_remount_recent**: Binary, ≤2 years since last run
- **is_remount_medium**: Binary, 2-4 years since last run
- **run_count_prior**: Number of previous runs

Post-COVID adjustment is applied as a configurable factor in config.yaml.

### k-NN Cold-Start Fallback

For titles without historical ticket data, the k-NN fallback (`ml/knn_fallback.py`):
- Uses baseline signal vectors (wiki, trends, youtube, chartmetric)
- Finds k nearest neighbors among titles with known outcomes
- Weights predictions by similarity AND recency of neighbor runs
- Configurable via config.yaml knn settings

### Time-Aware Backtesting

The backtest script (`scripts/backtest_timeaware.py`) evaluates:
1. **Heuristic/Composite**: Rule-based scoring from streamlit_app.py
2. **k-NN Fallback**: Similarity-based predictions
3. **Baseline-Only Model**: Trained on signals only
4. **Full Model**: Trained on all features

Outputs include MAE/RMSE/R² per method and visualizations.

### Time-Aware Train/Test Splitting (CRITICAL)

**Problem**: Random train/test splits (e.g., sklearn's `train_test_split`) can allow future data to leak into training. For example, 2018 and 2020 data could be in the training set while predicting 2019 tickets.

**Solution**: The `ml/time_splits.py` module provides utilities that guarantee strictly chronological splitting:

#### Core Functions

1. **`assert_chronological_split(train_df, test_df, date_column)`**
   - Raises `AssertionError` if train max date >= test min date
   - Also checks for index overlap between train and test

2. **`chronological_train_test_split(df, date_column, test_ratio=0.2)`**
   - Splits data so all training rows come before all test rows in time
   - Returns `(train_df, test_df)` tuple with chronological guarantee

3. **`TimeSeriesCVSplitter(n_splits, date_column)`**
   - sklearn-compatible cross-validator
   - Uses expanding window: fold 1 trains on earliest data, fold N trains on most data
   - Each fold guarantees train dates < test dates

4. **`rolling_origin_cv_splits(df, date_column, initial_train_period, horizon, step)`**
   - Walk-forward validation generator
   - Training window expands (or slides) forward over time
   - Test window always comes strictly after training window

5. **`assert_group_chronological_split(df, train_mask, date_column, group_column)`**
   - For grouped time series (multiple shows/titles)
   - Ensures within each group, train dates precede test dates

#### Example Usage

```python
from ml.time_splits import chronological_train_test_split, TimeSeriesCVSplitter

# Simple chronological split
train, test = chronological_train_test_split(df, "end_date", test_ratio=0.2)
# Guarantees: train.end_date.max() < test.end_date.min()

# Time-aware cross-validation
cv = TimeSeriesCVSplitter(n_splits=5, date_column="end_date")
for train_idx, test_idx in cv.split(X):
    # train_idx dates always before test_idx dates
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    score = model.score(X.iloc[test_idx], y.iloc[test_idx])
```

#### Implementation in Training Scripts

- **`ml/training.py`**: Uses `chronological_train_test_split()` when date column available
- **`scripts/train_safe_model.py`**: Uses `TimeSeriesCVSplitter` for CV
- **`scripts/backtest_timeaware.py`**: Uses `TimeSeriesCVSplitter` for backtesting

### Calibration

Linear calibration (`scripts/calibrate_predictions.py`) supports:
- **Global**: Single alpha/beta for all predictions
- **Per-Category**: Separate parameters per show category
- **By-Remount-Bin**: Parameters for recent/medium/old remounts

### Checklist: Avoiding Leakage

When modifying training code:
- [ ] Check that no columns matching "ticket" patterns are features (except allowed priors)
- [ ] Verify lagged features use only prior-season data
- [ ] **Ensure train/test splits are strictly chronological** (use `ml/time_splits.py`)
- [ ] Run `pytest tests/test_no_leakage_in_dataset.py tests/test_time_splits.py` before committing
- [ ] Use `scripts/build_modelling_dataset.py` output, not raw history

## Bank of Canada Valet API Integration

### Overview

The application now supports live economic data from the Bank of Canada Valet API as an **optional, supplemental** layer for the economic sentiment adjustment. This integration provides current market conditions from official BoC data sources.

**IMPORTANT**: This is NOT a replacement for historical economic data. The existing historical datasets (WCS oil prices, Alberta unemployment) remain fully intact and continue to be used for:
- Model training
- Backtesting
- Historical analysis

The BoC integration is purely for fetching **live/current** values to supplement the economic sentiment adjustment at forecast time.

### Configuration

The BoC integration is controlled by `config/economic_boc.yaml`:

```yaml
# Master toggle
use_boc_live_data: true

# Fallback behavior when BoC unavailable
fallback_mode: "historical"  # Options: historical, neutral, last_cached

# Configured BoC series
boc_series:
  policy_rate:
    id: "B114039"
    weight: 0.15
    direction: "negative"
  bcpi_energy:
    id: "A.ENER"
    weight: 0.25
    direction: "positive"
  # ... additional series
```

### BoC Series Used

| Key | Series ID | Description | Weight |
|-----|-----------|-------------|--------|
| policy_rate | B114039 | Bank of Canada Target Rate | 15% |
| corra | AVG.INTWO | Canadian Overnight Repo Rate Average | 5% |
| gov_bond_2y | BD.CDN.2YR.DQ.YLD | 2-year GoC bond yield | 8% |
| gov_bond_5y | BD.CDN.5YR.DQ.YLD | 5-year GoC bond yield | 8% |
| gov_bond_10y | BD.CDN.10YR.DQ.YLD | 10-year GoC bond yield | 9% |
| bcpi_total | A.BCPI | Commodity Price Index Total | 10% |
| bcpi_energy | A.ENER | Energy Commodity Index (critical for Alberta) | 25% |
| bcpi_ex_energy | A.BCNE | Non-energy commodities | 10% |
| cpi_core | ATOM_V41693242 | Core CPI excluding volatile items | 10% |

### Sentiment Calculation

The BoC sentiment factor is calculated as:

1. **Fetch current values** for each configured series from the Valet API
2. **Standardize** each value using z-scores relative to historical means
3. **Apply direction**: For "negative" indicators (interest rates), flip the sign
4. **Compute weighted average** of z-scores
5. **Convert to factor**: `factor = 1.0 + (average_z × sensitivity)`
6. **Clamp to bounds**: [0.85, 1.15]

### Combined Sentiment

When BoC live data is enabled, the UI offers a combined sentiment that blends:
- **Historical data** (70% weight): WCS oil prices, Alberta unemployment
- **BoC live data** (30% weight): Current market indicators

This provides stability from historical patterns while incorporating current market conditions.

### API Caching

To minimize API calls:
- Values are cached for the current day (24-hour TTL)
- Cache refreshes automatically after midnight UTC
- Cache can be manually cleared via `clear_cache()`

### Error Handling

When BoC API is unavailable:
- Falls back to historical economic sentiment (default)
- Alternatively can use neutral (1.0) or last cached values
- Warnings are displayed in the UI when fallback is used

### Files Added

- `utils/boc_client.py`: BoC Valet API client with caching
- `utils/economic_factors.py`: Sentiment calculation and integration
- `config/economic_boc.yaml`: BoC series configuration
- `tests/test_boc_client.py`: Unit tests for API client
- `tests/test_economic_factors.py`: Unit tests for sentiment calculation

## Alberta Economic Dashboard API Integration

### Overview

In addition to the Bank of Canada data, the application now supports live economic data from the **Alberta Economic Dashboard** (https://economicdashboard.alberta.ca/). This provides Alberta-specific economic indicators that are more directly relevant for predicting local arts attendance.

**IMPORTANT**: Like the BoC integration, this is NOT a replacement for historical economic data. The existing historical datasets remain fully intact and continue to be used for model training and backtesting.

### Alberta Indicators

The integration fetches 12 key Alberta economic indicators:

| Key | Description | Category | Weight |
|-----|-------------|----------|--------|
| ab_unemployment_rate | Unemployment rate in Alberta | Labour | 12% |
| ab_employment_rate | Employment rate in Alberta | Labour | 8% |
| ab_employment_level | Employment in Alberta (level) | Labour | 5% |
| ab_participation_rate | Participation rate in Alberta | Labour | 5% |
| ab_avg_weekly_earnings | Average Weekly Earnings | Labour | 10% |
| ab_cpi | Consumer Price Index for Alberta | Prices | 8% |
| ab_wcs_oil_price | WCS (Western Canadian Select) Oil Price | Energy | 15% |
| ab_retail_trade | Retail Trade in Alberta | Consumer | 10% |
| ab_restaurant_sales | Restaurant Sales in Alberta | Consumer | 7% |
| ab_air_passengers | Air Passengers (YEG + YYC total) | Consumer | 5% |
| ab_net_migration | Net Migration into Alberta | Population | 8% |
| ab_population_quarterly | Population (Quarterly) in Alberta | Population | 7% |

### Configuration

The Alberta integration is controlled by `config/economic_alberta.yaml`:

```yaml
# Master toggle
use_alberta_live_data: true

# Fallback behavior when Alberta data unavailable
fallback_mode: "neutral"  # Options: neutral, skip

# Indicator configuration
alberta_indicators:
  ab_unemployment_rate:
    api_code: "c1fe936a-324a-4a37-bfde-eeb3bb3d7c8c"
    baseline: 7.0
    weight: 0.12
    direction: "negative"
  # ... additional indicators
```

### Usage

```python
from utils.economic_factors import (
    get_alberta_economic_indicators,
    compute_alberta_economic_sentiment,
    get_current_economic_context,
)

# Fetch all Alberta indicators
indicators = get_alberta_economic_indicators()
unemployment = indicators.get("ab_unemployment_rate")
wcs_price = indicators.get("ab_wcs_oil_price")

# Compute Alberta-specific sentiment factor
factor, details = compute_alberta_economic_sentiment()
# factor is in range [0.85, 1.15] where 1.0 = neutral

# Get combined context (BOC + Alberta)
context = get_current_economic_context()
# Returns {boc: {...}, alberta: {...}, combined_sentiment: 1.05, ...}
```

### Combined Sentiment Calculation

When both BOC and Alberta data are available, the combined sentiment is calculated as:
- **BOC weight**: 40% (national macroeconomic conditions)
- **Alberta weight**: 60% (regional economic conditions - more directly relevant)

### Files Added/Modified

- `utils/alberta_client.py`: Alberta Economic Dashboard API client with caching
- `config/economic_alberta.yaml`: Alberta indicator configuration
- `utils/economic_factors.py`: Updated to integrate Alberta indicators
- `tests/test_alberta_client.py`: Unit tests for Alberta API client (33 tests)
- `tests/test_economic_factors.py`: Extended with Alberta integration tests

### API Caching

Like the BOC integration:
- Values are cached for the current day (24-hour TTL)
- Cache refreshes automatically after midnight UTC
- Separate cache instances for BOC and Alberta data

### Error Handling

When Alberta API is unavailable:
- Falls back to neutral sentiment factor (1.0)
- Does not crash the scoring flow
- Warnings are logged for debugging

## Model Interpretability & Analysis

The script `scripts/analyze_safe_model.py` provides tools to understand what drives the safe model's predictions.

### Running the Analysis

```bash
python scripts/analyze_safe_model.py
```

This command loads the trained model and produces:

### Output Artifacts

| File | Description |
|------|-------------|
| `results/feature_importances.csv` | Basic feature importance from training |
| `results/feature_importances_detailed.csv` | Tree and permutation importances with importance type |
| `results/model_recipe_linear.csv` | Linear surrogate model coefficients (Ridge regression) |
| `results/model_recipe_summary.md` | Human-readable summary of model behavior |
| `results/plots/feature_importances_bar.png` | Bar chart of top features by permutation importance |
| `results/plots/surrogate_vs_model_scatter.png` | Scatter plot: surrogate predictions vs Ridge model |

### Key Insights

The linear surrogate model approximates the Ridge regression behavior with a simple equation:
- R² typically > 0.99 (excellent approximation)
- Top positive drivers: `prior_total_tickets`, `ticket_median_prior`
- Top negative drivers: `prior_run_count`, `is_remount_recent`

### Regenerating Artifacts

To refresh the analysis after retraining:

```bash
# 1. Build dataset
python scripts/build_modelling_dataset.py --output data/modelling_dataset.csv

# 2. Train model
python scripts/train_safe_model.py --dataset data/modelling_dataset.csv

# 3. Analyze model
python scripts/analyze_safe_model.py
```

## Future Enhancements

Potential improvements:
1. ~~**Model Persistence**: Save trained models to disk for faster app startup~~ ✓ Implemented
2. ~~**Hyperparameter Tuning**: Add GridSearchCV for optimal hyperparameters~~ ✓ Via --tune flag
3. ~~**Feature Engineering**: Incorporate additional features (genre, gender, category)~~ ✓ Implemented
4. **Ensemble Methods**: Combine predictions from multiple models
5. **Uncertainty Quantification**: Add prediction intervals/confidence bounds
6. **Online Learning**: Update models incrementally as new data arrives
7. ~~**SHAP Explanations**: Feature importance visualization~~ ✓ Via --save-shap flag
8. ~~**Live Economic Data**: Bank of Canada Valet API integration~~ ✓ Implemented
9. ~~**Alberta Economic Data**: Alberta Economic Dashboard integration~~ ✓ Implemented
10. ~~**Model Analysis Script**: Interpretability artifacts and surrogate model~~ ✓ Implemented

## References

- scikit-learn documentation: https://scikit-learn.org/
- Ridge regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
- SHAP documentation: https://shap.readthedocs.io/
- Bank of Canada Valet API: https://www.bankofcanada.ca/valet/docs
- Alberta Economic Dashboard: https://economicdashboard.alberta.ca/
- Alberta Economic Data API: https://api.economicdata.alberta.ca/
- Test results: See ML_MODEL_TESTING.log (if available)
