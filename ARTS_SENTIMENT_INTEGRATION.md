# Arts Sentiment Integration Summary

## Overview

Successfully integrated `data/audiences/nanos_arts_donors.csv` into `features/economic_features.py` to add arts-specific sentiment as an economic indicator.

## Changes Made

### 1. Modified `features/economic_features.py`

#### Added Constants

```python
AUDIENCES_DIR = DATA_DIR / "audiences"
```

#### New Function: `load_arts_sentiment()`

- **Purpose**: Loads and filters Nanos arts donors survey data
- **Filter Criteria**:
  - `subcategory == 'Arts share'`
  - `metric == 'Avg %'`
- **Output**: DataFrame with columns `year` (int64) and `arts_sentiment` (float)
- **Handles**: 
  - CSV parsing with multiple sections
  - Numeric conversion for year and value columns
  - Missing/invalid data gracefully

#### New Function: `add_arts_sentiment_feature()`

- **Purpose**: Merges arts sentiment onto show data based on year
- **Logic**:
  - Extracts year from `start_date` column
  - Uses `pd.merge_asof` with `direction='backward'` for forward-fill behavior
  - Falls back to median sentiment for missing years
  - Preserves original DataFrame row order
- **Output**: Adds `Econ_ArtsSentiment` column

#### Updated Function: `add_economic_features()`

Now adds three economic features:
1. `Econ_BocFactor` - Bank of Canada indicators (CPI + energy)
2. `Econ_AlbertaFactor` - Alberta-specific (oil + unemployment)
3. `Econ_ArtsSentiment` - Arts giving sentiment (NEW)

#### Updated Function: `get_feature_names()`

Returns three features instead of two:
```python
['Econ_BocFactor', 'Econ_AlbertaFactor', 'Econ_ArtsSentiment']
```

## Data Loaded from nanos_arts_donors.csv

The integration loads **3 years** of arts sentiment data:

| Year | Arts Sentiment | Source |
|------|----------------|--------|
| 2023 | 11.0% | Nanos survey - % of $100 charitable donation going to arts |
| 2024 | 12.0% | Nanos survey - % of $100 charitable donation going to arts |
| 2025 | 12.0% | Nanos survey - % of $100 charitable donation going to arts |

**Statistics**:
- Median: 12.0%
- Range: 11.0% - 12.0%
- Used as fallback for shows outside data range

## Merge Logic

### Forward-Fill Behavior

The function uses `pd.merge_asof` with `direction='backward'` to implement forward-fill:

| Show Year | Matched Data Year | Arts Sentiment | Logic |
|-----------|-------------------|----------------|-------|
| 2020 | N/A | 12.0% (median) | Before data range, use median fallback |
| 2023 | 2023 | 11.0% | Exact match |
| 2024 | 2024 | 12.0% | Exact match |
| 2025 | 2025 | 12.0% | Exact match |
| 2026 | 2025 | 12.0% | After data range, use most recent (2025) |

### Example

```python
from features.economic_features import add_economic_features
import pandas as pd

df = pd.DataFrame({
    'show_title': ['Nutcracker 2023', 'Swan Lake 2024', 'New Show 2026'],
    'start_date': ['2023-12-15', '2024-06-20', '2026-03-10']
})

result = add_economic_features(df)

# Result:
#   show_title         Econ_ArtsSentiment
#   Nutcracker 2023    11.0
#   Swan Lake 2024     12.0
#   New Show 2026      12.0  (forward-filled from 2025)
```

## Testing

Created `test_arts_sentiment_integration.py` with three test suites:

### Test 1: Loading Arts Sentiment Data
✓ Loads 3 years of data  
✓ Filters correctly (subcategory='Arts share', metric='Avg %')  
✓ Numeric conversions work  
✓ Values in reasonable range (0-100%)  

### Test 2: Adding Arts Sentiment Feature
✓ Feature column created  
✓ Year-based matching works  
✓ 2023 → 11.0%, 2024 → 12.0%, 2025 → 12.0%  
✓ Forward-fill for future years (2026 → 12.0%)  
✓ Median fallback for past years (2020 → 12.0%)  
✓ Original row order preserved  

### Test 3: Full Economic Features Integration
✓ All three economic features present  
✓ `get_feature_names()` includes new feature  
✓ Integration with existing economic features works  

## Benefits

1. **Arts-Specific Indicator**: Complements generic economic factors with arts sector sentiment
2. **Real Survey Data**: Based on actual Nanos survey of Canadian donors
3. **Time-Aware**: Captures year-over-year changes in arts giving trends
4. **Forward-Fill Logic**: Handles missing years gracefully with most recent data
5. **Robust Fallback**: Uses median when no data available

## Technical Details

### Data Types
- Year: `int64` (ensures merge compatibility)
- Arts Sentiment: `float` (percentages as 11.0, 12.0, etc.)

### Performance
- Single merge operation per DataFrame
- No row-by-row iteration
- Preserves original DataFrame structure and order

### Error Handling
- Warns if CSV file not found
- Returns NaN if no data available
- Validates subcategory and metric filters
- Handles invalid dates/years gracefully

## Usage in ML Pipeline

The new feature is automatically available in:

1. **Feature Engineering**: `data/features.py`
2. **Model Training**: `scripts/train_safe_model.py`
3. **Dataset Building**: `scripts/build_modelling_dataset.py`
4. **Predictions**: Any code calling `add_economic_features()`

### Adding to Model

To use in model training:

```python
from features.economic_features import add_economic_features, get_feature_names

# Add economic features
df = add_economic_features(df, date_column='start_date')

# Get feature list for model
economic_features = get_feature_names()
# ['Econ_BocFactor', 'Econ_AlbertaFactor', 'Econ_ArtsSentiment']

# Use in training
X = df[economic_features + other_features]
```

## File Structure

```
data/audiences/nanos_arts_donors.csv   # Source data (Nanos survey)
features/economic_features.py          # Feature engineering (updated)
test_arts_sentiment_integration.py     # Verification script
ARTS_SENTIMENT_INTEGRATION.md          # This document
```

## Next Steps

To retrain models with the new feature:

1. Run full pipeline: `python scripts/run_full_pipeline.py`
2. The new `Econ_ArtsSentiment` feature will be automatically included
3. Check feature importance in `results/feature_importances.csv`

The feature is **production-ready** and all tests pass successfully.

## Comparison: Arts Sentiment vs Live Analytics

| Feature | Source | Type | Granularity | Purpose |
|---------|--------|------|-------------|---------|
| `LA_AddressableMarket` | live_analytics.csv | Category-level | Per category (pop_ip, family_classic, etc.) | Market size by category |
| `Econ_ArtsSentiment` | nanos_arts_donors.csv | Time-series | Per year (2023, 2024, 2025) | Arts giving sentiment over time |

These features complement each other:
- **Live Analytics**: "How big is the market for this type of show?"
- **Arts Sentiment**: "What's the overall climate for arts giving right now?"
