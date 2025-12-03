# Live Analytics Integration Summary

## Overview

Successfully integrated `data/audiences/live_analytics.csv` into `features/title_features.py` to replace hardcoded category assumptions with real market size data.

## Changes Made

### 1. Modified `features/title_features.py`

#### New Function: `load_live_analytics_mapping()`

- **Purpose**: Parses the complex CSV header structure and creates a category → customer count mapping
- **Input**: Path to `live_analytics.csv` (defaults to `data/audiences/live_analytics.csv`)
- **Output**: Dictionary mapping category names to addressable market sizes (cust_cnt)
- **Handles**: 
  - Complex CSV structure (row 1 = categories, row 2 = customer counts)
  - Comma-formatted numbers
  - Missing/empty values
  - Graceful fallback if file doesn't exist

#### Updated Function: `add_title_features()`

Added two new features to the existing function:

| Feature | Description | Example Value |
|---------|-------------|---------------|
| `LA_AddressableMarket` | Raw customer count from live analytics | 14,987 |
| `LA_AddressableMarket_Norm` | Normalized value (0-1 scale) | 0.945 |

**Logic**:
- Looks up the `Category` column in each row
- Retrieves corresponding customer count from mapping
- Falls back to median market size if category not found
- Normalizes by dividing by max market size

**Preserved Functionality**:
- ✓ `is_benchmark_classic` - unchanged
- ✓ `title_word_count` - unchanged

#### Updated Function: `get_feature_names()`

Now returns 4 features instead of 2:
```python
[
    'is_benchmark_classic',
    'title_word_count', 
    'LA_AddressableMarket',
    'LA_AddressableMarket_Norm'
]
```

## Data Loaded from live_analytics.csv

The integration successfully loads **13 unique categories**:

| Category | Customer Count | Normalized |
|----------|----------------|------------|
| pop_ip | 15,861 | 1.000 |
| classic_romance, classic_comedy, romantic_comedy | 15,294 | 0.964 |
| family_classic | 14,987 | 0.945 |
| contemporary | 10,402 | 0.656 |
| dramatic | 7,781 | 0.491 |
| romantic_tragedy | 5,800 | 0.366 |
| Pop Mus Bal | 15,861 | 1.000 |
| Classic Bal | 15,294 | 0.964 |
| Family Bal | 14,987 | 0.945 |
| CM Bill | 10,402 | 0.656 |
| CSNA | 5,800 | 0.366 |
| Contemp Narr | 7,869 | 0.496 |
| Cult Narr | 7,692 | 0.485 |

**Statistics**:
- Min: 5,800
- Max: 15,861
- Median: 10,402
- Mean: 11,387

## Testing

Created `test_live_analytics_integration.py` to verify:

✓ CSV parsing works correctly  
✓ Mapping is loaded with 13 categories  
✓ Features are added to DataFrames  
✓ Category lookup works (exact matches)  
✓ Median fallback works (unknown categories)  
✓ Normalization is in [0, 1] range  
✓ Existing features preserved  

## Usage Example

```python
from features.title_features import add_title_features
import pandas as pd

# Create sample data
df = pd.DataFrame({
    'show_title': ['The Nutcracker', 'Swan Lake'],
    'Category': ['family_classic', 'classic_romance, classic_comedy, romantic_comedy']
})

# Add features
df_with_features = add_title_features(df)

# Access new features
print(df_with_features['LA_AddressableMarket'])
# Output: [14987.0, 15294.0]

print(df_with_features['LA_AddressableMarket_Norm'])
# Output: [0.945, 0.964]
```

## Integration Points

The new features are automatically available in:

1. **Model Training Pipeline**: `scripts/train_safe_model.py`
2. **Feature Engineering**: `data/features.py`
3. **Dataset Builder**: `scripts/build_modelling_dataset.py`
4. **Streamlit UI**: `streamlit_app.py`

Any code that calls `add_title_features()` will now receive the two additional columns.

## Benefits

1. **Real Data**: Replaces hardcoded assumptions with actual market research
2. **Category-Specific**: Each category gets its true addressable market size
3. **Normalized Signal**: Model-ready 0-1 scale feature
4. **Graceful Fallback**: Uses median if category unknown or file missing
5. **Backwards Compatible**: Existing features unchanged

## File Structure

```
data/audiences/live_analytics.csv      # Source data (complex CSV)
features/title_features.py             # Feature engineering (updated)
test_live_analytics_integration.py     # Verification script
LIVE_ANALYTICS_INTEGRATION.md          # This document
```

## Next Steps

To use the new features in model training:

1. Ensure `Category` column exists in your dataset
2. Call `add_title_features(df)` during feature engineering
3. Include `LA_AddressableMarket_Norm` in your feature list
4. Retrain models with the new signal

The feature is **production-ready** and tested. All tests passed successfully.
