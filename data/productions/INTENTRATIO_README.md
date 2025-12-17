# IntentRatio Column Documentation

## What is IntentRatio?

IntentRatio represents the proportion of search traffic that is performance-specific for a given ballet title. It's a metric that helps distinguish between:
- **High clarity** searches (e.g., someone specifically looking for ballet performances)
- **Ambiguous** searches (e.g., someone looking for a movie, book, or other media with the same name)

## Value Range

IntentRatio values range from 0.0 to 1.0 (displayed as 0% to 100%):
- **< 5%**: Very high ambiguity - search traffic is mostly for non-performance content
- **5-30%**: Mixed signal - searches may include related media (movies, books, etc.)
- **> 30%**: High clarity - search traffic is predominantly performance-specific

## Calculation Logic

IntentRatio values in baselines.csv are calculated based on:

### 1. Category-Based Baseline
- **Family classics & pop_ip**: 0.10-0.15 (10-15%)
  - Examples: Aladdin, Wizard of Oz, Addams Family
  - Reason: High ambiguity due to popular movies, books, and other media
  
- **Contemporary works**: 0.40-0.50 (40-50%)
  - Examples: Adagio Hammerklavier, After the Rain, Agon
  - Reason: More specific to ballet/dance performances
  
- **Classical ballets**: 0.25-0.35 (25-35%)
  - Examples: Giselle, Swan Lake, Sleeping Beauty
  - Reason: Medium clarity - some confusion with films/adaptations

- **Adult literary drama**: 0.25 (25%)
  - Examples: A Streetcar Named Desire, Anna Karenina
  - Reason: Mix of theatrical performances and literary references

### 2. Signal Strength Adjustment
- **Popular titles** (trends > 50 or wiki > 50): -5% adjustment
  - More search volume often means more noise from non-performance sources
  
- **Obscure titles** (trends < 5 and wiki < 5): +10% adjustment
  - When these titles are searched, it's more likely for performance-specific intent

## UI Display

In the Streamlit app, IntentRatio is displayed:
1. **As a column** in the ticket estimation table, formatted as percentage (e.g., "35.0%")
2. **As a metric** above the table showing the median/mean IntentRatio across all scored titles
3. **With contextual feedback**:
   - ⚠️ "High ambiguity" warning for values < 5%
   - ✅ "High clarity" confirmation for values > 30%
   - ℹ️ "Mixed signal" notice for values 5-30%

## Examples

| Title | Category | IntentRatio | Interpretation |
|-------|----------|-------------|----------------|
| Adagio Hammerklavier | contemporary | 50% | High clarity - searches are performance-specific |
| Cinderella | family_classic | 10% | High ambiguity - searches include Disney movie |
| A Month in the Country | romantic_tragedy | 40% | Good clarity - mostly ballet-related searches |
| Aladdin | family_classic | 10% | High ambiguity - Disney movie dominates searches |

## Updating IntentRatio Values

To update IntentRatio values in baselines.csv:
1. Open `data/productions/baselines.csv`
2. Edit the `IntentRatio` column for specific titles
3. Ensure values are between 0.0 and 1.0
4. Round to 2 decimal places to avoid floating-point precision issues
5. Reload the Streamlit app to see changes

## Technical Notes

- IntentRatio is **optional** - if missing, cells will show NaN or blank
- Values are loaded in `load_baselines()` function in streamlit_app.py
- The column is case-insensitive when loading (can be "IntentRatio" or "intentratio")
- Missing or invalid values default to `np.nan` and are handled gracefully in the UI
