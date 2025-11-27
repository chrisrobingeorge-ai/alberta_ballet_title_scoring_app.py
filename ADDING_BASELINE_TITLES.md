# Adding Baseline Titles to Improve Model Accuracy

This guide explains how to add more baseline titles to improve the model's prediction accuracy, even when those titles don't have historical ticket sales data.

## Overview

The Alberta Ballet Title Scoring App stores all baseline data in a single file (`data/productions/baselines.csv`) with a `source` column that distinguishes between two types of titles:

### 1. Historical Baselines (`source = "historical"`)
- **Contains**: Titles that Alberta Ballet has performed with known ticket outcomes
- **Used for**: Training the ML model and making ticket predictions
- **Columns**: `title`, `wiki`, `trends`, `youtube`, `spotify`, `category`, `gender`, `source`, `notes`

### 2. Reference Baselines (`source = "external_reference"`)
- **Contains**: Well-known ballet/performance titles WITHOUT Alberta Ballet history
- **Used for**: 
  - k-NN similarity matching for cold-start predictions
  - Category signal calibration
  - Broader context for programming decisions
- **Columns**: Same as historical baselines

## Why Add Reference Baselines?

Adding reference baselines helps the model in several ways:

1. **Better Cold-Start Predictions**: When predicting for a new title, k-NN can find more similar titles to base predictions on
2. **Category Calibration**: Understand what "typical" signals look like for each category
3. **Broader Signal Context**: More data points for understanding signal-to-demand relationships
4. **Programming Insights**: Find which well-known titles are most similar to proposals

## How to Add Baseline Titles

### Step 1: Choose Titles to Add

Good candidates for reference baselines are:
- Well-known ballets from major companies (Royal Ballet, ABT, NYC Ballet, etc.)
- Titles with strong name recognition but not performed by AB
- Titles representing underrepresented categories
- Famous contemporary works that might be future programming options

### Step 2: Gather Signal Data

For each title, collect the four external signals:

| Signal | Source | Scale | How to Collect |
|--------|--------|-------|----------------|
| `wiki` | Wikipedia | 0-100 | Use the Title Scoring Helper app, or check monthly pageviews |
| `trends` | Google Trends | 0-100 | Search in Google Trends for Alberta region (past 12 months) |
| `youtube` | YouTube | 0-100 | Search for official performance videos, note view counts |
| `spotify` | Spotify | 0-100 | Search for associated music (composer/production) popularity |

**Using the Title Scoring Helper:**

```bash
streamlit run title_scoring_helper.py
```

Enter your titles, optionally add API keys for YouTube/Spotify, and click "Run Scoring" to fetch normalized 0-100 scores.

### Step 3: Add to baselines.csv

Add rows to `data/productions/baselines.csv` with this format:

```csv
title,wiki,trends,youtube,spotify,category,gender,source,notes
My New Title,75,30,85,60,family_classic,female,external_reference,"Brief description"
```

**Categories to use:**
- `family_classic` - Nutcracker, Cinderella, Sleeping Beauty
- `classic_romance` - Sleeping Beauty, Romeo and Juliet
- `romantic_tragedy` - Swan Lake, Giselle, Romeo and Juliet
- `classic_comedy` - Coppélia, La Fille mal gardée, Don Quixote
- `contemporary` - Modern abstract/neoclassical works
- `contemporary_mixed_bill` - Mixed bill programs
- `adult_literary_drama` - Story ballets for mature audiences
- `pop_ip` - Productions with pop culture tie-ins
- `touring_contemporary_company` - Guest company productions

**Gender values:**
- `female` - Female protagonist/focus
- `male` - Male protagonist/focus
- `co` - Ensemble or co-lead

**Source values:**
- `historical` - Has Alberta Ballet ticket sales data (use for titles you have actual outcomes for)
- `external_reference` - External reference title (use for well-known titles without AB ticket history)

### Step 4: Validate Your Additions

Run the validation script to check data quality:

```python
from data.loader import load_all_baselines

# Load all baselines including reference
all_baselines = load_all_baselines(include_reference=True)

# Check for issues
print(f"Total titles: {len(all_baselines)}")
print(f"Historical: {(all_baselines['source'] == 'historical').sum()}")
print(f"Reference: {(all_baselines['source'] == 'external_reference').sum()}")

# Check for duplicates
duplicates = all_baselines[all_baselines.duplicated(subset=['title'], keep=False)]
if len(duplicates) > 0:
    print(f"Warning: Found {len(duplicates)} duplicate titles")
    print(duplicates[['title', 'source']])

# Check signal ranges
for col in ['wiki', 'trends', 'youtube', 'spotify']:
    out_of_range = all_baselines[(all_baselines[col] < 0) | (all_baselines[col] > 100)]
    if len(out_of_range) > 0:
        print(f"Warning: {len(out_of_range)} titles have {col} outside 0-100 range")
```

### Step 5: Use Reference Baselines in Predictions

The k-NN fallback automatically uses reference baselines for similarity matching:

```python
from ml.knn_fallback import find_similar_titles
from data.loader import load_all_baselines

# Load all baselines
all_baselines = load_all_baselines(include_reference=True)

# Find similar titles for a new show
new_show = {"wiki": 70, "trends": 25, "youtube": 80, "spotify": 55}
similar = find_similar_titles(new_show, all_baselines, k=5)

print("Most similar titles:")
for _, row in similar.iterrows():
    print(f"  {row['title']} ({row['category']}) - similarity: {row['similarity']:.2f}")
```

## Best Practices

### Signal Quality

1. **Be consistent**: Use the same methodology for all titles
2. **Update periodically**: Signals change over time, especially trends
3. **Document sources**: Note in the `notes` column where data came from
4. **Cross-validate**: Compare with known titles to ensure reasonable values

### Category Balance

Ensure each category has adequate representation:

```python
category_counts = all_baselines.groupby(['category', 'source']).size().unstack(fill_value=0)
print(category_counts)
```

Aim for at least 5-10 reference titles per category for good k-NN matching.

### Avoiding Bias

- Include both high-signal and low-signal titles per category
- Mix iconic classics with lesser-known works
- Consider geographic/cultural diversity in selections

## Frequently Asked Questions

### Q: Do reference baselines affect the trained model?

**A**: No, reference baselines are NOT used for training the ML model (which requires actual ticket outcomes). They only help with k-NN similarity matching for cold-start predictions.

### Q: How do I know if a title is "reference" vs "historical"?

**A**: The `source` column distinguishes them:
- `historical` = Has Alberta Ballet ticket data
- `external_reference` = External reference title (no AB ticket history)

### Q: Can I add titles that Alberta Ballet has performed but I don't have ticket data for?

**A**: Yes! Add them to `baselines.csv` with `source="external_reference"`. Once you get ticket data, change the source to `"historical"`.

### Q: How many reference baselines should I add?

**A**: More is generally better for similarity matching, but focus on quality over quantity. Start with 50-100 well-researched titles, then expand as needed.

### Q: What if my signal estimates are wrong?

**A**: The k-NN algorithm is robust to some noise. Focus on getting the relative rankings correct (high vs. medium vs. low signals) rather than exact values.

## Example: Adding a New Title

Let's walk through adding "The Nutcracker" as a reference baseline:

1. **Gather signals**:
   - `wiki`: 95 (extremely high Wikipedia traffic during holiday season)
   - `trends`: 52 (strong seasonal search interest in Alberta)
   - `youtube`: 100 (many high-view professional recordings)
   - `spotify`: 82 (Tchaikovsky's score is widely streamed)

2. **Determine category**: `family_classic` (quintessential family holiday show)

3. **Determine gender**: `co` (ensemble piece with both Clara and Nutcracker Prince)

4. **Add to CSV**:
   ```csv
   The Nutcracker,95,52,100,82,family_classic,co,external_reference,"Classic Christmas ballet - highest demand title"
   ```

5. **Validate**: Run the validation script to confirm it loaded correctly.

---

*Last updated: November 2024*
