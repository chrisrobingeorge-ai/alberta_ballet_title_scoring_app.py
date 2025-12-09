# Score Normalization Guide

## Overview

The `normalize_export_scores.py` script aligns new external signal scores (wiki, trends, youtube, spotify) with the baseline calibration set using z-score normalization.

## Problem

When fetching new scores from APIs over time, even though they're on the same 0-100 scale, they may have different statistical properties than the baseline:

- **Different means**: New scores might be systematically higher or lower
- **Different variance**: New scores might be more/less spread out
- **Temporal drift**: API algorithms or data availability changes over time

This makes direct comparison invalid and can degrade ML model performance.

## Solution

Z-score normalization using baseline statistics as the reference distribution:

```
z = (new_score - baseline_mean) / baseline_std
normalized = baseline_mean + (z * baseline_std)
```

This ensures new scores have the same statistical distribution as baselines.

## Usage

### Basic Usage

```bash
python scripts/normalize_export_scores.py \
  --baselines data/productions/baselines.csv \
  --export 2025_export.csv \
  --output 2025_export_normalized.csv
```

### Custom Signal Columns

```bash
python scripts/normalize_export_scores.py \
  --baselines data/productions/baselines.csv \
  --export 2025_export.csv \
  --output 2025_export_normalized.csv \
  --signals wiki trends youtube spotify custom_signal
```

### Custom Title Column

```bash
python scripts/normalize_export_scores.py \
  --baselines data/productions/baselines.csv \
  --export 2025_export.csv \
  --output 2025_export_normalized.csv \
  --title-column show_title
```

## Input File Requirements

### Baseline File (`baselines.csv`)

Must contain:
- A title column (default: `title`)
- Signal columns to normalize (default: `wiki`, `trends`, `youtube`, `spotify`)
- At least 30+ titles for stable statistics (current: 288 titles)

Example:
```csv
title,wiki,trends,youtube,spotify,category,source
Swan Lake,85,30,90,70,classical,external_reference
The Nutcracker,90,40,95,80,classical,historical
...
```

### Export File (e.g., `2025_export.csv`)

Must contain:
- A title column matching the baseline format
- Signal columns to normalize (subset of baseline signals is OK)
- Can include additional columns (they will be preserved)

Example:
```csv
title,wiki,trends,youtube,spotify,category,notes
Swan Lake,88,32,92,75,classical,2025 revival
New Ballet,75,30,85,65,contemporary,World premiere
...
```

## Output

The script produces a normalized CSV with:
- All original columns preserved
- Signal columns normalized using baseline statistics
- Same row count as input (all titles preserved)
- Non-matched titles keep original scores

Example output:
```csv
title,wiki,trends,youtube,spotify,category,notes
Swan Lake,88.0,32.0,92.0,75.0,classical,2025 revival
New Ballet,75.0,30.0,85.0,65.0,contemporary,World premiere
...
```

## Baseline Statistics (Current)

Using all 288 titles (historical + external reference):

| Signal  | Mean  | Std Dev |
|---------|-------|---------|
| wiki    | 57.05 | 17.90   |
| trends  | 25.48 | 16.65   |
| youtube | 65.79 | 14.54   |
| spotify | 53.67 | 19.22   |

## How It Works

1. **Load baseline statistics**: Calculate mean and std for each signal from baselines.csv
2. **Load export file**: Read new scores to normalize
3. **Apply z-score normalization**: Transform each score using baseline statistics
4. **Preserve metadata**: Keep all non-signal columns unchanged
5. **Save output**: Write normalized scores to CSV

## When to Use

Use this script when:
- ✅ Fetching new scores from external APIs
- ✅ Comparing titles across different time periods
- ✅ Adding new titles to ML training datasets
- ✅ Updating the production recommendation system

Don't use this script when:
- ❌ Working with raw API responses (normalize after fetching)
- ❌ The baseline statistics have fundamentally changed (rebuild baselines first)
- ❌ Scores are already aligned to the baseline distribution

## Example Workflow

```bash
# 1. Fetch new scores from APIs
python scripts/fetch_external_signals.py --output 2025_export.csv

# 2. Normalize against baselines
python scripts/normalize_export_scores.py \
  --baselines data/productions/baselines.csv \
  --export 2025_export.csv \
  --output 2025_export_normalized.csv

# 3. Use normalized scores for analysis/modeling
python scripts/analyze_title_demand.py \
  --input 2025_export_normalized.csv
```

## Adapting to Other Signals

To normalize additional signal columns:

1. Ensure the baseline file has those columns
2. Add them to the `--signals` argument:

```bash
python scripts/normalize_export_scores.py \
  --baselines data/productions/baselines.csv \
  --export 2025_export.csv \
  --output 2025_export_normalized.csv \
  --signals wiki trends youtube spotify tiktok instagram
```

## Troubleshooting

### Missing signal columns

**Error**: `Missing signal columns in baseline file: ['tiktok']`

**Solution**: Either remove the signal from `--signals` or add it to your baseline file.

### Empty output

**Error**: Output file has 0 rows

**Solution**: Check that your export file has the correct title column name. Use `--title-column` if different from default.

### Values seem unchanged

**Note**: After z-score normalization and rescaling with the same statistics, values may appear unchanged. This is correct! The transformation aligns the distribution statistically, even if individual values look similar.

## Testing

Run the test suite to verify functionality:

```bash
# Run all normalization tests
python -m pytest tests/test_normalize_export_scores.py -v

# Run specific test
python -m pytest tests/test_normalize_export_scores.py::test_normalize_scores_zscore_calculation -v
```

## Technical Details

### Z-Score Normalization Formula

```python
z = (value - baseline_mean) / baseline_std
normalized = baseline_mean + (z * baseline_std)
```

### Why Rescale Back?

The rescaling step (`normalized = mean + z*std`) ensures:
1. Scores remain on the familiar 0-100 scale
2. Distribution matches baseline distribution
3. Easy comparison with historical data

### Alternative Approaches

Other normalization methods not used:
- **Min-Max scaling**: Sensitive to outliers
- **Robust scaling**: Less interpretable
- **Quantile transformation**: Changes distribution shape

Z-score normalization preserves distribution shape while aligning scale.

## See Also

- [Baseline Data Documentation](ADDING_BASELINE_TITLES.md)
- [ML Model Documentation](ML_MODEL_DOCUMENTATION.md)
- [Title Scoring Helper](../title_scoring_helper.py)
