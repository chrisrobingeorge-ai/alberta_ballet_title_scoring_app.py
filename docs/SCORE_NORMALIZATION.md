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

Distribution alignment using statistical transformation:

```
# Step 1: Calculate z-scores using NEW data statistics
z_new = (new_score - new_mean) / new_std

# Step 2: Rescale using BASELINE statistics
aligned = baseline_mean + (z_new * baseline_std)
```

This ensures aligned scores have the same statistical distribution (mean, std) as baselines.

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
2. **Load export file**: Read new scores to align
3. **Calculate export statistics**: Compute mean and std from the new export data
4. **Apply distribution alignment**: Transform scores using both sets of statistics
   - Convert to z-scores using export mean/std
   - Rescale using baseline mean/std
5. **Preserve metadata**: Keep all non-signal columns unchanged
6. **Save output**: Write aligned scores to CSV

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

### Values are very different

**Note**: After alignment, values will likely be quite different from the original export. This is expected! The transformation maps scores from the export's distribution to the baseline's distribution. If your new scores were systematically higher/lower than baseline, aligned scores will be adjusted accordingly.

## Testing

Run the test suite to verify functionality:

```bash
# Run all normalization tests
python -m pytest tests/test_normalize_export_scores.py -v

# Run specific test
python -m pytest tests/test_normalize_export_scores.py::test_normalize_scores_zscore_calculation -v
```

## Technical Details

### Distribution Alignment Formula

```python
# Step 1: Calculate export data statistics
export_mean = export_scores.mean()
export_std = export_scores.std()

# Step 2: Transform each score
z_new = (new_score - export_mean) / export_std
aligned_score = baseline_mean + (z_new * baseline_std)
```

### Why This Approach?

The two-step transformation ensures:
1. **Preserves relative ordering**: Highest new score → highest aligned score
2. **Matches baseline distribution**: Aligned scores have same mean/std as baseline
3. **Handles calibration drift**: Adjusts for systematic bias in new measurements
4. **Scale-invariant**: Works even if new scores use different absolute ranges

### Example Transformation

```
New scores:     [85, 90, 95]  (mean=90, std=5)
Baseline stats:              (mean=60, std=15)

Aligned scores: [45, 60, 75]  (mean=60, std=15) ✓
```

The middle value (90) was at the new mean, so it maps to the baseline mean (60).
The spread is preserved proportionally.

### Alternative Approaches

Other normalization methods not used:
- **Simple z-scores**: Would output standardized values (mean=0, std=1), not 0-100 scale
- **Min-Max scaling**: Sensitive to outliers and doesn't preserve distribution shape
- **Robust scaling**: Less interpretable for domain experts
- **Quantile transformation**: Changes distribution shape entirely

Distribution alignment preserves relative relationships while matching baseline scale.

## See Also

- [Baseline Data Documentation](ADDING_BASELINE_TITLES.md)
- [ML Model Documentation](ML_MODEL_DOCUMENTATION.md)
- [Title Scoring Helper](../title_scoring_helper.py)
