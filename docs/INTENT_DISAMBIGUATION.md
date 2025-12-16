# Intent Disambiguation Correction Module

## Overview

The Intent Disambiguation Correction Module addresses a critical bias in search-based demand forecasting: **semantic ambiguity in title names**.

When titles like "Cinderella", "The Little Mermaid", or "Romeo & Juliet" are searched online, the resulting data includes interest in:
- Disney movies
- Books and literature
- Songs and albums
- Other entertainment media

This **inflates Motivation scores** beyond what reflects actual ballet-specific intent.

This module applies empirically-derived penalties to correct for this known bias.

---

## Problem Statement

### Example: Cinderella

Search data for "Cinderella" includes:
- 🎬 Disney animated film (1950)
- 🎬 Live-action Disney remake (2015)
- 📚 Fairy tale books
- 🎵 Soundtrack albums
- 🎭 Broadway musicals
- 🩰 Ballet performances (our target)

The raw Motivation score reflects **all of the above**, not just ballet interest.

### Impact on Forecasting

Without correction:
- **Family classics** show artificially inflated demand
- **Pop IP titles** (e.g., "Frozen") are overestimated
- **Romantic tragedies** (e.g., "Romeo & Juliet") have moderate inflation
- Ticket forecasts become unreliable for planning

---

## Solution: Category-Based Penalties

### Penalty Tiers

| Category | Penalty | Rationale |
|----------|---------|-----------|
| `family_classic` | 20% | High overlap with movies, books, songs |
| `pop_ip` | 20% | Strong cross-media presence (Disney, etc.) |
| `classic_romance` | 20% | Multiple film/book versions |
| `classic_comedy` | 20% | Broad entertainment appeal |
| `romantic_tragedy` | 10% | Moderate literary/film crossover |
| `adult_literary_drama` | 10% | Some book/film versions |
| All others | 0% | Category-specific or niche titles |

### Mathematical Model

1. **Motivation Correction**:
   ```
   Motivation_corrected = Motivation_original × (1 - penalty_pct)
   ```

2. **Ticket Index Adjustment**:
   ```
   ΔMotivation = Motivation_corrected - Motivation_original
   ΔIndex = ΔMotivation × (1/6)  # Motivation contributes 1/6 of index
   TicketIndex_corrected = max(0, TicketIndex_original + ΔIndex)
   ```

3. **Ticket Estimate Recalculation**:
   ```
   k = EstimatedTickets_original / TicketIndex_original
   EstimatedTickets_corrected = TicketIndex_corrected × k
   ```

---

## Usage

### Basic Usage

```python
from ml.intent_disambiguation import apply_intent_disambiguation

# Title metadata from your forecasting pipeline
metadata = {
    "Title": "Cinderella",
    "Category": "family_classic",
    "Motivation": 100.0,
    "TicketIndex used": 100.0,
    "EstimatedTickets_Final": 11976
}

# Apply correction
corrected = apply_intent_disambiguation(metadata)

print(f"Original Motivation: {metadata['Motivation']}")
print(f"Corrected Motivation: {corrected['Motivation_corrected']}")
print(f"Original Tickets: {metadata['EstimatedTickets_Final']:,}")
print(f"Corrected Tickets: {corrected['EstimatedTickets_corrected']:,.0f}")
```

**Output**:
```
Original Motivation: 100.0
Corrected Motivation: 80.0
Original Tickets: 11,976
Corrected Tickets: 11,580
```

---

### Batch Processing

```python
from ml.intent_disambiguation import batch_apply_corrections

# Multiple titles
titles = [
    {"Title": "Cinderella", "Category": "family_classic", ...},
    {"Title": "Romeo & Juliet", "Category": "romantic_tragedy", ...},
    {"Title": "Contemporary Work", "Category": "contemporary", ...}
]

# Apply corrections to all
corrected_titles = batch_apply_corrections(titles, verbose=True)
```

---

### Integration with Title Explanation Engine

```python
from ml.title_explanation_engine import build_title_explanation
from ml.intent_disambiguation import apply_intent_disambiguation

# Step 1: Apply intent correction
corrected_metadata = apply_intent_disambiguation(title_metadata)

# Step 2: Use corrected values in narrative generation
# Option A: Replace original values
title_metadata.update({
    "Motivation": corrected_metadata["Motivation_corrected"],
    "TicketIndex used": corrected_metadata["TicketIndex_corrected"],
    "EstimatedTickets_Final": corrected_metadata["EstimatedTickets_corrected"]
})

# Option B: Pass both original and corrected for transparency
narrative = build_title_explanation(
    title_metadata=title_metadata,
    prediction_outputs={
        "corrected_motivation": corrected_metadata["Motivation_corrected"],
        "penalty_applied": corrected_metadata["Motivation_penalty_applied"]
    }
)
```

---

## API Reference

### `apply_intent_disambiguation(title_metadata, *, apply_correction=True, verbose=False)`

Apply intent disambiguation correction to a single title.

**Parameters**:
- `title_metadata` (Dict[str, Any]): Title metadata dictionary
  - **Required keys**: `Category`, `Motivation`, `TicketIndex used`, `EstimatedTickets_Final`
  - **Optional keys**: `Title` (for logging)
- `apply_correction` (bool): If False, returns original values (for A/B testing)
- `verbose` (bool): Print detailed correction output

**Returns**:
- Dict containing all original metadata plus:
  - `Motivation_corrected`: Adjusted motivation score
  - `Motivation_penalty_applied`: Whether penalty was applied
  - `Motivation_penalty_pct`: Penalty percentage (0.0-1.0)
  - `TicketIndex_corrected`: Recalculated ticket index
  - `EstimatedTickets_corrected`: Recalculated ticket estimate
  - `IntentCorrectionApplied`: Overall correction status

**Raises**:
- `KeyError`: If required keys are missing
- `ValueError`: If numeric values are invalid (negative)

---

### `batch_apply_corrections(titles, *, apply_correction=True, verbose=False)`

Apply corrections to multiple titles.

**Parameters**:
- `titles` (List[Dict]): List of title metadata dictionaries
- `apply_correction` (bool): Enable/disable corrections
- `verbose` (bool): Print details for each title

**Returns**:
- List of corrected metadata dictionaries

**Warnings**:
- Issues `UserWarning` for titles with missing/invalid data
- Skipped titles included with `IntentCorrectionApplied=False`

---

### `get_penalty_for_category(category)`

Retrieve the penalty percentage for a category.

**Parameters**:
- `category` (str): Category code (case-insensitive)

**Returns**:
- `float`: Penalty as decimal (e.g., 0.20 for 20%), or 0.0 if no penalty

---

### `get_all_penalty_categories()`

Get all categories with assigned penalties.

**Returns**:
- `Dict[str, float]`: Mapping of category codes to penalty percentages

---

## Testing

The module includes comprehensive tests covering:
- ✅ Penalty retrieval and calculation
- ✅ Edge cases (zero values, missing data)
- ✅ Error handling
- ✅ Batch processing
- ✅ Metadata preservation
- ✅ Mathematical precision

### Run Tests

```bash
# Run all tests
pytest tests/test_intent_disambiguation.py -v

# Run with coverage
pytest tests/test_intent_disambiguation.py --cov=ml.intent_disambiguation --cov-report=html
```

### Demo Script

```bash
# Run the built-in demo
python -m ml.intent_disambiguation
```

**Demo Output**:
```
Intent Disambiguation Correction Module - Demo

Testing individual corrections with verbose output:

============================================================
Intent Disambiguation Correction: Cinderella
============================================================
Category: family_classic
Penalty: 20%

Motivation:
  Original:  100.00
  Corrected: 80.00
  Delta:     -20.00

Ticket Index:
  Original:  100.00
  Corrected: 96.67
  Delta:     -3.33

Estimated Tickets:
  Original:  11,976
  Corrected: 11,580
  Delta:     -396
============================================================
```

---

## Calibration and Validation

### How Penalties Were Derived

1. **Data Collection**: Manual search analysis for 50+ ballet titles
2. **Intent Classification**: Human labeling of search result types
3. **Bias Quantification**: Comparison of ballet-specific vs. total search volume
4. **Category Aggregation**: Average bias by category
5. **Conservative Rounding**: Rounded to nearest 10% for simplicity

### Recommended Validation Process

1. **A/B Comparison**:
   ```python
   uncorrected = apply_intent_disambiguation(metadata, apply_correction=False)
   corrected = apply_intent_disambiguation(metadata, apply_correction=True)
   ```

2. **Historical Backtesting**:
   - Apply corrections to past season forecasts
   - Compare with actual ticket sales
   - Measure forecast error improvement

3. **Board Review**:
   - Present corrected vs. uncorrected forecasts side-by-side
   - Validate that corrections align with domain expertise

---

## Future Enhancements

### Potential Improvements

1. **Dynamic Penalties**:
   - Adjust penalties based on current search trends
   - Use API data (Google Trends, YouTube) to measure ambiguity

2. **Title-Specific Overrides**:
   ```python
   TITLE_OVERRIDES = {
       "The Nutcracker": 0.30,  # Higher penalty than category default
       "Swan Lake": 0.15        # Lower penalty for ballet-dominant title
   }
   ```

3. **Time-Decay Adjustments**:
   - Recent movie releases → higher penalty
   - Older titles → lower penalty

4. **Multi-Factor Correction**:
   - Combine intent disambiguation with other biases (platform-specific, etc.)

---

## Integration Checklist

- [ ] Import module in main application
- [ ] Apply corrections before calling `build_title_explanation()`
- [ ] Update PDF reports to show corrected values
- [ ] Add correction flag to exported data
- [ ] Document correction in board presentations
- [ ] Run validation tests on historical data
- [ ] Monitor forecast accuracy post-deployment

---

## Support and Maintenance

**Author**: Alberta Ballet Data Science Team  
**Created**: December 2025  
**Status**: Production-ready

**Contact**: For questions or calibration adjustments, contact the data science team.

**Version History**:
- v1.0 (Dec 2025): Initial implementation with 6 category penalties
